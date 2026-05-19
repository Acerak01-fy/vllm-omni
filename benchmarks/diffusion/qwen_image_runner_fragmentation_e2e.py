# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-level Qwen-Image Cache-DiT fragmentation experiment.

This benchmark uses the real ``DiffusionModelRunner`` with Qwen-Image and the
original contiguous Cache-DiT backend. It does not use paged cache. The traffic
trace is synthesized from Qwen-Image Dataset C and replayed at runner level so
we can keep request state resident while other requests arrive, suspend, resume,
finish, or get released early.

The script intentionally avoids the HTTP serving benchmark path because the
CUDA allocator snapshot must be collected in the same process that owns the
runner and its Cache-DiT tensors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import DiffusionCacheConfig, DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

MIB = 1024 * 1024


@dataclass(frozen=True)
class Profile:
    name: str
    width: int
    height: int
    num_inference_steps: int
    weight: float


@dataclass(frozen=True)
class RequestSpec:
    req_id: str
    profile_name: str
    width: int
    height: int
    num_inference_steps: int
    arrival_tick: int
    seed: int
    prompt: str


@dataclass
class RequestRecord:
    req_id: str
    profile_name: str
    width: int
    height: int
    num_inference_steps: int
    arrival_tick: int
    first_run_tick: int | None = None
    finish_tick: int | None = None
    release_tick: int | None = None
    executed_steps: int = 0
    status: str = "pending"
    error: str = ""


@dataclass
class SnapshotRow:
    tick: int
    action: str
    req_id: str
    profile_name: str
    step_index: int
    total_steps: int
    live_reqs: int
    pending_reqs: int
    finished_reqs: int
    released_reqs: int
    resident_cache_mib: float
    total_inactive_mib: float
    largest_inactive_mib: float
    frag_ratio: float
    reserved_mib: float
    allocated_mib: float
    inactive_split_mib: float
    global_free_mib: float
    latency_s: float
    note: str


@dataclass
class ProbeSweepRow:
    probe_index: int
    requested_mib: float
    status: str
    pre_total_inactive_mib: float
    pre_largest_inactive_mib: float
    pre_frag_ratio: float
    pre_reserved_mib: float
    pre_allocated_mib: float
    pre_inactive_split_mib: float
    pre_global_free_mib: float
    post_total_inactive_mib: float
    post_largest_inactive_mib: float
    post_frag_ratio: float
    post_reserved_mib: float
    post_allocated_mib: float
    post_inactive_split_mib: float
    post_global_free_mib: float
    error: str


QWEN_IMAGE_DATASET_C = [
    Profile("qwen_c_512_20", width=512, height=512, num_inference_steps=20, weight=0.15),
    Profile("qwen_c_768_20", width=768, height=768, num_inference_steps=20, weight=0.25),
    Profile("qwen_c_1024_25", width=1024, height=1024, num_inference_steps=25, weight=0.45),
    Profile("qwen_c_1536_35", width=1536, height=1536, num_inference_steps=35, weight=0.15),
]

PROMPTS = [
    "a ceramic teapot on a wooden table, soft studio lighting",
    "a futuristic city skyline at sunset with clean architectural details",
    "a bowl of strawberries beside a glass of water, realistic texture",
    "a small cabin in snowy mountains, clear sky, cinematic composition",
    "a watercolor painting of a mountain lake with reflections",
    "a vintage camera on a desk with notebooks and pencils",
    "a robot gardener watering flowers in a greenhouse",
    "a minimalist product photo of running shoes on a white background",
    "a cozy reading corner with warm lamp light and plants",
    "a detailed fantasy castle built on coastal cliffs",
]


def _allocator_snapshot(device_index: int) -> dict[str, float]:
    free_blocks: list[int] = []
    active_bytes = 0
    for segment in torch.cuda.memory_snapshot():
        if "device" in segment and int(segment["device"]) != device_index:
            continue
        for block in segment.get("blocks", []):
            size = int(block.get("size", 0))
            state = str(block.get("state", ""))
            if state == "inactive":
                free_blocks.append(size)
            elif state.startswith("active"):
                active_bytes += size

    total_free = float(sum(free_blocks))
    largest_free = float(max(free_blocks) if free_blocks else 0)
    frag_ratio = 0.0 if total_free <= 0 else 1.0 - largest_free / total_free
    stats = torch.cuda.memory_stats(device_index)
    global_free, _ = torch.cuda.mem_get_info(device_index)
    return {
        "total_inactive_mib": total_free / MIB,
        "largest_inactive_mib": largest_free / MIB,
        "frag_ratio": frag_ratio,
        "reserved_mib": float(stats.get("reserved_bytes.all.current", 0)) / MIB,
        "allocated_mib": float(stats.get("allocated_bytes.all.current", 0)) / MIB,
        "inactive_split_mib": float(stats.get("inactive_split_bytes.all.current", 0)) / MIB,
        "global_free_mib": float(global_free) / MIB,
        "active_snapshot_mib": active_bytes / MIB,
        "num_free_blocks": float(len(free_blocks)),
    }


def _allocate_tensor_mib(device: torch.device, mib: float) -> torch.Tensor:
    num_bytes = max(1, int(mib * MIB))
    numel = max(1, (num_bytes + 1) // 2)
    return torch.empty((numel,), dtype=torch.float16, device=device)


def _allocate_pressure_to_target(
    *,
    device: torch.device,
    device_index: int,
    target_global_free_mib: float,
    min_pressure_mib: float,
) -> tuple[list[torch.Tensor], float, str]:
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    free_mib = free_bytes / MIB
    requested_mib = free_mib - target_global_free_mib
    if requested_mib <= 0:
        return (
            [],
            0.0,
            (
                f"Skipped pressure allocation because global_free={free_mib:.2f} MiB "
                f"is already <= target_global_free={target_global_free_mib:.2f} MiB."
            ),
        )

    requested_mib = max(0.0, requested_mib)
    last_error = ""
    while requested_mib >= min_pressure_mib:
        try:
            tensor = _allocate_tensor_mib(device, requested_mib)
            torch.cuda.synchronize(device_index)
            after_free_mib = torch.cuda.mem_get_info(device_index)[0] / MIB
            return (
                [tensor],
                requested_mib,
                (
                    f"Allocated pressure tensor {requested_mib:.2f} MiB; "
                    f"global_free_before={free_mib:.2f} MiB; "
                    f"global_free_after={after_free_mib:.2f} MiB."
                ),
            )
        except torch.cuda.OutOfMemoryError as exc:
            last_error = str(exc).splitlines()[0]
            requested_mib *= 0.85

    return (
        [],
        0.0,
        (
            f"Failed to allocate pressure tensor down to min_pressure={min_pressure_mib:.2f} MiB; "
            f"last_error={last_error}"
        ),
    )


def _write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _parse_mib_list(value: str) -> list[float]:
    sizes: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        sizes.append(float(item))
    if not sizes:
        raise ValueError("--probe-sweep-mib must contain at least one MiB value.")
    return sizes


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _write_svg_line_chart(
    path: Path,
    *,
    title: str,
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    if not x_values:
        return
    width, height = 900, 460
    left, right, top, bottom = 78, 24, 52, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    if y_max is None:
        max_seen = max(max(values) for _, values, _ in series if values)
        y_max = max(max_seen * 1.08, 1.0)
    if y_max <= y_min:
        y_max = y_min + 1.0
    x_min, x_max = min(x_values), max(x_values)
    x_span = max(x_max - x_min, 1)

    def sx(x: float) -> float:
        return left + ((x - x_min) / x_span) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".label{font-size:12px}",
        ".tick{font-size:11px;fill:#555}",
        ".grid{stroke:#d0d7de;stroke-width:1;opacity:.7}",
        ".axis{stroke:#24292f;stroke-width:1.2}",
        "</style>",
        f'<text class="title" x="{width / 2:.1f}" y="26" text-anchor="middle">{_xml_escape(title)}</text>',
        f'<text class="label" x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle">event tick</text>',
        (
            f'<text class="label" transform="translate(18 {top + plot_h / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle">{_xml_escape(y_label)}</text>'
        ),
    ]
    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')
    tick_stride = max(1, len(x_values) // 12)
    for x in x_values[::tick_stride]:
        px = sx(x)
        parts.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{px:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{x}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')

    legend_x, legend_y = left + 12, top + 18
    for idx, (label, values, color) in enumerate(series):
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values, strict=True))
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" points="{points}"/>'
        )
        ly = legend_y + idx * 20
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(f'<text class="label" x="{legend_x + 32}" y="{ly + 4}">{_xml_escape(label)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _make_trace(args: argparse.Namespace) -> list[RequestSpec]:
    rng = random.Random(args.seed)
    profiles = QWEN_IMAGE_DATASET_C
    weights = [profile.weight for profile in profiles]
    arrival_time = 0.0
    trace: list[RequestSpec] = []
    for idx in range(args.num_requests):
        if idx > 0:
            arrival_time += rng.expovariate(args.arrival_rate)
        profile = rng.choices(profiles, weights=weights, k=1)[0]
        trace.append(
            RequestSpec(
                req_id=f"qwen-c-{idx:03d}",
                profile_name=profile.name,
                width=profile.width,
                height=profile.height,
                num_inference_steps=profile.num_inference_steps,
                arrival_tick=int(round(arrival_time)),
                seed=args.seed * 1000 + idx,
                prompt=PROMPTS[idx % len(PROMPTS)],
            )
        )
    return trace


def _make_request(spec: RequestSpec, device: torch.device) -> OmniDiffusionRequest:
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    sampling = OmniDiffusionSamplingParams(
        width=spec.width,
        height=spec.height,
        num_inference_steps=spec.num_inference_steps,
        guidance_scale=0.0,
        guidance_scale_provided=True,
        true_cfg_scale=1.0,
        generator=generator,
        seed=spec.seed,
        generator_device=str(device),
        num_outputs_per_prompt=1,
    )
    return OmniDiffusionRequest(
        prompts=[{"prompt": spec.prompt, "negative_prompt": "blurry, low quality"}],
        sampling_params=sampling,
        request_ids=[spec.req_id],
    )


def _make_sched_output(
    *,
    step_id: int,
    new_req: tuple[str, OmniDiffusionRequest] | None = None,
    cached_req_id: str | None = None,
) -> DiffusionSchedulerOutput:
    new_reqs = []
    if new_req is not None:
        sched_req_id, req = new_req
        new_reqs.append(NewRequestData(sched_req_id=sched_req_id, req=req))
    cached = CachedRequestData.make_empty() if cached_req_id is None else CachedRequestData([cached_req_id])
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs) + len(cached.sched_req_ids),
        num_waiting_reqs=0,
    )


def _resident_cache_mib(runner: DiffusionModelRunner) -> float:
    total = 0
    for state in runner.state_cache.values():
        slot = state.cache_slot
        if slot is not None:
            total += int(slot.resident_bytes)
    return total / MIB


def _append_snapshot(
    rows: list[SnapshotRow],
    *,
    tick: int,
    action: str,
    req_id: str,
    profile_name: str,
    step_index: int,
    total_steps: int,
    records: dict[str, RequestRecord],
    runner: DiffusionModelRunner,
    device_index: int,
    latency_s: float = 0.0,
    note: str = "",
) -> None:
    gc.collect()
    torch.cuda.synchronize(device_index)
    snap = _allocator_snapshot(device_index)
    live = sum(1 for record in records.values() if record.status == "live")
    pending = sum(1 for record in records.values() if record.status == "pending")
    finished = sum(1 for record in records.values() if record.status == "finished")
    released = sum(1 for record in records.values() if record.status == "released")
    rows.append(
        SnapshotRow(
            tick=tick,
            action=action,
            req_id=req_id,
            profile_name=profile_name,
            step_index=step_index,
            total_steps=total_steps,
            live_reqs=live,
            pending_reqs=pending,
            finished_reqs=finished,
            released_reqs=released,
            resident_cache_mib=_resident_cache_mib(runner),
            total_inactive_mib=snap["total_inactive_mib"],
            largest_inactive_mib=snap["largest_inactive_mib"],
            frag_ratio=snap["frag_ratio"],
            reserved_mib=snap["reserved_mib"],
            allocated_mib=snap["allocated_mib"],
            inactive_split_mib=snap["inactive_split_mib"],
            global_free_mib=snap["global_free_mib"],
            latency_s=latency_s,
            note=note,
        )
    )


def _init_runner(args: argparse.Namespace) -> tuple[DiffusionModelRunner, torch.device, int]:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ["LOCAL_RANK"] = str(args.device_index)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    device = current_omni_platform.get_torch_device(args.device_index)
    current_omni_platform.set_device(device)

    cache_config = DiffusionCacheConfig(
        Fn_compute_blocks=args.fn_compute_blocks,
        Bn_compute_blocks=args.bn_compute_blocks,
        max_warmup_steps=args.max_warmup_steps,
        residual_diff_threshold=args.residual_diff_threshold,
        max_continuous_cached_steps=args.max_continuous_cached_steps,
        enable_paged_cache=False,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        model_class_name="QwenImagePipeline",
        dtype=args.dtype,
        cache_backend="cache_dit",
        cache_config=cache_config,
        step_execution=True,
        enforce_eager=True,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cache_dit_summary=False,
        max_num_seqs=1,
        master_port=args.master_port,
    )
    vllm_config = VllmConfig(compilation_config=CompilationConfig())

    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        init_distributed_environment(world_size=1, rank=0)
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            fully_shard_degree=1,
            hsdp_replicate_size=1,
            enable_expert_parallel=False,
        )
        init_workspace_manager(device)
        runner = DiffusionModelRunner(vllm_config=vllm_config, od_config=od_config, device=device)
        runner.load_model(load_format="default")

    return runner, device, args.device_index


def _release_request(
    runner: DiffusionModelRunner,
    records: dict[str, RequestRecord],
    req_id: str,
    tick: int,
) -> None:
    state = runner.state_cache.pop(req_id, None)
    if state is not None and runner.cache_manager is not None:
        runner.cache_manager.free(state)
    record = records[req_id]
    record.status = "released"
    record.release_tick = tick


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _make_trace(args)
    records = {
        spec.req_id: RequestRecord(
            req_id=spec.req_id,
            profile_name=spec.profile_name,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.num_inference_steps,
            arrival_tick=spec.arrival_tick,
        )
        for spec in trace
    }
    _write_csv(output_dir / "requests.csv", list(records.values()))
    (output_dir / "trace.json").write_text(json.dumps([asdict(spec) for spec in trace], indent=2), encoding="utf-8")

    runner, device, device_index = _init_runner(args)
    rng = random.Random(args.seed + 17)
    snapshots: list[SnapshotRow] = []
    completed_with_error = ""
    oom_observed = False
    probe_attempted = False
    probe_oom_observed = False
    probe_success = False
    pressure_allocated_mib = 0.0
    step_id = 0
    next_pending_idx = 0
    arrived_pending: list[RequestSpec] = []
    pressure_tensors: list[torch.Tensor] = []

    try:
        _append_snapshot(
            snapshots,
            tick=0,
            action="after_model_load",
            req_id="",
            profile_name="",
            step_index=0,
            total_steps=0,
            records=records,
            runner=runner,
            device_index=device_index,
            note="Qwen-Image loaded with cache_backend=cache_dit and original contiguous tensors.",
        )

        tick = 0
        while tick < args.max_ticks:
            while next_pending_idx < len(trace) and trace[next_pending_idx].arrival_tick <= tick:
                arrived_pending.append(trace[next_pending_idx])
                next_pending_idx += 1

            live_ids = list(runner.state_cache)
            releasable = [
                req_id for req_id in live_ids if records[req_id].executed_steps >= args.min_steps_before_release
            ]
            if releasable and len(live_ids) > args.min_live_requests and rng.random() < args.release_probability:
                req_id = rng.choice(releasable)
                _release_request(runner, records, req_id, tick)
                _append_snapshot(
                    snapshots,
                    tick=tick,
                    action="release_early",
                    req_id=req_id,
                    profile_name=records[req_id].profile_name,
                    step_index=records[req_id].executed_steps,
                    total_steps=records[req_id].num_inference_steps,
                    records=records,
                    runner=runner,
                    device_index=device_index,
                    note="Early release simulates cancellation/finished postprocessing freeing a resident cache slot.",
                )

            scheduled_spec: RequestSpec | None = None
            cached_req_id: str | None = None
            live_ids = list(runner.state_cache)
            should_start_new = arrived_pending and (
                len(live_ids) < args.target_live_requests or rng.random() < args.new_request_probability
            )
            if should_start_new:
                scheduled_spec = arrived_pending.pop(0)
                req = _make_request(scheduled_spec, device)
                sched_output = _make_sched_output(
                    step_id=step_id,
                    new_req=(scheduled_spec.req_id, req),
                )
                record = records[scheduled_spec.req_id]
                record.status = "live"
                record.first_run_tick = tick
            elif live_ids:
                cached_req_id = rng.choice(live_ids)
                sched_output = _make_sched_output(step_id=step_id, cached_req_id=cached_req_id)
            elif next_pending_idx >= len(trace):
                break
            else:
                tick += 1
                continue

            active_req_id = scheduled_spec.req_id if scheduled_spec is not None else cached_req_id
            assert active_req_id is not None
            active_record = records[active_req_id]
            t0 = time.perf_counter()
            try:
                output = runner.execute_stepwise(sched_output)
            except torch.cuda.OutOfMemoryError as exc:
                oom_observed = True
                completed_with_error = str(exc).splitlines()[0]
                _append_snapshot(
                    snapshots,
                    tick=tick,
                    action="oom",
                    req_id=active_req_id,
                    profile_name=active_record.profile_name,
                    step_index=active_record.executed_steps,
                    total_steps=active_record.num_inference_steps,
                    records=records,
                    runner=runner,
                    device_index=device_index,
                    latency_s=time.perf_counter() - t0,
                    note=completed_with_error,
                )
                break
            except Exception as exc:
                completed_with_error = f"{type(exc).__name__}: {exc}"
                raise

            latency_s = time.perf_counter() - t0
            active_record.executed_steps += 1
            action = "run_new_step" if scheduled_spec is not None else "run_cached_step"
            finished = bool(output.finished)
            if finished:
                active_record.status = "finished"
                active_record.finish_tick = tick
                action = "finish_request"

            step_index = active_record.executed_steps
            result = output.result
            note = ""
            if isinstance(result, DiffusionOutput) and result.error:
                note = result.error
                active_record.error = result.error

            _append_snapshot(
                snapshots,
                tick=tick,
                action=action,
                req_id=active_req_id,
                profile_name=active_record.profile_name,
                step_index=step_index,
                total_steps=active_record.num_inference_steps,
                records=records,
                runner=runner,
                device_index=device_index,
                latency_s=latency_s,
                note=note,
            )

            latest = snapshots[-1]
            should_probe_oom = (
                args.enable_oom_probe
                and not probe_attempted
                and next_pending_idx >= len(trace)
                and latest.frag_ratio >= args.probe_min_frag_ratio
                and latest.total_inactive_mib >= args.probe_min_total_inactive_mib
                and latest.largest_inactive_mib <= args.probe_max_largest_inactive_mib
            )
            if should_probe_oom:
                probe_attempted = True
                (
                    pressure_tensors,
                    pressure_allocated_mib,
                    pressure_note,
                ) = _allocate_pressure_to_target(
                    device=device,
                    device_index=device_index,
                    target_global_free_mib=args.probe_target_global_free_mib,
                    min_pressure_mib=args.probe_min_pressure_mib,
                )
                probe_tensor: torch.Tensor | None = None
                probe_note = (
                    f"OOM probe requested contiguous tensor {args.probe_allocation_mib:.2f} MiB after "
                    f"fragmented snapshot: total_inactive={latest.total_inactive_mib:.2f} MiB, "
                    f"largest_inactive={latest.largest_inactive_mib:.2f} MiB, "
                    f"frag_ratio={latest.frag_ratio:.4f}. {pressure_note}"
                )
                try:
                    probe_tensor = _allocate_tensor_mib(device, args.probe_allocation_mib)
                    torch.cuda.synchronize(device_index)
                    probe_success = True
                    _append_snapshot(
                        snapshots,
                        tick=tick,
                        action="oom_probe_success",
                        req_id=active_req_id,
                        profile_name=active_record.profile_name,
                        step_index=step_index,
                        total_steps=active_record.num_inference_steps,
                        records=records,
                        runner=runner,
                        device_index=device_index,
                        note=probe_note + " Probe allocation unexpectedly succeeded.",
                    )
                    if args.stop_after_probe:
                        break
                except torch.cuda.OutOfMemoryError as exc:
                    probe_oom_observed = True
                    oom_observed = True
                    completed_with_error = str(exc).splitlines()[0]
                    _append_snapshot(
                        snapshots,
                        tick=tick,
                        action="oom_probe_oom",
                        req_id=active_req_id,
                        profile_name=active_record.profile_name,
                        step_index=step_index,
                        total_steps=active_record.num_inference_steps,
                        records=records,
                        runner=runner,
                        device_index=device_index,
                        note=probe_note + f" Probe allocation failed with OOM: {completed_with_error}",
                    )
                    break
                finally:
                    del probe_tensor
                    if not probe_oom_observed:
                        pressure_tensors.clear()
                        gc.collect()
                        torch.cuda.empty_cache()

            step_id += 1
            tick += 1

            if all(record.status in {"finished", "released"} for record in records.values()):
                break
    finally:
        pressure_tensors.clear()
        for req_id in list(runner.state_cache):
            _release_request(runner, records, req_id, tick if "tick" in locals() else -1)
        gc.collect()
        torch.cuda.empty_cache()
        destroy_distributed_env()

    _write_csv(output_dir / "timeline.csv", snapshots)
    _write_csv(output_dir / "requests.csv", list(records.values()))

    frag_values = [row.frag_ratio for row in snapshots]
    max_frag_row = max(snapshots, key=lambda row: row.frag_ratio) if snapshots else None
    summary = {
        "model": args.model,
        "dataset": "qwen_image_dataset_c",
        "num_requests": args.num_requests,
        "seed": args.seed,
        "oom_observed": oom_observed,
        "error": completed_with_error,
        "oom_probe_attempted": probe_attempted,
        "oom_probe_oom_observed": probe_oom_observed,
        "oom_probe_success": probe_success,
        "oom_probe_allocation_mib": args.probe_allocation_mib,
        "oom_probe_pressure_allocated_mib": pressure_allocated_mib,
        "max_frag_ratio": max(frag_values) if frag_values else 0.0,
        "mean_frag_ratio": sum(frag_values) / len(frag_values) if frag_values else 0.0,
        "max_frag_tick": None if max_frag_row is None else max_frag_row.tick,
        "max_frag_total_inactive_mib": None if max_frag_row is None else max_frag_row.total_inactive_mib,
        "max_frag_largest_inactive_mib": None if max_frag_row is None else max_frag_row.largest_inactive_mib,
        "max_reserved_mib": max((row.reserved_mib for row in snapshots), default=0.0),
        "max_allocated_mib": max((row.allocated_mib for row in snapshots), default=0.0),
        "max_resident_cache_mib": max((row.resident_cache_mib for row in snapshots), default=0.0),
        "finished_requests": sum(1 for record in records.values() if record.status == "finished"),
        "released_requests": sum(1 for record in records.values() if record.status == "released"),
        "pending_requests": sum(1 for record in records.values() if record.status == "pending"),
        "profiles": [asdict(profile) for profile in QWEN_IMAGE_DATASET_C],
        "cache_config": {
            "Fn_compute_blocks": args.fn_compute_blocks,
            "Bn_compute_blocks": args.bn_compute_blocks,
            "max_warmup_steps": args.max_warmup_steps,
            "residual_diff_threshold": args.residual_diff_threshold,
            "max_continuous_cached_steps": args.max_continuous_cached_steps,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = [row.tick for row in snapshots]
    _write_svg_line_chart(
        output_dir / "charts" / "fragmentation_ratio.svg",
        title="Qwen-Image Runner Fragmentation Ratio",
        x_values=x,
        series=[("frag_ratio", [row.frag_ratio for row in snapshots], "#0072B2")],
        y_label="ratio",
        y_min=0.0,
        y_max=1.0,
    )
    _write_svg_line_chart(
        output_dir / "charts" / "inactive_free_blocks.svg",
        title="Allocator Inactive Free Memory",
        x_values=x,
        series=[
            ("total inactive", [row.total_inactive_mib for row in snapshots], "#0072B2"),
            ("largest inactive block", [row.largest_inactive_mib for row in snapshots], "#D55E00"),
        ],
        y_label="MiB",
    )
    _write_svg_line_chart(
        output_dir / "charts" / "resident_cache.svg",
        title="Resident Cache-DiT Slot Bytes",
        x_values=x,
        series=[("resident cache", [row.resident_cache_mib for row in snapshots], "#009E73")],
        y_label="MiB",
    )

    readme = f"""# Qwen-Image Runner Fragmentation E2E

This run uses real `DiffusionModelRunner.execute_stepwise()` with
`cache_backend=cache_dit` and original contiguous CUDA tensors. It does not use
the paged cache path.

Dataset: Qwen-Image Dataset C.

Result:

- oom_observed: `{summary["oom_observed"]}`
- oom_probe_attempted: `{summary["oom_probe_attempted"]}`
- oom_probe_oom_observed: `{summary["oom_probe_oom_observed"]}`
- oom_probe_allocation_mib: `{summary["oom_probe_allocation_mib"]:.2f}`
- oom_probe_pressure_allocated_mib: `{summary["oom_probe_pressure_allocated_mib"]:.2f}`
- max_frag_ratio: `{summary["max_frag_ratio"]:.4f}`
- mean_frag_ratio: `{summary["mean_frag_ratio"]:.4f}`
- max_resident_cache_mib: `{summary["max_resident_cache_mib"]:.2f}`
- finished_requests: `{summary["finished_requests"]}`
- released_requests: `{summary["released_requests"]}`
- pending_requests: `{summary["pending_requests"]}`
- error: `{summary["error"]}`

Files:

- `trace.json`: generated random arrivals and request profiles
- `requests.csv`: request lifecycle summary
- `timeline.csv`: allocator/cache snapshot after each runner event
- `summary.json`: aggregate metrics
- `charts/fragmentation_ratio.svg`
- `charts/inactive_free_blocks.svg`
- `charts/resident_cache.svg`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    default_output = (
        Path("benchmarks")
        / "diffusion"
        / "results"
        / "qwen_image_runner_fragmentation"
        / time.strftime("dataset_c_cache_dit_%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(description="Run Qwen-Image runner-level Cache-DiT fragmentation experiment.")
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen-Image"),
        help="Local Qwen-Image model path.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=30973)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=180)
    parser.add_argument("--arrival-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-live-requests", type=int, default=4)
    parser.add_argument("--min-live-requests", type=int, default=2)
    parser.add_argument("--new-request-probability", type=float, default=0.45)
    parser.add_argument("--release-probability", type=float, default=0.12)
    parser.add_argument("--min-steps-before-release", type=int, default=2)
    parser.add_argument("--fn-compute-blocks", type=int, default=1)
    parser.add_argument("--bn-compute-blocks", type=int, default=0)
    parser.add_argument("--max-warmup-steps", type=int, default=1)
    parser.add_argument("--residual-diff-threshold", type=float, default=0.24)
    parser.add_argument("--max-continuous-cached-steps", type=int, default=3)
    parser.add_argument("--enable-oom-probe", action="store_true")
    parser.add_argument("--probe-min-frag-ratio", type=float, default=0.92)
    parser.add_argument("--probe-min-total-inactive-mib", type=float, default=3000.0)
    parser.add_argument("--probe-max-largest-inactive-mib", type=float, default=256.0)
    parser.add_argument("--probe-target-global-free-mib", type=float, default=128.0)
    parser.add_argument("--probe-min-pressure-mib", type=float, default=1024.0)
    parser.add_argument("--probe-allocation-mib", type=float, default=3072.0)
    parser.add_argument("--stop-after-probe", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true", default=True)
    parser.add_argument("--vae-use-tiling", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
