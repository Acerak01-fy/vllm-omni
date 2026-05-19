# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-B validation for paged Cache-DiT Triton kernels.

This script has two independent checks:

1. A kernel microbenchmark comparing Phase-B Triton ops against the Phase-A
   gather/copy baseline for Qwen-Image-like buffer shapes.
2. A small real Qwen-Image ``DiffusionModelRunner.execute_stepwise`` smoke run
   with two batched requests and ``enable_paged_cache=True``. The smoke reports
   per-process kernel counters so we can verify the real runner reached the
   Triton scatter/diff/apply paths.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.cache.kernels.paged_cache_ops import (
    get_paged_cache_kernel_stats,
    paged_abs_diff_stats,
    paged_residual_add_,
    paged_scatter_write,
    reset_paged_cache_kernel_stats,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig
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


@dataclass
class BenchRow:
    op: str
    num_tokens: int
    hidden_dim: int
    page_size: int
    dtype: str
    baseline_ms: float
    triton_ms: float
    speedup: float


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


def _time_cuda(fn: Callable[[], Any], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def _phase_a_scatter(
    src: torch.Tensor,
    page_pool: torch.Tensor,
    page_ids: list[int],
    *,
    num_tokens: int,
    page_size: int,
) -> None:
    for page_idx, page_id in enumerate(page_ids):
        start = page_idx * page_size
        end = min(start + page_size, num_tokens)
        if start >= end:
            break
        page_pool[page_id, : end - start].copy_(src[start:end])


def _phase_a_gather(page_pool: torch.Tensor, page_table: torch.Tensor, num_tokens: int) -> torch.Tensor:
    hidden_dim = int(page_pool.shape[-1])
    return torch.index_select(page_pool, 0, page_table).reshape(-1, hidden_dim)[:num_tokens]


def run_kernel_benchmark(args: argparse.Namespace, output_dir: Path) -> list[BenchRow]:
    device = torch.device(f"cuda:{args.device_index}")
    dtype = getattr(torch, args.benchmark_dtype)
    hidden_dim = args.hidden_dim
    page_size = args.page_size
    rows: list[BenchRow] = []

    for num_tokens in args.token_sizes:
        num_pages = (num_tokens + page_size - 1) // page_size
        page_pool = torch.empty((num_pages + 8, page_size, hidden_dim), dtype=dtype, device=device)
        page_ids = list(range(num_pages - 1, -1, -1))
        page_table = torch.tensor(page_ids, dtype=torch.int32, device=device)
        src = torch.randn((num_tokens, hidden_dim), dtype=dtype, device=device)
        probe = src + 0.01
        target = torch.zeros_like(src)

        paged_scatter_write(
            src,
            page_pool,
            page_table,
            num_tokens=num_tokens,
            page_size=page_size,
            hidden_dim=hidden_dim,
        )
        torch.cuda.synchronize()

        measurements: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
            (
                "scatter_write",
                lambda src=src, page_pool=page_pool, page_ids=page_ids: _phase_a_scatter(
                    src,
                    page_pool,
                    page_ids,
                    num_tokens=num_tokens,
                    page_size=page_size,
                ),
                lambda src=src, page_pool=page_pool, page_table=page_table: paged_scatter_write(
                    src,
                    page_pool,
                    page_table,
                    num_tokens=num_tokens,
                    page_size=page_size,
                    hidden_dim=hidden_dim,
                ),
            ),
            (
                "residual_add",
                lambda target=target, page_pool=page_pool, page_table=page_table: target.add_(
                    _phase_a_gather(page_pool, page_table, num_tokens)
                ),
                lambda target=target, page_pool=page_pool, page_table=page_table: paged_residual_add_(
                    target,
                    page_pool,
                    page_table,
                    num_tokens=num_tokens,
                    page_size=page_size,
                    hidden_dim=hidden_dim,
                    add_input=True,
                ),
            ),
            (
                "abs_diff_stats",
                lambda probe=probe, page_pool=page_pool, page_table=page_table: (
                    lambda cache: torch.stack(
                        [
                            (probe - cache).abs().sum(),
                            cache.abs().sum(),
                        ]
                    )
                )(_phase_a_gather(page_pool, page_table, num_tokens)),
                lambda probe=probe, page_pool=page_pool, page_table=page_table: paged_abs_diff_stats(
                    probe,
                    page_pool,
                    page_table,
                    num_tokens=num_tokens,
                    page_size=page_size,
                    hidden_dim=hidden_dim,
                ),
            ),
        ]

        for op, baseline_fn, triton_fn in measurements:
            baseline_ms = _time_cuda(baseline_fn, warmup=args.warmup, repeats=args.repeats)
            triton_ms = _time_cuda(triton_fn, warmup=args.warmup, repeats=args.repeats)
            rows.append(
                BenchRow(
                    op=op,
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    page_size=page_size,
                    dtype=str(dtype),
                    baseline_ms=baseline_ms,
                    triton_ms=triton_ms,
                    speedup=baseline_ms / triton_ms if triton_ms > 0 else 0.0,
                )
            )

    _write_csv(output_dir / "kernel_benchmark.csv", rows)
    return rows


def _make_request(
    req_id: str,
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> OmniDiffusionRequest:
    generator = torch.Generator(device=device).manual_seed(seed)
    sampling = OmniDiffusionSamplingParams(
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=0.0,
        guidance_scale_provided=True,
        true_cfg_scale=1.0,
        generator=generator,
        seed=seed,
        generator_device=str(device),
        num_outputs_per_prompt=1,
    )
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a small ceramic vase on a wooden table", "negative_prompt": "blurry, low quality"}],
        sampling_params=sampling,
        request_ids=[req_id],
    )


def _make_sched_output(
    *,
    step_id: int,
    new_reqs: list[tuple[str, OmniDiffusionRequest]] | None = None,
    cached_req_ids: list[str] | None = None,
) -> DiffusionSchedulerOutput:
    new_reqs = new_reqs or []
    cached_req_ids = cached_req_ids or []
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(sched_req_id=req_id, req=req) for req_id, req in new_reqs],
        scheduled_cached_reqs=CachedRequestData.make_empty()
        if not cached_req_ids
        else CachedRequestData(sched_req_ids=cached_req_ids),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs) + len(cached_req_ids),
        num_waiting_reqs=0,
    )


def _init_runner(args: argparse.Namespace) -> tuple[DiffusionModelRunner, torch.device]:
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
        enable_paged_cache=True,
        paged_cache_page_size=args.page_size,
        paged_cache_num_pages=args.paged_cache_num_pages,
        paged_cache_max_seq_len=args.paged_cache_max_seq_len,
        paged_cache_max_concurrent_requests=args.qwen_batch_size,
        paged_cache_buffers_per_block=3,
        paged_cache_safety_factor=1.0,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        model_class_name="QwenImagePipeline",
        dtype=args.dtype,
        cache_backend="cache_dit",
        cache_config=cache_config,
        step_execution=True,
        enforce_eager=True,
        vae_use_slicing=True,
        vae_use_tiling=False,
        enable_cache_dit_summary=False,
        max_num_seqs=args.qwen_batch_size,
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

    return runner, device


def _resident_cache_mib(runner: DiffusionModelRunner) -> float:
    total = 0
    for state in runner.state_cache.values():
        if state.cache_slot is not None:
            total += int(state.cache_slot.resident_bytes)
    return total / (1024 * 1024)


def run_qwen_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    reset_paged_cache_kernel_stats()
    runner, device = _init_runner(args)
    step_latencies: list[float] = []
    outputs: list[dict[str, Any]] = []
    req_ids = [f"phase-b-{idx}" for idx in range(args.qwen_batch_size)]

    try:
        requests = [
            (
                req_id,
                _make_request(
                    req_id,
                    width=args.qwen_width,
                    height=args.qwen_height,
                    steps=args.qwen_steps,
                    seed=args.seed + idx,
                    device=device,
                ),
            )
            for idx, req_id in enumerate(req_ids)
        ]
        schedules = [
            _make_sched_output(step_id=0, new_reqs=requests),
            *[_make_sched_output(step_id=step_id, cached_req_ids=req_ids) for step_id in range(1, args.qwen_steps)],
        ]

        for sched in schedules:
            torch.cuda.synchronize(args.device_index)
            t0 = time.perf_counter()
            output = runner.execute_stepwise(sched)
            torch.cuda.synchronize(args.device_index)
            latency = time.perf_counter() - t0
            step_latencies.append(latency)
            outputs.append(
                {
                    "step_id": sched.step_id,
                    "req_id": output.req_id,
                    "step_index": output.step_index,
                    "finished": output.finished,
                    "latency_s": latency,
                    "resident_cache_mib": _resident_cache_mib(runner),
                    "kernel_stats": get_paged_cache_kernel_stats(),
                }
            )
            if isinstance(output.finished, list) and all(output.finished):
                break
            if isinstance(output.finished, bool) and output.finished:
                break

        driver = runner.cache_manager.driver if runner.cache_manager is not None else None
        pool_stats = driver.pool.stats() if hasattr(driver, "pool") else {}
        summary = {
            "model": args.model,
            "width": args.qwen_width,
            "height": args.qwen_height,
            "steps_requested": args.qwen_steps,
            "batch_size": args.qwen_batch_size,
            "num_steps_run": len(step_latencies),
            "mean_step_latency_s": sum(step_latencies) / len(step_latencies) if step_latencies else 0.0,
            "max_step_latency_s": max(step_latencies) if step_latencies else 0.0,
            "kernel_stats": get_paged_cache_kernel_stats(),
            "pool_stats": pool_stats,
            "outputs": outputs,
        }
    finally:
        for state in list(runner.state_cache.values()):
            if runner.cache_manager is not None:
                runner.cache_manager.free(state)
        runner.state_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        destroy_distributed_env()

    (output_dir / "qwen_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    default_output = (
        Path("benchmarks")
        / "diffusion"
        / "results"
        / "qwen_image_paged_cache_phase_b"
        / time.strftime("phase_b_%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(description="Validate paged Cache-DiT Phase-B Triton kernels.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=30991)
    parser.add_argument("--model", default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen-Image"))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--benchmark-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--token-sizes", type=int, nargs="+", default=[256, 1024, 2304])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--skip-qwen-smoke", action="store_true")
    parser.add_argument("--qwen-width", type=int, default=512)
    parser.add_argument("--qwen-height", type=int, default=512)
    parser.add_argument("--qwen-steps", type=int, default=4)
    parser.add_argument("--qwen-batch-size", type=int, default=2)
    parser.add_argument("--paged-cache-num-pages", type=int, default=None)
    parser.add_argument("--paged-cache-max-seq-len", type=int, default=1024)
    parser.add_argument("--fn-compute-blocks", type=int, default=1)
    parser.add_argument("--bn-compute-blocks", type=int, default=0)
    parser.add_argument("--max-warmup-steps", type=int, default=1)
    parser.add_argument("--residual-diff-threshold", type=float, default=0.999999)
    parser.add_argument("--max-continuous-cached-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_paged_cache_kernel_stats()
    bench_rows = run_kernel_benchmark(args, output_dir)
    result = {
        "kernel_benchmark": [asdict(row) for row in bench_rows],
        "kernel_benchmark_stats": get_paged_cache_kernel_stats(),
    }

    if not args.skip_qwen_smoke:
        result["qwen_smoke"] = run_qwen_smoke(args, output_dir)

    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
