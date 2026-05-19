# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Motivation benchmark for Cache-DiT external fragmentation.

This script intentionally uses the original PyTorch CUDA allocator path:
each simulated request owns one contiguous CUDA tensor representing its
resident Cache-DiT state. It does not import or use the paged cache pool.

The workload is runner-level synthetic because scheduler-side waiting and
preemption are still evolving. The default scenario is derived from the
Wan2.2 Dataset C mix in ``performance_dashboard/wan_2_2_serving_performance.md``.
It pins several small waiting requests, releases medium requests to create
holes, and then attempts to admit a large 720p request.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

MIB = 1024 * 1024


@dataclass(frozen=True)
class DatasetCProfile:
    name: str
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    weight: float

    @property
    def spatial_temporal_units(self) -> int:
        return self.width * self.height * self.num_frames


@dataclass
class AllocationRecord:
    req_id: str
    profile: str
    role: str
    cache_mib: int
    arrival_tick: int
    release_tick: int | None = None
    status: str = "live"


@dataclass
class SnapshotRow:
    tick: int
    action: str
    req_id: str
    profile: str
    cache_mib: int
    total_free_mib: float
    largest_free_mib: float
    frag_ratio: float
    reserved_mib: float
    allocated_mib: float
    inactive_split_mib: float
    live_request_mib: int
    live_requests: str
    note: str


WAN_DATASET_C = [
    DatasetCProfile(
        name="wan_c_480p_80f_3step",
        width=854,
        height=480,
        num_frames=80,
        fps=16,
        num_inference_steps=3,
        weight=0.15,
    ),
    DatasetCProfile(
        name="wan_c_480p_120f_4step",
        width=854,
        height=480,
        num_frames=120,
        fps=24,
        num_inference_steps=4,
        weight=0.25,
    ),
    DatasetCProfile(
        name="wan_c_720p_80f_6step",
        width=1280,
        height=720,
        num_frames=80,
        fps=16,
        num_inference_steps=6,
        weight=0.60,
    ),
]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_cache_mib(
    profiles: list[DatasetCProfile],
    max_cache_mib: int,
) -> dict[str, int]:
    max_units = max(profile.spatial_temporal_units for profile in profiles)
    sizes: dict[str, int] = {}
    for profile in profiles:
        scaled = max_cache_mib * profile.spatial_temporal_units / max_units
        sizes[profile.name] = max(1, int(round(scaled)))
    return sizes


def _allocate_contiguous_mib(
    cache_mib: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_bytes = int(cache_mib) * MIB
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    if num_bytes % dtype_bytes != 0:
        raise ValueError(f"{cache_mib} MiB is not divisible by dtype size {dtype_bytes}")
    return torch.empty((num_bytes // dtype_bytes,), dtype=dtype, device=device)


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
    frag_ratio = 0.0 if total_free <= 0 else 1.0 - (largest_free / total_free)

    stats = torch.cuda.memory_stats(device_index)
    return {
        "total_free_mib": total_free / MIB,
        "largest_free_mib": largest_free / MIB,
        "frag_ratio": frag_ratio,
        "reserved_mib": float(stats.get("reserved_bytes.all.current", 0)) / MIB,
        "allocated_mib": float(stats.get("allocated_bytes.all.current", 0)) / MIB,
        "inactive_split_mib": float(stats.get("inactive_split_bytes.all.current", 0)) / MIB,
        "active_snapshot_mib": active_bytes / MIB,
        "num_free_blocks": float(len(free_blocks)),
    }


def _append_snapshot(
    rows: list[SnapshotRow],
    *,
    tick: int,
    action: str,
    req_id: str,
    profile: str,
    cache_mib: int,
    records: dict[str, AllocationRecord],
    device_index: int,
    note: str = "",
) -> None:
    gc.collect()
    torch.cuda.synchronize(device_index)
    snap = _allocator_snapshot(device_index)
    live_records = [record for record in records.values() if record.status == "live"]
    live_request_mib = sum(record.cache_mib for record in live_records)
    rows.append(
        SnapshotRow(
            tick=tick,
            action=action,
            req_id=req_id,
            profile=profile,
            cache_mib=cache_mib,
            total_free_mib=snap["total_free_mib"],
            largest_free_mib=snap["largest_free_mib"],
            frag_ratio=snap["frag_ratio"],
            reserved_mib=snap["reserved_mib"],
            allocated_mib=snap["allocated_mib"],
            inactive_split_mib=snap["inactive_split_mib"],
            live_request_mib=live_request_mib,
            live_requests=";".join(record.req_id for record in live_records),
            note=note,
        )
    )


def _write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_polyline_chart(
    path: Path,
    *,
    title: str,
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width = 900
    height = 480
    left = 78
    right = 24
    top = 54
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not x_values:
        return
    if y_max is None:
        max_seen = max(max(values) for _, values, _ in series if values)
        y_max = max(max_seen * 1.08, 1.0)
    if y_max <= y_min:
        y_max = y_min + 1.0

    x_min = min(x_values)
    x_max = max(x_values)
    x_span = max(x_max - x_min, 1)

    def sx(x: float) -> float:
        return left + ((x - x_min) / x_span) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    y_ticks = 5
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
        f'<text class="label" x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle">event tick</text>',
        (
            f'<text class="label" transform="translate(18 {top + plot_h / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle">{_xml_escape(y_label)}</text>'
        ),
    ]

    for idx in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * idx / y_ticks
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')

    for x in x_values:
        px = sx(x)
        parts.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{px:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{x}</text>')

    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')

    legend_x = left + 12
    legend_y = top + 18
    for series_idx, (label, values, color) in enumerate(series):
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values, strict=True))
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" points="{points}"/>'
        )
        for x, y in zip(x_values, values, strict=True):
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}"/>')
        ly = legend_y + series_idx * 20
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(f'<text class="label" x="{legend_x + 32}" y="{ly + 4}">{_xml_escape(label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_plot(output_dir: Path, timeline: list[SnapshotRow], new_cache_mib: int) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    x = [row.tick for row in timeline]
    total_free = [row.total_free_mib for row in timeline]
    largest_free = [row.largest_free_mib for row in timeline]
    frag = [row.frag_ratio for row in timeline]

    _svg_polyline_chart(
        charts_dir / "free_vs_largest.svg",
        title="Contiguous CUDA Allocator Fragmentation",
        x_values=x,
        series=[
            ("total free inside allocator", total_free, "#0072B2"),
            ("largest free block", largest_free, "#D55E00"),
            ("new 720p request", [float(new_cache_mib)] * len(x), "#B00020"),
        ],
        y_label="MiB",
    )
    _svg_polyline_chart(
        charts_dir / "fragmentation_ratio.svg",
        title="Fragmentation Ratio",
        x_values=x,
        series=[("fragmentation ratio", frag, "#0072B2")],
        y_label="ratio",
        y_min=0.0,
        y_max=1.0,
    )


def _write_readme(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    profiles: list[DatasetCProfile],
    cache_mib_by_profile: dict[str, int],
    summary: dict[str, Any],
) -> None:
    profile_lines = []
    for profile in profiles:
        profile_lines.append(
            f"| {profile.name} | {profile.width}x{profile.height} | "
            f"{profile.num_frames} | {profile.num_inference_steps} | "
            f"{cache_mib_by_profile[profile.name]} |"
        )

    readme = f"""# Cache-DiT Fragmentation Motivation Run

This run uses original contiguous PyTorch CUDA tensor allocations. It does not
use the paged cache implementation.

## Scenario

The request mix is Wan2.2 Dataset C. Cache sizes are scaled by
`width * height * num_frames`, with the largest 720p profile normalized to
`{args.max_cache_mib}` MiB. `num_inference_steps` is kept as request lifetime
metadata.

| profile | resolution | frames | steps | contiguous cache MiB |
|---|---:|---:|---:|---:|
{os.linesep.join(profile_lines)}

The allocator is seeded with one cached arena, then split into alternating
medium and small request tensors. Medium requests are released to create holes;
small requests remain live as pinned waiting/running Cache-DiT state. A large
720p request is then admitted.

## Result

- failure_observed: `{summary["failure_observed"]}`
- new_request_mib: `{summary["new_request_mib"]:.1f}`
- total_free_mib_before_new_request: `{summary["total_free_mib_before_new_request"]:.1f}`
- largest_free_mib_before_new_request: `{summary["largest_free_mib_before_new_request"]:.1f}`
- fragmentation_ratio_before_new_request: `{summary["frag_ratio_before_new_request"]:.4f}`
- total_free_ge_request: `{summary["total_free_ge_request"]}`
- largest_free_lt_request: `{summary["largest_free_lt_request"]}`

The failed allocation satisfies the motivation condition when
`total_free_ge_request=True` and `largest_free_lt_request=True`.

## Files

- `summary.json`: final evidence and allocator OOM line
- `timeline.csv`: allocator state after each runner-level event
- `allocations.csv`: simulated request/cache allocation records
- `charts/free_vs_largest.svg`: free memory versus largest contiguous block
- `charts/fragmentation_ratio.svg`: fragmentation ratio over the event timeline
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this motivation benchmark.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}")
    device_index = 0 if device.index is None else device.index
    torch.cuda.set_device(device_index)

    dtype = _dtype_from_name(args.dtype)
    profiles = WAN_DATASET_C
    cache_mib_by_profile = _normalize_cache_mib(profiles, args.max_cache_mib)

    small = profiles[0]
    medium = profiles[1]
    large = profiles[2]
    small_mib = cache_mib_by_profile[small.name]
    medium_mib = cache_mib_by_profile[medium.name]
    large_mib = cache_mib_by_profile[large.name]

    arena_mib = args.arena_mib or (args.max_cache_mib * 4)
    tail_slack_mib = args.tail_slack_mib
    if tail_slack_mib is None:
        tail_slack_mib = max(128, args.max_cache_mib // 4)

    layout_mib = 3 * (medium_mib + small_mib)
    guard_mib = arena_mib - layout_mib - tail_slack_mib
    if guard_mib <= 0:
        raise ValueError(
            "Arena is too small for the default fragmentation layout: "
            f"arena={arena_mib} MiB, layout={layout_mib} MiB, "
            f"tail_slack={tail_slack_mib} MiB."
        )

    cap_mib = args.cap_mib or (arena_mib + max(128, args.max_cache_mib // 4))
    if cap_mib - arena_mib >= large_mib:
        raise ValueError(
            "Cap slack must be smaller than the new large request, otherwise "
            "the allocator can satisfy it by creating a new segment."
        )

    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    cap_fraction = (cap_mib * MIB) / total_memory
    if not (0 < cap_fraction <= 1.0):
        raise ValueError(f"Invalid cap fraction {cap_fraction:.6f} from cap_mib={cap_mib}")

    torch.cuda.set_per_process_memory_fraction(cap_fraction, device_index)
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    records: dict[str, AllocationRecord] = {}
    timeline: list[SnapshotRow] = []
    tick = 0

    # Create one large cached segment, then split it with request allocations.
    arena = _allocate_contiguous_mib(arena_mib, dtype=dtype, device=device)
    _append_snapshot(
        timeline,
        tick=tick,
        action="seed_arena_alloc",
        req_id="arena",
        profile="allocator_seed",
        cache_mib=arena_mib,
        records=records,
        device_index=device_index,
        note="Allocate one contiguous arena to force subsequent splits.",
    )
    tick += 1
    del arena
    gc.collect()
    _append_snapshot(
        timeline,
        tick=tick,
        action="seed_arena_free",
        req_id="arena",
        profile="allocator_seed",
        cache_mib=arena_mib,
        records=records,
        device_index=device_index,
        note="Arena is cached by the original CUDA allocator.",
    )
    tick += 1

    for i in range(3):
        req_id = f"req_medium_{i}"
        tensors[req_id] = _allocate_contiguous_mib(medium_mib, dtype=dtype, device=device)
        records[req_id] = AllocationRecord(
            req_id=req_id,
            profile=medium.name,
            role="released_to_create_hole",
            cache_mib=medium_mib,
            arrival_tick=tick,
        )
        _append_snapshot(
            timeline,
            tick=tick,
            action="arrive_allocate",
            req_id=req_id,
            profile=medium.name,
            cache_mib=medium_mib,
            records=records,
            device_index=device_index,
            note="Medium Dataset C request allocates one contiguous cache tensor.",
        )
        tick += 1

        req_id = f"req_small_waiting_{i}"
        tensors[req_id] = _allocate_contiguous_mib(small_mib, dtype=dtype, device=device)
        records[req_id] = AllocationRecord(
            req_id=req_id,
            profile=small.name,
            role="waiting_pinned_cache",
            cache_mib=small_mib,
            arrival_tick=tick,
        )
        _append_snapshot(
            timeline,
            tick=tick,
            action="arrive_allocate",
            req_id=req_id,
            profile=small.name,
            cache_mib=small_mib,
            records=records,
            device_index=device_index,
            note="Small request remains waiting/running with Cache-DiT state pinned.",
        )
        tick += 1

    tensors["workspace_guard"] = _allocate_contiguous_mib(guard_mib, dtype=dtype, device=device)
    _append_snapshot(
        timeline,
        tick=tick,
        action="workspace_guard_allocate",
        req_id="workspace_guard",
        profile="model_workspace_budget",
        cache_mib=guard_mib,
        records=records,
        device_index=device_index,
        note="Guard emulates model/workspace memory that leaves only a cache budget.",
    )
    tick += 1

    for i in range(3):
        req_id = f"req_medium_{i}"
        del tensors[req_id]
        records[req_id].status = "released"
        records[req_id].release_tick = tick
        gc.collect()
        _append_snapshot(
            timeline,
            tick=tick,
            action="release",
            req_id=req_id,
            profile=medium.name,
            cache_mib=medium_mib,
            records=records,
            device_index=device_index,
            note="Release creates a free hole between live waiting requests.",
        )
        tick += 1

    before_new = timeline[-1]
    oom_line = ""
    failure_observed = False
    try:
        tensors["req_large_new"] = _allocate_contiguous_mib(large_mib, dtype=dtype, device=device)
        records["req_large_new"] = AllocationRecord(
            req_id="req_large_new",
            profile=large.name,
            role="new_admission",
            cache_mib=large_mib,
            arrival_tick=tick,
            status="live",
        )
        note = "Unexpected success: allocator found or created a contiguous block."
        action = "admission_attempt_success"
    except torch.cuda.OutOfMemoryError as exc:
        failure_observed = True
        oom_line = str(exc).splitlines()[0]
        records["req_large_new"] = AllocationRecord(
            req_id="req_large_new",
            profile=large.name,
            role="new_admission",
            cache_mib=large_mib,
            arrival_tick=tick,
            status="failed_oom",
        )
        note = oom_line
        action = "admission_attempt_failed"

    _append_snapshot(
        timeline,
        tick=tick,
        action=action,
        req_id="req_large_new",
        profile=large.name,
        cache_mib=large_mib,
        records=records,
        device_index=device_index,
        note=note,
    )

    total_free_ge_request = before_new.total_free_mib >= large_mib
    largest_free_lt_request = before_new.largest_free_mib < large_mib
    summary = {
        "failure_observed": failure_observed,
        "device": torch.cuda.get_device_name(device_index),
        "dtype": str(dtype),
        "cap_mib": cap_mib,
        "arena_mib": arena_mib,
        "guard_mib": guard_mib,
        "tail_slack_mib": tail_slack_mib,
        "new_request_profile": large.name,
        "new_request_mib": float(large_mib),
        "total_free_mib_before_new_request": before_new.total_free_mib,
        "largest_free_mib_before_new_request": before_new.largest_free_mib,
        "frag_ratio_before_new_request": before_new.frag_ratio,
        "reserved_mib_before_new_request": before_new.reserved_mib,
        "allocated_mib_before_new_request": before_new.allocated_mib,
        "inactive_split_mib_before_new_request": before_new.inactive_split_mib,
        "total_free_ge_request": total_free_ge_request,
        "largest_free_lt_request": largest_free_lt_request,
        "motivation_condition_met": failure_observed and total_free_ge_request and largest_free_lt_request,
        "oom_line": oom_line,
        "profiles": [asdict(profile) for profile in profiles],
        "cache_mib_by_profile": cache_mib_by_profile,
    }

    _write_csv(output_dir / "timeline.csv", timeline)
    _write_csv(output_dir / "allocations.csv", list(records.values()))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plot(output_dir, timeline, large_mib)
    _write_readme(
        output_dir,
        args=args,
        profiles=profiles,
        cache_mib_by_profile=cache_mib_by_profile,
        summary=summary,
    )

    # Drop references before leaving so a failed experiment does not keep HBM.
    tensors.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a contiguous CUDA allocator fragmentation motivation experiment.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor dtype for simulated Cache-DiT buffers.",
    )
    parser.add_argument(
        "--max-cache-mib",
        type=int,
        default=1024,
        help="Contiguous cache size assigned to the largest Dataset C profile.",
    )
    parser.add_argument(
        "--arena-mib",
        type=int,
        default=None,
        help="Initial cached arena size. Default: 4 * --max-cache-mib.",
    )
    parser.add_argument(
        "--cap-mib",
        type=int,
        default=None,
        help="PyTorch per-process allocation cap. Default: arena + max(128, max_cache/4).",
    )
    parser.add_argument(
        "--tail-slack-mib",
        type=int,
        default=None,
        help="Unallocated tail left in the arena. Default: max(128, max_cache/4).",
    )
    default_output = (
        Path("benchmarks")
        / "diffusion"
        / "results"
        / "cache_fragmentation_motivation"
        / time.strftime("wan_dataset_c_contiguous_%Y%m%d_%H%M%S")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for summary.json, CSV files, and SVG charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
