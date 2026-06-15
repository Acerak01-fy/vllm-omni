# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark dense Hunyuan Image3 KV reuse against vLLM paged attention reuse."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import hunyuan_image3_paged_kv_smoke as smoke  # noqa: E402

DEFAULT_SCENARIOS = (
    "text2img-nocfg",
    "text2img-cfg",
    "img2img-1ref-nocfg",
    "img2img-3ref-nocfg",
    "img2img-3ref-cfg",
)
DEFAULT_VARIANTS = ("dense", "paged")
BENCHMARK_METRICS_PREFIX = "[Benchmark Metrics] "
PAGED_KV_STATS_PREFIX = "[Paged KV Stats] "
SAVED_IMAGE_RE = re.compile(r"^\[Output\] Saved image to (?P<path>.+)$")
PROFILE_COUNT_KEYS = (
    "profile_dense_reuse_calls",
    "profile_dense_later_attention_calls",
    "profile_paged_metadata_build_calls",
    "profile_paged_custom_mask_build_calls",
    "profile_paged_runner_calls",
    "profile_vllm_cache_write_calls",
    "profile_vllm_paged_attention_calls",
)
PROFILE_TIME_KEYS = (
    "profile_dense_reuse_total_ms",
    "profile_dense_later_attention_total_ms",
    "profile_paged_metadata_build_total_ms",
    "profile_paged_custom_mask_build_total_ms",
    "profile_paged_runner_total_ms",
    "profile_vllm_cache_write_total_ms",
    "profile_vllm_paged_attention_total_ms",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    modality: str
    guidance_scale: float
    reference_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hunyuan Image3 paged KV performance benchmark.")
    parser.add_argument("--model", default=smoke.DEFAULT_MODEL)
    parser.add_argument("--deploy-config", default=str(smoke.DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--devices", default=None, help="Optional comma-separated device list for a temp deploy file.")
    parser.add_argument("--quantization", default=None, help="Optional quantization override, e.g. fp8.")
    parser.add_argument("--output-dir", default=None, help="Benchmark output directory. Defaults to a temp directory.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A brown and white dog is running on the grass.")
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale used by *-cfg benchmark scenarios.",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=f"Comma-separated scenarios. Known: {', '.join(DEFAULT_SCENARIOS)}.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants: dense,paged.",
    )
    parser.add_argument(
        "--reference-images",
        default=None,
        help="Comma-separated image paths for img2img scenarios. Need at least the max scenario reference count.",
    )
    parser.add_argument("--bot-task", default="none")
    parser.add_argument("--sys-type", default="en_unified")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile-paged-kv",
        action="store_true",
        help="Enable synchronized dense/paged KV reuse profiling in the offline script.",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Per subprocess timeout in seconds.")
    return parser.parse_args()


def _parse_csv(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def scenario_catalog(cfg_guidance_scale: float) -> dict[str, Scenario]:
    return {
        "text2img-nocfg": Scenario("text2img-nocfg", "text2img", 1.0, 0),
        "text2img-cfg": Scenario("text2img-cfg", "text2img", cfg_guidance_scale, 0),
        "img2img-1ref-nocfg": Scenario("img2img-1ref-nocfg", "img2img", 1.0, 1),
        "img2img-1ref-cfg": Scenario("img2img-1ref-cfg", "img2img", cfg_guidance_scale, 1),
        "img2img-3ref-nocfg": Scenario("img2img-3ref-nocfg", "img2img", 1.0, 3),
        "img2img-3ref-cfg": Scenario("img2img-3ref-cfg", "img2img", cfg_guidance_scale, 3),
    }


def resolve_scenarios(raw_value: str, cfg_guidance_scale: float) -> list[Scenario]:
    catalog = scenario_catalog(cfg_guidance_scale)
    names = _parse_csv(raw_value)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Known scenarios: {sorted(catalog)}")
    if not names:
        raise ValueError("--scenarios must contain at least one scenario.")
    return [catalog[name] for name in names]


def resolve_variants(raw_value: str) -> list[str]:
    variants = _parse_csv(raw_value)
    unknown = [variant for variant in variants if variant not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Known variants: {list(DEFAULT_VARIANTS)}")
    if not variants:
        raise ValueError("--variants must contain at least one variant.")
    return variants


def validate_args(args: argparse.Namespace) -> None:
    smoke.validate_args(args)
    if args.repeat <= 0:
        raise ValueError(f"--repeat must be positive, got {args.repeat}.")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}.")
    if args.cfg_guidance_scale <= 1.0 and any(name.endswith("-cfg") for name in _parse_csv(args.scenarios)):
        raise ValueError("--cfg-guidance-scale must be > 1.0 when running CFG scenarios.")
    resolve_scenarios(args.scenarios, args.cfg_guidance_scale)
    resolve_variants(args.variants)


def build_variant_env(args: argparse.Namespace, variant: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if getattr(args, "hf_home", None):
        env["HF_HOME"] = str(pathlib.Path(args.hf_home).expanduser())
    if not args.allow_download:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if variant == "paged":
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] = "required"
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"] = str(args.page_size)
        env.setdefault("VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS", "1")
    else:
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] = "0"
        env.pop("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", None)
    if args.profile_paged_kv:
        env["VLLM_OMNI_HY3_PAGED_KV_PROFILE"] = "1"
    else:
        env.pop("VLLM_OMNI_HY3_PAGED_KV_PROFILE", None)

    pythonpath_parts = [str(smoke.REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _generate_reference_images(output_dir: pathlib.Path, count: int, width: int, height: int) -> list[pathlib.Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("Pillow is required to generate synthetic reference images.") from exc

    ref_dir = output_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    palette = ((180, 74, 69), (69, 131, 176), (93, 145, 91))
    paths: list[pathlib.Path] = []
    for idx in range(count):
        base = palette[idx % len(palette)]
        image = Image.new("RGB", (width, height), base)
        draw = ImageDraw.Draw(image)
        step = max(min(width, height) // 8, 1)
        for offset in range(0, width + height, step):
            color = tuple(min(channel + 40 + idx * 8, 255) for channel in base)
            draw.line((offset, 0, 0, offset), fill=color, width=max(step // 8, 1))
        margin = max(min(width, height) // 8, 1)
        draw.rectangle((margin, margin, width - margin, height - margin), outline=(245, 245, 235), width=4)
        path = ref_dir / f"reference_{idx}.png"
        image.save(path)
        paths.append(path)
    return paths


def resolve_reference_images(
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    scenarios: list[Scenario],
) -> list[pathlib.Path]:
    max_reference_count = max((scenario.reference_count for scenario in scenarios), default=0)
    if max_reference_count <= 0:
        return []
    if args.reference_images:
        paths = [pathlib.Path(path).expanduser() for path in _parse_csv(args.reference_images)]
        if len(paths) < max_reference_count:
            raise ValueError(
                f"--reference-images needs at least {max_reference_count} images for the selected scenarios; "
                f"got {len(paths)}."
            )
        missing = [str(path) for path in paths[:max_reference_count] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Reference image path(s) do not exist: {missing}")
        return paths
    return _generate_reference_images(output_dir, max_reference_count, args.width, args.height)


def case_output_dir(
    base_output_dir: pathlib.Path,
    scenario: Scenario,
    variant: str,
    run_index: int,
    warmup: bool,
) -> pathlib.Path:
    run_type = "warmup" if warmup else "run"
    return base_output_dir / scenario.name / variant / f"{run_type}_{run_index}"


def build_offline_command(
    args: argparse.Namespace,
    deploy_config: pathlib.Path,
    scenario: Scenario,
    variant: str,
    output_dir: pathlib.Path,
    reference_images: list[pathlib.Path],
) -> list[str]:
    cmd = [
        sys.executable,
        str(smoke.OFFLINE_SCRIPT),
        "--modality",
        scenario.modality,
        "--model",
        args.model,
        "--deploy-config",
        str(deploy_config),
        "--prompts",
        args.prompt,
        "--output",
        str(output_dir),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(scenario.guidance_scale),
        "--seed",
        str(args.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--bot-task",
        args.bot_task,
        "--sys-type",
        args.sys_type,
        "--enforce-eager",
        "--print-paged-kv-stats",
    ]
    if scenario.reference_count > 0:
        cmd.extend(["--image-path", ",".join(str(path) for path in reference_images[: scenario.reference_count])])
    if variant == "paged":
        cmd.extend(
            [
                "--require-paged-kv-cache",
                "--paged-kv-cache-page-size",
                str(args.page_size),
            ]
        )
    if args.profile_paged_kv:
        cmd.append("--profile-paged-kv")
    return cmd


def parse_prefixed_json(output: str, prefix: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = None
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        parsed = json.loads(line[len(prefix) :])
    return parsed


def parse_saved_images(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.splitlines():
        match = SAVED_IMAGE_RE.match(line)
        if match is not None:
            paths.append(match.group("path"))
    return paths


def derive_paged_kv_metrics(stats: dict[str, Any] | None, steps: int) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    enabled_layers = int(stats.get("enabled_layers", 0))
    expected_calls = stats.get("paged_attention_expected_calls")
    if expected_calls is None:
        expected_calls = enabled_layers * max(steps - 1, 0)
    expected_calls = int(expected_calls)
    actual_calls = int(stats.get("paged_attention_calls", 0))
    reuse_coverage = stats.get("paged_attention_reuse_coverage", stats.get("paged_attention_hit_rate"))
    if reuse_coverage is None and expected_calls > 0:
        reuse_coverage = actual_calls / expected_calls
    prefix_page_lookups = int(stats.get("paged_kv_prefix_page_lookups", 0))
    prefix_token_lookups = int(stats.get("paged_kv_prefix_token_lookups", 0))
    prefix_page_hit_rate = stats.get("paged_kv_prefix_page_hit_rate")
    if prefix_page_hit_rate is None and prefix_page_lookups > 0:
        prefix_page_hit_rate = int(stats.get("paged_kv_prefix_page_hits", 0)) / prefix_page_lookups
    prefix_token_hit_rate = stats.get("paged_kv_prefix_token_hit_rate")
    if prefix_token_hit_rate is None and prefix_token_lookups > 0:
        prefix_token_hit_rate = int(stats.get("paged_kv_prefix_token_hits", 0)) / prefix_token_lookups

    cached_token_uses = None
    layer_stats = stats.get("layers_detail")
    if isinstance(layer_stats, list):
        cached_token_uses = sum(
            int(layer.get("paged_kv_cached_tokens", 0)) * int(layer.get("paged_attention_calls", 0))
            for layer in layer_stats
            if isinstance(layer, dict)
        )
    elif enabled_layers > 0:
        cached_tokens_per_layer = int(stats.get("paged_kv_cached_tokens", 0)) / enabled_layers
        cached_token_uses = cached_tokens_per_layer * actual_calls

    derived = {
        "paged_attention_actual_calls": actual_calls,
        "paged_attention_expected_calls": expected_calls,
        "paged_attention_reuse_coverage": reuse_coverage,
        "paged_attention_hit_rate": prefix_page_hit_rate if prefix_page_hit_rate is not None else reuse_coverage,
        "paged_kv_prefix_page_hit_rate": prefix_page_hit_rate,
        "paged_kv_prefix_token_hit_rate": prefix_token_hit_rate,
        "paged_kv_prefix_page_hits": stats.get("paged_kv_prefix_page_hits"),
        "paged_kv_prefix_page_lookups": stats.get("paged_kv_prefix_page_lookups"),
        "paged_kv_prefix_token_hits": stats.get("paged_kv_prefix_token_hits"),
        "paged_kv_prefix_token_lookups": stats.get("paged_kv_prefix_token_lookups"),
        "paged_kv_page_table_entries": stats.get("paged_kv_page_table_entries"),
        "paged_kv_current_page_entries": stats.get("paged_kv_current_page_entries"),
        "paged_kv_cached_token_uses": cached_token_uses,
        "paged_kv_cached_tokens": stats.get("paged_kv_cached_tokens"),
        "paged_kv_max_cached_tokens": stats.get("paged_kv_max_cached_tokens"),
        "paged_kv_num_pages": stats.get("paged_kv_num_pages"),
        "paged_kv_prefix_pages": stats.get("paged_kv_prefix_pages"),
    }
    for key in PROFILE_COUNT_KEYS:
        derived[key] = int(stats.get(key, 0))
    for key in PROFILE_TIME_KEYS:
        derived[key] = float(stats.get(key, 0.0))

    avg_pairs = (
        ("profile_dense_reuse", "profile_dense_reuse_calls", "profile_dense_reuse_total_ms"),
        (
            "profile_dense_later_attention",
            "profile_dense_later_attention_calls",
            "profile_dense_later_attention_total_ms",
        ),
        (
            "profile_paged_metadata_build",
            "profile_paged_metadata_build_calls",
            "profile_paged_metadata_build_total_ms",
        ),
        (
            "profile_paged_custom_mask_build",
            "profile_paged_custom_mask_build_calls",
            "profile_paged_custom_mask_build_total_ms",
        ),
        ("profile_paged_runner", "profile_paged_runner_calls", "profile_paged_runner_total_ms"),
        ("profile_vllm_cache_write", "profile_vllm_cache_write_calls", "profile_vllm_cache_write_total_ms"),
        (
            "profile_vllm_paged_attention",
            "profile_vllm_paged_attention_calls",
            "profile_vllm_paged_attention_total_ms",
        ),
    )
    for prefix, calls_key, total_key in avg_pairs:
        calls = int(derived.get(calls_key, 0))
        total_ms = float(derived.get(total_key, 0.0))
        derived[f"{prefix}_avg_ms"] = total_ms / calls if calls > 0 else None
    return derived


def run_case(
    args: argparse.Namespace,
    deploy_config: pathlib.Path,
    scenario: Scenario,
    variant: str,
    output_dir: pathlib.Path,
    reference_images: list[pathlib.Path],
    *,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_offline_command(args, deploy_config, scenario, variant, output_dir, reference_images)
    env = build_variant_env(args, variant)
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    wall_time_s = time.perf_counter() - started
    output = completed.stdout or ""
    stdout_path = output_dir / "stdout.log"
    stdout_path.write_text(output)
    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(
            f"Benchmark case failed: scenario={scenario.name} variant={variant} "
            f"warmup={warmup} run_index={run_index} returncode={completed.returncode}\n{tail}"
        )

    benchmark_metrics = parse_prefixed_json(output, BENCHMARK_METRICS_PREFIX) or {}
    paged_kv_stats = parse_prefixed_json(output, PAGED_KV_STATS_PREFIX)
    result: dict[str, Any] = {
        "scenario": scenario.name,
        "modality": scenario.modality,
        "guidance_scale": scenario.guidance_scale,
        "reference_count": scenario.reference_count,
        "variant": variant,
        "warmup": warmup,
        "run_index": run_index,
        "wall_time_s": wall_time_s,
        "generation_wall_time_s": benchmark_metrics.get("generation_wall_time_s"),
        "benchmark_metrics": benchmark_metrics,
        "paged_kv_stats": paged_kv_stats,
        "derived_paged_kv_metrics": derive_paged_kv_metrics(paged_kv_stats, args.steps),
        "output_images": parse_saved_images(output),
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "command": cmd,
    }
    return result


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = sorted({result["scenario"] for result in results})
    for scenario in scenarios:
        scenario_results = [result for result in results if result["scenario"] == scenario and not result["warmup"]]
        dense_results = [result for result in scenario_results if result["variant"] == "dense"]
        paged_results = [result for result in scenario_results if result["variant"] == "paged"]
        dense_generation = [
            float(result["generation_wall_time_s"])
            for result in dense_results
            if result.get("generation_wall_time_s") is not None
        ]
        paged_generation = [
            float(result["generation_wall_time_s"])
            for result in paged_results
            if result.get("generation_wall_time_s") is not None
        ]
        dense_wall = [float(result["wall_time_s"]) for result in dense_results]
        paged_wall = [float(result["wall_time_s"]) for result in paged_results]
        dense_generation_mean = _mean(dense_generation)
        paged_generation_mean = _mean(paged_generation)
        speedup = (
            dense_generation_mean / paged_generation_mean
            if dense_generation_mean is not None and paged_generation_mean not in (None, 0.0)
            else None
        )
        dense_metrics = dense_results[-1].get("derived_paged_kv_metrics", {}) if dense_results else {}
        paged_metrics = paged_results[-1].get("derived_paged_kv_metrics", {}) if paged_results else {}
        rows.append(
            {
                "scenario": scenario,
                "dense_generation_wall_time_s_mean": dense_generation_mean,
                "paged_generation_wall_time_s_mean": paged_generation_mean,
                "generation_speedup_dense_over_paged": speedup,
                "dense_outer_wall_time_s_mean": _mean(dense_wall),
                "paged_outer_wall_time_s_mean": _mean(paged_wall),
                "paged_attention_hit_rate": paged_metrics.get("paged_attention_hit_rate"),
                "paged_attention_reuse_coverage": paged_metrics.get("paged_attention_reuse_coverage"),
                "paged_kv_prefix_page_hit_rate": paged_metrics.get("paged_kv_prefix_page_hit_rate"),
                "paged_kv_prefix_token_hit_rate": paged_metrics.get("paged_kv_prefix_token_hit_rate"),
                "paged_attention_actual_calls": paged_metrics.get("paged_attention_actual_calls"),
                "paged_attention_expected_calls": paged_metrics.get("paged_attention_expected_calls"),
                "paged_kv_cached_tokens": paged_metrics.get("paged_kv_cached_tokens"),
                "paged_kv_max_cached_tokens": paged_metrics.get("paged_kv_max_cached_tokens"),
                "paged_kv_prefix_pages": paged_metrics.get("paged_kv_prefix_pages"),
                "paged_kv_prefix_page_hits": paged_metrics.get("paged_kv_prefix_page_hits"),
                "paged_kv_prefix_page_lookups": paged_metrics.get("paged_kv_prefix_page_lookups"),
                "paged_kv_prefix_token_hits": paged_metrics.get("paged_kv_prefix_token_hits"),
                "paged_kv_prefix_token_lookups": paged_metrics.get("paged_kv_prefix_token_lookups"),
                "paged_kv_page_table_entries": paged_metrics.get("paged_kv_page_table_entries"),
                "paged_kv_current_page_entries": paged_metrics.get("paged_kv_current_page_entries"),
                "paged_kv_cached_token_uses": paged_metrics.get("paged_kv_cached_token_uses"),
                "profile_dense_reuse_total_ms": dense_metrics.get("profile_dense_reuse_total_ms"),
                "profile_dense_later_attention_total_ms": dense_metrics.get("profile_dense_later_attention_total_ms"),
                "profile_paged_metadata_build_total_ms": paged_metrics.get("profile_paged_metadata_build_total_ms"),
                "profile_paged_custom_mask_build_total_ms": paged_metrics.get(
                    "profile_paged_custom_mask_build_total_ms"
                ),
                "profile_paged_runner_total_ms": paged_metrics.get("profile_paged_runner_total_ms"),
                "profile_vllm_cache_write_total_ms": paged_metrics.get("profile_vllm_cache_write_total_ms"),
                "profile_vllm_paged_attention_total_ms": paged_metrics.get("profile_vllm_paged_attention_total_ms"),
                "profile_vllm_cache_write_avg_ms": paged_metrics.get("profile_vllm_cache_write_avg_ms"),
                "profile_vllm_paged_attention_avg_ms": paged_metrics.get("profile_vllm_paged_attention_avg_ms"),
            }
        )
    return rows


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown_summary(path: pathlib.Path, args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Hunyuan Image3 Paged KV Benchmark",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(sys.argv),
        "```",
        "",
        "## Summary",
        "",
        "| Scenario | Dense gen s | Paged gen s | Speedup | Prefix page hit rate | "
        "Reuse coverage | Calls | Prefix page hits | Prefix token hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        calls = (
            f"{row.get('paged_attention_actual_calls')}/{row.get('paged_attention_expected_calls')}"
            if row.get("paged_attention_actual_calls") is not None
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    _format_number(row.get("dense_generation_wall_time_s_mean")),
                    _format_number(row.get("paged_generation_wall_time_s_mean")),
                    _format_number(row.get("generation_speedup_dense_over_paged")),
                    _format_number(row.get("paged_kv_prefix_page_hit_rate")),
                    _format_number(row.get("paged_attention_reuse_coverage")),
                    calls,
                    (
                        f"{_format_number(row.get('paged_kv_prefix_page_hits'), digits=0)}"
                        f"/{_format_number(row.get('paged_kv_prefix_page_lookups'), digits=0)}"
                    ),
                    (
                        f"{_format_number(row.get('paged_kv_prefix_token_hits'), digits=0)}"
                        f"/{_format_number(row.get('paged_kv_prefix_token_lookups'), digits=0)}"
                    ),
                ]
            )
            + " |"
        )
    if any(row.get("profile_paged_runner_total_ms") for row in summary):
        lines.extend(
            [
                "",
                "## Profile Breakdown",
                "",
                "Profile mode synchronizes CUDA around measured regions; use this table for "
                "attribution, not throughput.",
                "",
                "| Scenario | Dense reuse ms | Dense attn ms | Paged metadata ms | Mask build ms | "
                "Paged runner ms | vLLM cache write ms | vLLM paged attn ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["scenario"]),
                        _format_number(row.get("profile_dense_reuse_total_ms")),
                        _format_number(row.get("profile_dense_later_attention_total_ms")),
                        _format_number(row.get("profile_paged_metadata_build_total_ms")),
                        _format_number(row.get("profile_paged_custom_mask_build_total_ms")),
                        _format_number(row.get("profile_paged_runner_total_ms")),
                        _format_number(row.get("profile_vllm_cache_write_total_ms")),
                        _format_number(row.get("profile_vllm_paged_attention_total_ms")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            f"- model: `{args.model}`",
            f"- steps: `{args.steps}`",
            f"- image size: `{args.width}x{args.height}`",
            f"- quantization override: `{args.quantization}`",
            f"- devices override: `{args.devices}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def print_dry_run_plan(
    args: argparse.Namespace,
    deploy_config: pathlib.Path,
    scenarios: list[Scenario],
    variants: list[str],
    reference_images: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> None:
    payload = {
        "validation": "dry-run only; no runtime, checkpoint, or inference validation was executed",
        "model": args.model,
        "deploy_config": str(deploy_config),
        "effective_deploy_config": (
            "temporary single-stage deploy rewrite at runtime"
            if args.devices is not None or args.quantization is not None
            else str(deploy_config)
        ),
        "devices_override": args.devices,
        "quantization_override": args.quantization,
        "output_dir": str(output_dir),
        "scenarios": [scenario.name for scenario in scenarios],
        "variants": variants,
        "reference_images": [str(path) for path in reference_images],
        "commands": [
            {
                "scenario": scenario.name,
                "variant": variant,
                "command": build_offline_command(
                    args,
                    deploy_config,
                    scenario,
                    variant,
                    case_output_dir(output_dir, scenario, variant, 0, False),
                    reference_images,
                ),
                "proof_env": smoke.proof_env_subset(build_variant_env(args, variant)),
            }
            for scenario in scenarios
            for variant in variants
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    deploy_config = pathlib.Path(args.deploy_config)
    scenarios = resolve_scenarios(args.scenarios, args.cfg_guidance_scale)
    variants = resolve_variants(args.variants)
    output_dir = pathlib.Path(args.output_dir or tempfile.mkdtemp(prefix="hy3-paged-kv-benchmark-")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        reference_images = resolve_reference_images(args, output_dir, scenarios)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    if args.dry_run:
        print_dry_run_plan(args, deploy_config, scenarios, variants, reference_images, output_dir)
        return

    if not args.preflight_only:
        if args.allow_download:
            try:
                smoke.validate_model_download_preflight(args)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from None
        elif not smoke.model_available_locally(args.model, hf_home=getattr(args, "hf_home", None)):
            raise SystemExit(
                f"Model {args.model!r} is not present in the Hugging Face cache. "
                "Run with --allow-download in an environment with enough disk, or pre-populate HF_HOME."
            )

    temp_deploy_config: pathlib.Path | None = None
    if args.devices is not None or args.quantization is not None:
        temp_deploy_config = smoke.write_temp_deploy_config(
            deploy_config,
            args.devices,
            quantization=args.quantization,
        )
        deploy_config = temp_deploy_config

    try:
        try:
            smoke.preflight_runtime(deploy_config)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        if args.preflight_only:
            print("Hunyuan Image3 paged KV benchmark preflight passed.")
            return

        results: list[dict[str, Any]] = []
        for scenario in scenarios:
            for variant in variants:
                for warmup_index in range(args.warmup):
                    print(f"[Benchmark] warmup scenario={scenario.name} variant={variant} index={warmup_index}")
                    results.append(
                        run_case(
                            args,
                            deploy_config,
                            scenario,
                            variant,
                            case_output_dir(output_dir, scenario, variant, warmup_index, True),
                            reference_images,
                            run_index=warmup_index,
                            warmup=True,
                        )
                    )
                for run_index in range(args.repeat):
                    print(f"[Benchmark] run scenario={scenario.name} variant={variant} index={run_index}")
                    results.append(
                        run_case(
                            args,
                            deploy_config,
                            scenario,
                            variant,
                            case_output_dir(output_dir, scenario, variant, run_index, False),
                            reference_images,
                            run_index=run_index,
                            warmup=False,
                        )
                    )

        summary = summarize_results(results)
        result_path = output_dir / "benchmark_results.json"
        summary_path = output_dir / "benchmark_summary.md"
        result_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2, sort_keys=True) + "\n")
        write_markdown_summary(summary_path, args, summary)
        print("[Benchmark Summary] " + json.dumps(summary, sort_keys=True))
        print(f"[Benchmark Output] results={result_path}")
        print(f"[Benchmark Output] summary={summary_path}")
    finally:
        if temp_deploy_config is not None:
            temp_deploy_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
