# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Hunyuan Image3 inference across main and dit-kvcache worktrees."""

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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAIN_WORKTREE = REPO_ROOT.parent / "vllm-omni-main-bench"
DEFAULT_MODEL = "tencent/HunyuanImage-3.0-Instruct"
DEFAULT_PROMPT = "A brown and white dog is running on the grass."
BENCHMARK_METRICS_PREFIX = "[Benchmark Metrics] "
PAGED_KV_STATS_PREFIX = "[Paged KV Stats] "
SAVED_IMAGE_RE = re.compile(r"^\[Output\] Saved image to (?P<path>.+)$")


@dataclass(frozen=True)
class Variant:
    name: str
    repo_root: pathlib.Path
    paged: bool
    preflight_paged_probe: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark main dense Hunyuan Image3 inference against dit-kvcache paged attention.",
    )
    parser.add_argument("--main-worktree", default=str(DEFAULT_MAIN_WORKTREE))
    parser.add_argument("--dit-worktree", default=str(REPO_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cuda-compat-dir", default=None)
    parser.add_argument("--profile-paged-kv", action="store_true")
    parser.add_argument(
        "--include-dit-dense",
        action="store_true",
        help="Also run the current dit-kvcache worktree with paged KV disabled for same-commit attribution.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def git_commit(repo_root: pathlib.Path) -> str:
    completed = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to read git commit for {repo_root}:\n{completed.stdout}")
    return completed.stdout.strip()


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("PyYAML is required to rewrite Hunyuan Image3 deploy configs.") from exc
    return yaml.safe_load(path.read_text())


def dump_yaml(config: dict[str, Any], path: pathlib.Path) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("PyYAML is required to rewrite Hunyuan Image3 deploy configs.") from exc
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _parse_devices(devices: str) -> list[str]:
    parsed = [device.strip() for device in devices.split(",") if device.strip()]
    if not parsed:
        raise ValueError("--devices must contain at least one CUDA device id.")
    return parsed


def write_temp_dit_deploy(
    repo_root: pathlib.Path,
    output_dir: pathlib.Path,
    devices: str,
    quantization: str | None,
) -> pathlib.Path:
    source = repo_root / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"DiT-only deploy config does not exist: {source}")
    config = load_yaml(source)
    stages = config.get("stages") or []
    if len(stages) != 1:
        raise ValueError(f"Expected one DiT stage in {source}, found {len(stages)}.")
    device_ids = _parse_devices(devices)
    stage0 = stages[0]
    stage0["devices"] = ",".join(device_ids)
    stage0.setdefault("parallel_config", {})["tensor_parallel_size"] = len(device_ids)
    if quantization is not None:
        stage0["quantization"] = quantization
    path = output_dir / f"deploy_{repo_root.name}_{len(device_ids)}gpu.yaml"
    dump_yaml(config, path)
    return path


def build_env(args: argparse.Namespace, variant: Variant) -> dict[str, str]:
    env = os.environ.copy()
    # This benchmark compares HF HunyuanImage-3.0 paths. A truthy inherited
    # VLLM_USE_MODELSCOPE makes Omni resolve the model through ModelScope, where
    # the Tencent HF repo id is not available.
    env.pop("VLLM_USE_MODELSCOPE", None)
    env.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if args.hf_home:
        env["HF_HOME"] = str(pathlib.Path(args.hf_home).expanduser())
    if not args.allow_download:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

    pythonpath_parts = [str(variant.repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if args.cuda_compat_dir:
        compat_dir = str(pathlib.Path(args.cuda_compat_dir).expanduser())
        ld_path_parts = [compat_dir]
        cu13_runtime_dir = (
            pathlib.Path(sys.executable).resolve().parents[1]
            / "lib"
            / (f"python{sys.version_info.major}.{sys.version_info.minor}")
            / "site-packages"
            / "nvidia"
            / "cu13"
            / "lib"
        )
        if cu13_runtime_dir.is_dir():
            ld_path_parts.append(str(cu13_runtime_dir))
        existing_ld_path = env.get("LD_LIBRARY_PATH")
        if existing_ld_path:
            ld_path_parts.append(existing_ld_path)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_path_parts)

    if variant.paged:
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] = "required"
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"] = str(args.page_size)
        env.setdefault("VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS", "1")
        if args.profile_paged_kv:
            env["VLLM_OMNI_HY3_PAGED_KV_PROFILE"] = "1"
        else:
            env.pop("VLLM_OMNI_HY3_PAGED_KV_PROFILE", None)
    else:
        env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] = "0"
        env.pop("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", None)
        env.pop("VLLM_OMNI_HY3_PAGED_KV_PROFILE", None)
    return env


def build_command(
    args: argparse.Namespace,
    variant: Variant,
    deploy_config: pathlib.Path,
    output_dir: pathlib.Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(variant.repo_root / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"),
        "--modality",
        "text2img",
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
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--bot-task",
        "none",
        "--sys-type",
        "en_unified",
        "--enforce-eager",
    ]
    if variant.paged:
        cmd.extend(
            [
                "--require-paged-kv-cache",
                "--paged-kv-cache-page-size",
                str(args.page_size),
                "--print-paged-kv-stats",
            ]
        )
        if args.profile_paged_kv:
            cmd.append("--profile-paged-kv")
    return cmd


def parse_prefixed_json(output: str, prefix: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = None
    for line in output.splitlines():
        if line.startswith(prefix):
            parsed = json.loads(line[len(prefix) :])
    return parsed


def parse_saved_images(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.splitlines():
        match = SAVED_IMAGE_RE.match(line)
        if match is not None:
            paths.append(match.group("path"))
    return paths


def preflight_variant(
    args: argparse.Namespace,
    variant: Variant,
    deploy_config: pathlib.Path,
    env: dict[str, str],
) -> None:
    paged_probe = ""
    if variant.preflight_paged_probe:
        paged_probe = """
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
B, S, H, D, BS = 1, 8, 2, 16, 16
q = torch.randn((B * S, H, D), device="cuda", dtype=torch.float16)
k = torch.randn((B * S, H, D), device="cuda", dtype=torch.float16)
v = torch.randn((B * S, H, D), device="cuda", dtype=torch.float16)
key_cache = torch.empty((1, BS, H, D), device="cuda", dtype=torch.float16)
value_cache = torch.empty((1, BS, H, D), device="cuda", dtype=torch.float16)
slot_mapping = torch.arange(S, device="cuda", dtype=torch.int64)
block_table = torch.zeros((B, 1), device="cuda", dtype=torch.int32)
cu_q = torch.tensor([0, S], device="cuda", dtype=torch.int32)
seq = torch.tensor([S], device="cuda", dtype=torch.int32)
one = torch.ones((), device="cuda", dtype=torch.float32)
reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping, "auto", one, one)
out = flash_attn_varlen_func(
    q,
    key_cache,
    value_cache,
    cu_seqlens_q=cu_q,
    max_seqlen_q=S,
    seqused_k=seq,
    max_seqlen_k=S,
    softmax_scale=D ** -0.5,
    causal=False,
    block_table=block_table,
)
torch.cuda.synchronize()
print("vllm_paged_fa_probe_ok=" + str(tuple(out.shape)))
"""
    code = f"""
import pathlib
import torch
from vllm_omni.entrypoints.omni import Omni
print("repo_root={variant.repo_root}")
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_device_count=" + str(torch.cuda.device_count()))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
probe = torch.empty((1,), device="cuda:0")
probe.zero_()
torch.cuda.synchronize(0)
print("omni_import_ok=" + Omni.__name__)
print("deploy_config_exists=" + str(pathlib.Path({str(deploy_config)!r}).is_file()))
{paged_probe}
"""
    completed = _run([sys.executable, "-c", code], cwd=variant.repo_root, env=env, timeout=args.timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"Preflight failed for {variant.name}:\n{completed.stdout}")


def run_case(
    args: argparse.Namespace,
    variant: Variant,
    deploy_config: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args, variant)
    cmd = build_command(args, variant, deploy_config, output_dir)
    started = time.perf_counter()
    completed = _run(cmd, cwd=variant.repo_root, env=env, timeout=args.timeout)
    wall_time_s = time.perf_counter() - started
    output = completed.stdout or ""
    stdout_path = output_dir / "stdout.log"
    stdout_path.write_text(output)
    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(
            f"Benchmark failed for variant={variant.name} warmup={warmup} run_index={run_index} "
            f"returncode={completed.returncode}\n{tail}"
        )
    return {
        "variant": variant.name,
        "repo_root": str(variant.repo_root),
        "commit": git_commit(variant.repo_root),
        "paged": variant.paged,
        "warmup": warmup,
        "run_index": run_index,
        "outer_wall_time_s": wall_time_s,
        "benchmark_metrics": parse_prefixed_json(output, BENCHMARK_METRICS_PREFIX) or {},
        "paged_kv_stats": parse_prefixed_json(output, PAGED_KV_STATS_PREFIX),
        "output_images": parse_saved_images(output),
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "command": cmd,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in sorted({result["variant"] for result in results}):
        selected = [result for result in results if result["variant"] == variant and not result["warmup"]]
        outer_times = [float(result["outer_wall_time_s"]) for result in selected]
        generation_times = [
            float(result["benchmark_metrics"]["generation_wall_time_s"])
            for result in selected
            if result.get("benchmark_metrics", {}).get("generation_wall_time_s") is not None
        ]
        rows.append(
            {
                "variant": variant,
                "commit": selected[-1]["commit"] if selected else None,
                "outer_wall_time_s_mean": _mean(outer_times),
                "generation_wall_time_s_mean": _mean(generation_times),
                "runs": len(selected),
            }
        )
    by_variant = {row["variant"]: row for row in rows}
    main_time = by_variant.get("main_dense", {}).get("outer_wall_time_s_mean")
    dit_dense_time = by_variant.get("dit_dense", {}).get("outer_wall_time_s_mean")
    paged_time = by_variant.get("dit_paged", {}).get("outer_wall_time_s_mean")
    main_speedup = main_time / paged_time if main_time is not None and paged_time not in (None, 0.0) else None
    dit_speedup = dit_dense_time / paged_time if dit_dense_time is not None and paged_time not in (None, 0.0) else None
    return {
        "rows": rows,
        "speedup_main_dense_over_dit_paged_outer_wall": main_speedup,
        "speedup_dit_dense_over_dit_paged_outer_wall": dit_speedup,
    }


def write_summary(path: pathlib.Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# Hunyuan Image3 Main Dense vs DiT Paged Benchmark",
        "",
        "## Summary",
        "",
        "| Variant | Commit | Outer wall s | Generation wall s | Runs |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        gen_time = row["generation_wall_time_s_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["commit"]),
                    f"{row['outer_wall_time_s_mean']:.3f}" if row["outer_wall_time_s_mean"] is not None else "n/a",
                    f"{gen_time:.3f}" if gen_time is not None else "n/a",
                    str(row["runs"]),
                ]
            )
            + " |"
        )
    speedup = summary.get("speedup_main_dense_over_dit_paged_outer_wall")
    dit_speedup = summary.get("speedup_dit_dense_over_dit_paged_outer_wall")
    lines.extend(
        [
            "",
            f"- speedup main_dense / dit_paged by outer wall: `{speedup:.3f}x`" if speedup else "- speedup: `n/a`",
            f"- speedup dit_dense / dit_paged by outer wall: `{dit_speedup:.3f}x`"
            if dit_speedup
            else "- dit_dense speedup: `n/a`",
            f"- model: `{args.model}`",
            f"- steps: `{args.steps}`",
            f"- image size: `{args.width}x{args.height}`",
            f"- guidance scale: `{args.guidance_scale}`",
            f"- devices: `{args.devices}`",
            f"- cuda compat dir: `{args.cuda_compat_dir}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 to exercise later denoise KV reuse.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    for label, raw_path in (("main", args.main_worktree), ("dit", args.dit_worktree)):
        path = pathlib.Path(raw_path).expanduser()
        if not (path / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py").is_file():
            raise FileNotFoundError(f"{label} worktree does not look like vllm-omni: {path}")
    if args.cuda_compat_dir is not None and not pathlib.Path(args.cuda_compat_dir).expanduser().is_dir():
        raise FileNotFoundError(f"--cuda-compat-dir does not exist: {args.cuda_compat_dir}")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    main_worktree = pathlib.Path(args.main_worktree).expanduser().resolve()
    dit_worktree = pathlib.Path(args.dit_worktree).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir or tempfile.mkdtemp(prefix="hy3-branch-benchmark-")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [Variant("main_dense", main_worktree, paged=False)]
    if args.include_dit_dense:
        variants.append(Variant("dit_dense", dit_worktree, paged=False, preflight_paged_probe=True))
    variants.append(Variant("dit_paged", dit_worktree, paged=True, preflight_paged_probe=True))
    deploy_configs = {
        variant.name: write_temp_dit_deploy(variant.repo_root, output_dir, args.devices, args.quantization)
        for variant in variants
    }

    metadata = {
        "main_worktree": str(main_worktree),
        "dit_worktree": str(dit_worktree),
        "main_commit": git_commit(main_worktree),
        "dit_commit": git_commit(dit_worktree),
        "args": vars(args),
        "deploy_configs": {name: str(path) for name, path in deploy_configs.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    if args.dry_run:
        commands = []
        for variant in variants:
            env = build_env(args, variant)
            commands.append(
                {
                    "variant": variant.name,
                    "command": build_command(
                        args,
                        variant,
                        deploy_configs[variant.name],
                        output_dir / variant.name / "run_0",
                    ),
                    "env_subset": {
                        key: env[key]
                        for key in (
                            "PYTHONPATH",
                            "LD_LIBRARY_PATH",
                            "HF_HOME",
                            "HF_ENDPOINT",
                            "HF_HUB_OFFLINE",
                            "TRANSFORMERS_OFFLINE",
                            "VLLM_OMNI_HY3_PAGED_KV_CACHE",
                            "VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE",
                            "VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS",
                            "VLLM_OMNI_HY3_PAGED_KV_PROFILE",
                        )
                        if key in env
                    },
                }
            )
        print(json.dumps({"metadata": metadata, "commands": commands}, indent=2, sort_keys=True))
        return

    try:
        for variant in variants:
            preflight_variant(args, variant, deploy_configs[variant.name], build_env(args, variant))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    if args.preflight_only:
        print(f"Hunyuan Image3 branch benchmark preflight passed. output_dir={output_dir}")
        return

    results: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for warmup_index in range(args.warmup):
                print(f"[Branch Benchmark] warmup variant={variant.name} index={warmup_index}")
                results.append(
                    run_case(
                        args,
                        variant,
                        deploy_configs[variant.name],
                        output_dir / variant.name / f"warmup_{warmup_index}",
                        run_index=warmup_index,
                        warmup=True,
                    )
                )
            for run_index in range(args.repeat):
                print(f"[Branch Benchmark] run variant={variant.name} index={run_index}")
                results.append(
                    run_case(
                        args,
                        variant,
                        deploy_configs[variant.name],
                        output_dir / variant.name / f"run_{run_index}",
                        run_index=run_index,
                        warmup=False,
                    )
                )
        summary = summarize(results)
        payload = {"metadata": metadata, "results": results, "summary": summary}
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        write_summary(output_dir / "summary.md", args, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[Branch Benchmark] output_dir={output_dir}")
    except RuntimeError as exc:
        (output_dir / "partial_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
