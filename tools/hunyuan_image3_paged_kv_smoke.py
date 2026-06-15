# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a real Hunyuan Image3 paged KV smoke with required-mode assertions.

This wrapper intentionally refuses to download the model unless
``--allow-download`` is set. The checkpoint is large, and a missing model is a
validation blocker rather than a successful smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "tencent/HunyuanImage-3.0-Instruct"
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml"
OFFLINE_SCRIPT = REPO_ROOT / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"
REQUIRED_VLLM_IMPORTS = (
    ("vllm.v1.request.StreamingUpdate", "from vllm.v1.request import StreamingUpdate"),
    ("vllm.v1.attention.backend", "import vllm.v1.attention.backend"),
    ("vllm.ir", "import vllm.ir"),
    ("vllm.inputs.engine", "import vllm.inputs.engine"),
)
REQUIRED_OMNI_IMPORTS = (
    ("cache_dit", "import cache_dit"),
    ("vllm_omni.entrypoints.omni.Omni", "from vllm_omni.entrypoints.omni import Omni"),
)
PROOF_ENV_KEYS = (
    "HF_ENDPOINT",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "VLLM_OMNI_HY3_PAGED_KV_CACHE",
    "VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE",
    "VLLM_OMNI_HY3_PAGED_KV_PROFILE",
    "VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS",
    "PYTHONPATH",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hunyuan Image3 paged KV full-checkpoint smoke.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--devices", default=None, help="Optional comma-separated device list for a temp deploy file.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to a temporary directory.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A brown and white dog is running on the grass.")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional quantization value to write into the temporary single-stage DiT deploy config, e.g. fp8.",
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional HF_HOME for checkpoint cache/download. Useful when the default home cache lacks space.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate deploy/runtime requirements and exit before checking or loading the checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the copyable wrapper command and proof env without checking runtime/model or starting inference.",
    )
    return parser.parse_args()


def model_available_locally(model: str, hf_home: str | None = None) -> bool:
    model_path = pathlib.Path(model).expanduser()
    if model_path.exists():
        return True

    return hf_snapshot_cached(model, hf_home=hf_home)


def hf_snapshot_cached(model_id: str, hf_home: str | None = None) -> bool:
    hf_home = hf_home or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_root = pathlib.Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snap_root.is_dir() and any(snap_root.iterdir())


def _existing_parent(path: pathlib.Path) -> pathlib.Path:
    current = path.expanduser()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def _hf_home_for_args(args: argparse.Namespace) -> pathlib.Path:
    hf_home = getattr(args, "hf_home", None) or os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return pathlib.Path(hf_home).expanduser()


def estimate_hf_model_size_bytes(model_id: str, endpoint: str) -> int:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("huggingface_hub is required to preflight checkpoint download size.") from exc

    try:
        info = HfApi(endpoint=endpoint).model_info(model_id, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to query model metadata for {model_id!r} from {endpoint!r}.") from exc

    total_size = 0
    missing_sizes: list[str] = []
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size is None:
            missing_sizes.append(getattr(sibling, "rfilename", "<unknown>"))
            continue
        total_size += int(size)
    if missing_sizes:
        examples = ", ".join(missing_sizes[:3])
        raise RuntimeError(
            f"Model metadata for {model_id!r} is missing file sizes ({examples}); cannot safely preflight disk usage."
        )
    return total_size


def validate_download_space(
    *,
    model_id: str,
    hf_home: pathlib.Path,
    endpoint: str,
    safety_factor: float = 1.10,
    min_extra_bytes: int = 10 * 1024**3,
) -> None:
    required_bytes = int(estimate_hf_model_size_bytes(model_id, endpoint) * safety_factor) + min_extra_bytes
    usage_path = _existing_parent(hf_home)
    free_bytes = shutil.disk_usage(usage_path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Not enough free disk for checkpoint download into HF_HOME={hf_home}. "
            f"Need at least {required_bytes / 1024**3:.1f} GiB "
            f"(model size plus safety margin), found {free_bytes / 1024**3:.1f} GiB "
            f"on {usage_path}."
        )


def validate_model_download_preflight(args: argparse.Namespace) -> None:
    model_path = pathlib.Path(args.model).expanduser()
    if model_path.exists() or hf_snapshot_cached(args.model, hf_home=getattr(args, "hf_home", None)):
        return
    validate_download_space(
        model_id=args.model,
        hf_home=_hf_home_for_args(args),
        endpoint=args.hf_endpoint,
    )


def _parse_devices(devices: str) -> list[str]:
    parsed = [device.strip() for device in devices.split(",") if device.strip()]
    if not parsed:
        raise ValueError("--devices must contain at least one device id.")
    return parsed


def _parse_device_ids(devices: str) -> list[int]:
    parsed = _parse_devices(devices)
    try:
        return [int(device) for device in parsed]
    except ValueError as exc:
        raise ValueError(f"Device ids must be integers, got {devices!r}.") from exc


def collect_deploy_device_ids(config: dict[str, Any]) -> list[int]:
    device_ids: set[int] = set()
    for stage in config.get("stages") or []:
        devices = stage.get("devices")
        if devices is None:
            continue
        device_ids.update(_parse_device_ids(str(devices)))
    return sorted(device_ids)


def validate_cuda_devices(device_ids: list[int], cuda_device_count: int) -> None:
    if cuda_device_count <= 0:
        raise RuntimeError("CUDA is not available; Hunyuan Image3 paged KV smoke requires CUDA GPUs.")
    missing = [device_id for device_id in device_ids if device_id < 0 or device_id >= cuda_device_count]
    if missing:
        raise RuntimeError(
            f"Deploy config references unavailable CUDA device ids {missing}; "
            f"visible CUDA device count is {cuda_device_count}."
        )


def validate_cuda_runtime(torch_module: Any, device_ids: list[int]) -> None:
    probe_device_id = min(device_ids) if device_ids else 0
    try:
        probe = torch_module.empty((1,), device=f"cuda:{probe_device_id}")
        probe.zero_()
        torch_module.cuda.synchronize(probe_device_id)
    except Exception as exc:
        raise RuntimeError(
            "CUDA runtime is not usable; Hunyuan Image3 paged KV smoke requires "
            "a PyTorch CUDA build compatible with the installed NVIDIA driver."
        ) from exc


def validate_python_imports(required_imports: tuple[tuple[str, str], ...]) -> None:
    failures: list[str] = []
    for label, statement in required_imports:
        try:
            exec(statement, {})
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Required Python imports failed: {joined}")


def validate_vllm_runtime() -> None:
    try:
        validate_python_imports(REQUIRED_VLLM_IMPORTS)
    except RuntimeError as exc:
        raise RuntimeError(
            f"vLLM runtime is incompatible with the current vLLM-Omni Hunyuan Image3 smoke. {exc}"
        ) from exc


def validate_omni_runtime() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        validate_python_imports(REQUIRED_OMNI_IMPORTS)
    except RuntimeError as exc:
        raise RuntimeError(f"vLLM-Omni runtime imports are incomplete for the Hunyuan Image3 smoke. {exc}") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 so later denoise paged KV reuse is exercised.")
    if args.page_size <= 0:
        raise ValueError(f"--page-size must be positive, got {args.page_size}.")
    deploy_config = pathlib.Path(args.deploy_config)
    if not deploy_config.is_file():
        raise FileNotFoundError(f"Deploy config does not exist: {deploy_config}")
    if args.devices is not None:
        _parse_devices(args.devices)
    if args.devices is not None or getattr(args, "quantization", None) is not None:
        config = load_deploy_config(deploy_config)
        stages = config.get("stages") or []
        if len(stages) != 1:
            raise ValueError(
                "Temporary deploy rewriting is supported only for the DiT-only single-stage deploy config. "
                f"Found {len(stages)} stages in {deploy_config}."
            )


def load_deploy_config(source: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("PyYAML is required to inspect or rewrite the deploy config.") from exc

    return yaml.safe_load(source.read_text())


def preflight_runtime(deploy_config: pathlib.Path) -> None:
    config = load_deploy_config(deploy_config)
    device_ids = collect_deploy_device_ids(config)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for the Hunyuan Image3 paged KV smoke.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Hunyuan Image3 paged KV smoke requires CUDA GPUs.")
    validate_cuda_devices(device_ids, torch.accelerator.device_count())
    validate_cuda_runtime(torch, device_ids)
    validate_vllm_runtime()
    validate_omni_runtime()
    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,  # noqa: F401
            is_flash_attn_varlen_func_available,
            reshape_and_cache_flash,  # noqa: F401
        )
    except Exception as exc:
        raise RuntimeError(
            "vLLM FlashAttention paged prefill APIs are required: "
            "vllm.v1.attention.backends.fa_utils.reshape_and_cache_flash and flash_attn_varlen_func."
        ) from exc
    if not is_flash_attn_varlen_func_available():
        raise RuntimeError("vLLM flash_attn_varlen_func is unavailable in this runtime.")


def write_temp_deploy_config(
    source: pathlib.Path,
    devices: str | None = None,
    *,
    quantization: str | None = None,
) -> pathlib.Path:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("--devices requires PyYAML to rewrite the deploy config.") from exc

    config = load_deploy_config(source)
    stages = config.get("stages") or []
    if len(stages) != 1:
        raise ValueError(
            "Temporary deploy rewriting is supported only for the DiT-only single-stage deploy config. "
            f"Found {len(stages)} stages in {source}."
        )
    stage0: dict[str, Any] = config["stages"][0]
    if devices is not None:
        device_ids = _parse_devices(devices)
        stage0["devices"] = devices
        stage0.setdefault("parallel_config", {})["tensor_parallel_size"] = len(device_ids)
    if quantization is not None:
        stage0["quantization"] = quantization
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)
    return pathlib.Path(tmp.name)


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if getattr(args, "hf_home", None):
        env["HF_HOME"] = str(pathlib.Path(args.hf_home).expanduser())
    if not args.allow_download:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] = "required"
    env["VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"] = str(args.page_size)
    env.setdefault("VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS", "1")

    pythonpath_parts = [str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def build_smoke_command(args: argparse.Namespace, deploy_config: pathlib.Path, output_dir: str) -> list[str]:
    return [
        sys.executable,
        str(OFFLINE_SCRIPT),
        "--modality",
        "text2img",
        "--model",
        args.model,
        "--deploy-config",
        str(deploy_config),
        "--prompts",
        args.prompt,
        "--output",
        output_dir,
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
        "--require-paged-kv-cache",
        "--paged-kv-cache-page-size",
        str(args.page_size),
        "--print-paged-kv-stats",
    ]


def build_wrapper_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--model",
        args.model,
        "--deploy-config",
        str(args.deploy_config),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--prompt",
        args.prompt,
        "--page-size",
        str(args.page_size),
        "--hf-endpoint",
        args.hf_endpoint,
    ]
    if args.quantization is not None:
        cmd.extend(["--quantization", str(args.quantization)])
    if getattr(args, "hf_home", None) is not None:
        cmd.extend(["--hf-home", str(args.hf_home)])
    if args.devices is not None:
        cmd.extend(["--devices", args.devices])
    if args.output is not None:
        cmd.extend(["--output", args.output])
    if args.allow_download:
        cmd.append("--allow-download")
    return cmd


def proof_env_subset(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in PROOF_ENV_KEYS if key in env}


def print_dry_run_plan(args: argparse.Namespace, deploy_config: pathlib.Path, env: dict[str, str]) -> None:
    payload = {
        "validation": "dry-run only; no runtime, checkpoint, or inference validation was executed",
        "model": args.model,
        "deploy_config": str(deploy_config),
        "devices_override": args.devices,
        "quantization_override": args.quantization,
        "effective_deploy_config": (
            "temporary single-stage deploy rewrite at runtime"
            if args.devices is not None or args.quantization is not None
            else str(deploy_config)
        ),
        "allow_download": bool(args.allow_download),
        "command": build_wrapper_command(args),
        "offline_required_flags": [
            "--require-paged-kv-cache",
            "--paged-kv-cache-page-size",
            "--print-paged-kv-stats",
        ],
        "proof_env": proof_env_subset(env),
        "required_stats_gate": {
            "paged_kv_cache_required": True,
            "paged_cache_builds_per_layer": "> 0",
            "paged_attention_calls_per_layer": "> 0",
            "paged_attention_fallbacks": 0,
            "paged_attention_runner_errors": 0,
            "paged_cache_build_failures": 0,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    deploy_config = pathlib.Path(args.deploy_config)
    if args.dry_run:
        env = build_subprocess_env(args)
        print_dry_run_plan(args, deploy_config, env)
        return

    if not args.preflight_only:
        if args.allow_download:
            try:
                validate_model_download_preflight(args)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from None
        elif not model_available_locally(args.model, hf_home=getattr(args, "hf_home", None)):
            raise SystemExit(
                f"Model {args.model!r} is not present in the Hugging Face cache. "
                "Run with --allow-download in an environment with enough disk, or pre-populate HF_HOME."
            )

    temp_deploy_config: pathlib.Path | None = None
    if args.devices is not None or args.quantization is not None:
        temp_deploy_config = write_temp_deploy_config(
            deploy_config,
            args.devices,
            quantization=args.quantization,
        )
        deploy_config = temp_deploy_config
    try:
        env = build_subprocess_env(args)
        preflight_runtime(deploy_config)
    except (RuntimeError, ValueError) as exc:
        if temp_deploy_config is not None:
            temp_deploy_config.unlink(missing_ok=True)
        raise SystemExit(str(exc)) from None
    if args.preflight_only:
        if temp_deploy_config is not None:
            temp_deploy_config.unlink(missing_ok=True)
        print("Hunyuan Image3 paged KV smoke preflight passed.")
        return

    output_dir = args.output or tempfile.mkdtemp(prefix="hy3-paged-kv-smoke-")
    cmd = build_smoke_command(args, deploy_config, output_dir)
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        if temp_deploy_config is not None:
            temp_deploy_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
