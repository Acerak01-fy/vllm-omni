# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Hunyuan Image3 paged KV smoke wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SMOKE_TOOL = _REPO_ROOT / "tools" / "hunyuan_image3_paged_kv_smoke.py"


def _load_smoke_tool():
    spec = importlib.util.spec_from_file_location("hunyuan_image3_paged_kv_smoke", _SMOKE_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_model_available_locally_accepts_checkpoint_path(tmp_path):
    tool = _load_smoke_tool()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    assert tool.model_available_locally(str(checkpoint_dir))


def test_hf_snapshot_cached_uses_custom_hf_home(tmp_path):
    tool = _load_smoke_tool()
    hf_home = tmp_path / "hf"
    snapshot = hf_home / "hub" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    assert tool.hf_snapshot_cached("org/model", hf_home=str(hf_home))
    assert tool.model_available_locally("org/model", hf_home=str(hf_home))


def test_validate_download_space_rejects_insufficient_cache_disk(monkeypatch, tmp_path):
    tool = _load_smoke_tool()
    gib = 1024**3
    hf_home = tmp_path / "hf"

    monkeypatch.setattr(tool, "estimate_hf_model_size_bytes", lambda _model_id, _endpoint: 20 * gib)
    monkeypatch.setattr(tool.shutil, "disk_usage", lambda _path: SimpleNamespace(free=25 * gib))

    with pytest.raises(RuntimeError, match="Not enough free disk"):
        tool.validate_download_space(
            model_id="org/model",
            hf_home=hf_home,
            endpoint="https://hf-mirror.com",
            safety_factor=1.0,
            min_extra_bytes=10 * gib,
        )


def test_validate_model_download_preflight_skips_cached_snapshot(monkeypatch, tmp_path):
    tool = _load_smoke_tool()
    hf_home = tmp_path / "hf"
    snapshot = hf_home / "hub" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    args = argparse.Namespace(
        model="org/model",
        hf_home=str(hf_home),
        hf_endpoint="https://hf-mirror.com",
    )

    monkeypatch.setattr(
        tool,
        "estimate_hf_model_size_bytes",
        lambda *_args, **_kwargs: pytest.fail("cached snapshot should not query remote metadata"),
    )

    tool.validate_model_download_preflight(args)


def test_validate_args_rejects_single_step_smoke(tmp_path):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(deploy_config, {"stages": [{"stage_id": 0}]})
    args = argparse.Namespace(
        steps=1,
        page_size=16,
        deploy_config=str(deploy_config),
        devices=None,
        quantization=None,
    )

    with pytest.raises(ValueError, match="at least 2"):
        tool.validate_args(args)


def test_collect_deploy_device_ids_from_all_stages():
    tool = _load_smoke_tool()
    config = {
        "stages": [
            {"stage_id": 0, "devices": "0,1"},
            {"stage_id": 1, "devices": "2,3"},
        ]
    }

    assert tool.collect_deploy_device_ids(config) == [0, 1, 2, 3]


def test_collect_deploy_device_ids_rejects_non_integer_ids():
    tool = _load_smoke_tool()
    config = {"stages": [{"stage_id": 0, "devices": "0,gpu1"}]}

    with pytest.raises(ValueError, match="must be integers"):
        tool.collect_deploy_device_ids(config)


def test_validate_cuda_devices_rejects_missing_devices():
    tool = _load_smoke_tool()

    with pytest.raises(RuntimeError, match="unavailable CUDA device ids"):
        tool.validate_cuda_devices([0, 2], cuda_device_count=2)


def test_validate_cuda_devices_rejects_cpu_only_runtime():
    tool = _load_smoke_tool()

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        tool.validate_cuda_devices([0], cuda_device_count=0)


def test_validate_cuda_runtime_rejects_unusable_driver():
    tool = _load_smoke_tool()

    class FakeCuda:
        @staticmethod
        def synchronize(_device_id):
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def empty(_shape, *, device):
            assert device == "cuda:0"
            raise RuntimeError("driver is too old")

    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        tool.validate_cuda_runtime(FakeTorch, [0])


def test_validate_python_imports_reports_missing_runtime_api():
    tool = _load_smoke_tool()

    with pytest.raises(RuntimeError, match="definitely_missing_hy3_smoke_module"):
        tool.validate_python_imports((("missing-test-module", "import definitely_missing_hy3_smoke_module"),))


def test_write_temp_deploy_config_rewrites_single_stage_devices(tmp_path):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(
        deploy_config,
        {
            "pipeline": "hunyuan_image3_dit",
            "stages": [
                {
                    "stage_id": 0,
                    "devices": "0,1,2,3",
                    "parallel_config": {"tensor_parallel_size": 4},
                }
            ],
        },
    )

    temp_path = tool.write_temp_deploy_config(deploy_config, "0,1")
    try:
        rewritten = yaml.safe_load(temp_path.read_text())
    finally:
        temp_path.unlink(missing_ok=True)

    stage0 = rewritten["stages"][0]
    assert stage0["devices"] == "0,1"
    assert stage0["parallel_config"]["tensor_parallel_size"] == 2


def test_write_temp_deploy_config_rewrites_single_stage_quantization(tmp_path):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(
        deploy_config,
        {
            "pipeline": "hunyuan_image3_dit",
            "stages": [
                {
                    "stage_id": 0,
                    "devices": "0,1",
                    "parallel_config": {"tensor_parallel_size": 2},
                }
            ],
        },
    )

    temp_path = tool.write_temp_deploy_config(deploy_config, quantization="fp8")
    try:
        rewritten = yaml.safe_load(temp_path.read_text())
    finally:
        temp_path.unlink(missing_ok=True)

    stage0 = rewritten["stages"][0]
    assert stage0["devices"] == "0,1"
    assert stage0["parallel_config"]["tensor_parallel_size"] == 2
    assert stage0["quantization"] == "fp8"


def test_write_temp_deploy_config_rejects_multi_stage_devices(tmp_path):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(
        deploy_config,
        {
            "pipeline": "hunyuan_image_3_moe",
            "stages": [{"stage_id": 0}, {"stage_id": 1}],
        },
    )

    with pytest.raises(ValueError, match="single-stage"):
        tool.write_temp_deploy_config(deploy_config, "0,1")


def test_build_subprocess_env_adds_repo_root_to_pythonpath(monkeypatch):
    tool = _load_smoke_tool()
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    hf_home = "/tmp/hy3-hf-cache"
    args = argparse.Namespace(
        hf_endpoint="https://hf-mirror.com",
        hf_home=hf_home,
        allow_download=False,
        page_size=16,
    )

    env = tool.build_subprocess_env(args)

    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == str(tool.REPO_ROOT)
    assert "/existing/path" in pythonpath
    assert env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] == "required"
    assert env["VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == hf_home


def test_preflight_only_skips_model_check_and_subprocess(monkeypatch, tmp_path, capsys):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(deploy_config, {"stages": [{"stage_id": 0, "devices": "0"}]})

    monkeypatch.setattr(
        tool,
        "parse_args",
        lambda: argparse.Namespace(
            model="missing-model",
            deploy_config=str(deploy_config),
            devices=None,
            output=None,
            steps=2,
            page_size=16,
            quantization=None,
            allow_download=False,
            hf_home=None,
            preflight_only=True,
            dry_run=False,
            hf_endpoint="https://hf-mirror.com",
        ),
    )
    monkeypatch.setattr(
        tool,
        "model_available_locally",
        lambda _model: pytest.fail("preflight-only should not check model availability"),
    )
    monkeypatch.setattr(tool, "preflight_runtime", lambda _deploy_config: None)
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("preflight-only should not start inference"),
    )

    tool.main()

    assert "preflight passed" in capsys.readouterr().out


def test_dry_run_prints_command_without_model_or_runtime_checks(monkeypatch, tmp_path, capsys):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(deploy_config, {"stages": [{"stage_id": 0, "devices": "0"}]})

    monkeypatch.setattr(
        tool,
        "parse_args",
        lambda: argparse.Namespace(
            model="missing-model",
            deploy_config=str(deploy_config),
            devices=None,
            output="/tmp/hy3-proof",
            steps=2,
            guidance_scale=1.0,
            height=1024,
            width=1024,
            seed=42,
            prompt="A prompt",
            page_size=16,
            quantization="fp8",
            hf_home="/tmp/hy3-hf-cache",
            allow_download=False,
            preflight_only=False,
            dry_run=True,
            hf_endpoint="https://hf-mirror.com",
        ),
    )
    monkeypatch.setattr(
        tool,
        "model_available_locally",
        lambda _model: pytest.fail("dry-run should not check model availability"),
    )
    monkeypatch.setattr(tool, "preflight_runtime", lambda _deploy_config: pytest.fail("dry-run should not preflight"))
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("dry-run should not start inference"),
    )

    tool.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"].startswith("dry-run only")
    assert payload["model"] == "missing-model"
    assert "--dry-run" not in payload["command"]
    assert "--quantization" in payload["command"]
    assert payload["quantization_override"] == "fp8"
    assert "--require-paged-kv-cache" in payload["offline_required_flags"]
    assert "--paged-kv-cache-page-size" in payload["offline_required_flags"]
    assert payload["proof_env"]["VLLM_OMNI_HY3_PAGED_KV_CACHE"] == "required"
    assert payload["proof_env"]["VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS"] == "1"
    assert payload["proof_env"]["HF_HOME"] == "/tmp/hy3-hf-cache"
    assert payload["required_stats_gate"]["paged_attention_fallbacks"] == 0


def test_main_reports_preflight_value_errors_without_traceback(monkeypatch, tmp_path):
    tool = _load_smoke_tool()
    deploy_config = tmp_path / "deploy.yaml"
    _write_yaml(deploy_config, {"stages": [{"stage_id": 0, "devices": "gpu0"}]})

    monkeypatch.setattr(
        tool,
        "parse_args",
        lambda: argparse.Namespace(
            model=str(tmp_path),
            deploy_config=str(deploy_config),
            devices=None,
            output="/tmp/hy3-proof",
            steps=2,
            guidance_scale=1.0,
            height=1024,
            width=1024,
            seed=42,
            prompt="A prompt",
            page_size=16,
            quantization=None,
            hf_home=None,
            allow_download=False,
            preflight_only=False,
            dry_run=False,
            hf_endpoint="https://hf-mirror.com",
        ),
    )

    with pytest.raises(SystemExit, match="Device ids must be integers"):
        tool.main()
