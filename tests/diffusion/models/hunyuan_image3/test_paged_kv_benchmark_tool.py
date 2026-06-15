# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Hunyuan Image3 paged KV benchmark wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BENCHMARK_TOOL = _REPO_ROOT / "tools" / "hunyuan_image3_paged_kv_benchmark.py"


def _load_benchmark_tool():
    spec = importlib.util.spec_from_file_location("hunyuan_image3_paged_kv_benchmark", _BENCHMARK_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "model": "tencent/HunyuanImage-3.0-Instruct",
        "deploy_config": str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml"),
        "devices": "0,1",
        "quantization": "fp8",
        "output_dir": "/tmp/hy3-bench",
        "steps": 50,
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "prompt": "A prompt",
        "cfg_guidance_scale": 5.0,
        "page_size": 16,
        "repeat": 1,
        "warmup": 0,
        "scenarios": "text2img-nocfg,img2img-3ref-cfg",
        "variants": "dense,paged",
        "reference_images": None,
        "bot_task": "none",
        "sys_type": "en_unified",
        "hf_endpoint": "https://hf-mirror.com",
        "hf_home": "/tmp/hy3-hf-cache",
        "allow_download": False,
        "preflight_only": False,
        "dry_run": False,
        "profile_paged_kv": False,
        "timeout": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolve_scenarios_expands_cfg_and_reference_counts():
    tool = _load_benchmark_tool()

    scenarios = tool.resolve_scenarios("text2img-cfg,img2img-3ref-cfg", 4.5)

    assert [scenario.name for scenario in scenarios] == ["text2img-cfg", "img2img-3ref-cfg"]
    assert scenarios[0].guidance_scale == 4.5
    assert scenarios[0].reference_count == 0
    assert scenarios[1].modality == "img2img"
    assert scenarios[1].reference_count == 3


def test_resolve_scenarios_rejects_unknown_name():
    tool = _load_benchmark_tool()

    with pytest.raises(ValueError, match="Unknown scenarios"):
        tool.resolve_scenarios("missing-scenario", 5.0)


def test_build_variant_env_forces_dense_off_and_paged_required(monkeypatch):
    tool = _load_benchmark_tool()
    args = _args()
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "required")

    dense_env = tool.build_variant_env(args, "dense")
    paged_env = tool.build_variant_env(args, "paged")

    assert dense_env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] == "0"
    assert "VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE" not in dense_env
    assert paged_env["VLLM_OMNI_HY3_PAGED_KV_CACHE"] == "required"
    assert paged_env["VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"] == "16"
    assert paged_env["VLLM_OMNI_HY3_PAGED_KV_VALIDATE_RUN_INPUTS"] == "1"
    assert dense_env["PYTHONPATH"].split(os.pathsep)[0] == str(tool.smoke.REPO_ROOT)
    assert "/existing" in dense_env["PYTHONPATH"].split(os.pathsep)


def test_build_variant_env_enables_profile_for_both_variants():
    tool = _load_benchmark_tool()
    args = _args(profile_paged_kv=True)

    dense_env = tool.build_variant_env(args, "dense")
    paged_env = tool.build_variant_env(args, "paged")

    assert dense_env["VLLM_OMNI_HY3_PAGED_KV_PROFILE"] == "1"
    assert paged_env["VLLM_OMNI_HY3_PAGED_KV_PROFILE"] == "1"


def test_build_offline_command_sets_img2img_refs_and_paged_flags(tmp_path):
    tool = _load_benchmark_tool()
    args = _args(prompt="Edit the image")
    deploy_config = tmp_path / "deploy.yaml"
    refs = [tmp_path / f"ref_{idx}.png" for idx in range(3)]
    scenario = tool.scenario_catalog(5.0)["img2img-3ref-cfg"]

    cmd = tool.build_offline_command(args, deploy_config, scenario, "paged", tmp_path / "out", refs)

    assert "--modality" in cmd
    assert cmd[cmd.index("--modality") + 1] == "img2img"
    assert "--image-path" in cmd
    assert cmd[cmd.index("--image-path") + 1] == ",".join(str(path) for path in refs)
    assert "--require-paged-kv-cache" in cmd
    assert "--paged-kv-cache-page-size" in cmd
    assert cmd[cmd.index("--guidance-scale") + 1] == "5.0"


def test_build_offline_command_dense_prints_stats_without_required_flags(tmp_path):
    tool = _load_benchmark_tool()
    args = _args()
    scenario = tool.scenario_catalog(5.0)["text2img-nocfg"]

    cmd = tool.build_offline_command(args, tmp_path / "deploy.yaml", scenario, "dense", tmp_path / "out", [])

    assert "--print-paged-kv-stats" in cmd
    assert "--require-paged-kv-cache" not in cmd
    assert "--paged-kv-cache-page-size" not in cmd
    assert "--image-path" not in cmd


def test_build_offline_command_forwards_profile_flag(tmp_path):
    tool = _load_benchmark_tool()
    args = _args(profile_paged_kv=True)
    scenario = tool.scenario_catalog(5.0)["text2img-nocfg"]

    cmd = tool.build_offline_command(args, tmp_path / "deploy.yaml", scenario, "dense", tmp_path / "out", [])

    assert "--profile-paged-kv" in cmd


def test_parse_metrics_and_derive_paged_page_hit_rate():
    tool = _load_benchmark_tool()
    stats = {
        "enabled_layers": 2,
        "paged_attention_calls": 4,
        "paged_attention_expected_calls": 4,
        "paged_attention_reuse_coverage": 1.0,
        "paged_kv_prefix_page_hits": 12,
        "paged_kv_prefix_page_lookups": 12,
        "paged_kv_prefix_token_hits": 60,
        "paged_kv_prefix_token_lookups": 60,
        "paged_kv_page_table_entries": 20,
        "paged_kv_current_page_entries": 8,
        "paged_kv_cached_tokens": 30,
        "paged_kv_max_cached_tokens": 20,
        "paged_kv_num_pages": 4,
        "paged_kv_prefix_pages": 3,
        "profile_vllm_paged_attention_calls": 4,
        "profile_vllm_paged_attention_total_ms": 8.0,
        "layers_detail": [
            {"paged_kv_cached_tokens": 10, "paged_attention_calls": 2},
            {"paged_kv_cached_tokens": 20, "paged_attention_calls": 2},
        ],
    }
    output = "\n".join(
        [
            'noise {"ignored": true}',
            tool.BENCHMARK_METRICS_PREFIX + json.dumps({"generation_wall_time_s": 12.5}),
            tool.PAGED_KV_STATS_PREFIX + json.dumps(stats),
            "[Output] Saved image to /tmp/out.png",
        ]
    )

    assert tool.parse_prefixed_json(output, tool.BENCHMARK_METRICS_PREFIX)["generation_wall_time_s"] == 12.5
    assert tool.parse_prefixed_json(output, tool.PAGED_KV_STATS_PREFIX) == stats
    assert tool.parse_saved_images(output) == ["/tmp/out.png"]
    derived = tool.derive_paged_kv_metrics(stats, steps=3)
    assert derived["paged_attention_actual_calls"] == 4
    assert derived["paged_attention_expected_calls"] == 4
    assert derived["paged_attention_hit_rate"] == 1.0
    assert derived["paged_kv_prefix_page_hit_rate"] == 1.0
    assert derived["paged_kv_prefix_token_hit_rate"] == 1.0
    assert derived["paged_kv_prefix_page_hits"] == 12
    assert derived["paged_kv_prefix_page_lookups"] == 12
    assert derived["paged_kv_cached_token_uses"] == 60
    assert derived["paged_attention_reuse_coverage"] == 1.0
    assert derived["profile_vllm_paged_attention_avg_ms"] == 2.0


def test_summarize_results_computes_generation_speedup():
    tool = _load_benchmark_tool()
    results = [
        {
            "scenario": "img2img-3ref-cfg",
            "variant": "dense",
            "warmup": False,
            "generation_wall_time_s": 20.0,
            "wall_time_s": 25.0,
        },
        {
            "scenario": "img2img-3ref-cfg",
            "variant": "paged",
            "warmup": False,
            "generation_wall_time_s": 10.0,
            "wall_time_s": 15.0,
            "derived_paged_kv_metrics": {
                "paged_attention_hit_rate": 1.0,
                "paged_attention_reuse_coverage": 1.0,
                "paged_kv_prefix_page_hit_rate": 1.0,
                "paged_kv_prefix_token_hit_rate": 1.0,
                "paged_attention_actual_calls": 1568,
                "paged_attention_expected_calls": 1568,
                "paged_kv_cached_tokens": 64000,
                "paged_kv_prefix_pages": 4000,
                "paged_kv_prefix_page_hits": 6272000,
                "paged_kv_prefix_page_lookups": 6272000,
                "paged_kv_prefix_token_hits": 100352000,
                "paged_kv_prefix_token_lookups": 100352000,
                "profile_vllm_cache_write_total_ms": 100.0,
            },
        },
    ]

    summary = tool.summarize_results(results)

    assert len(summary) == 1
    assert summary[0]["generation_speedup_dense_over_paged"] == 2.0
    assert summary[0]["paged_attention_hit_rate"] == 1.0
    assert summary[0]["paged_kv_prefix_page_hit_rate"] == 1.0
    assert summary[0]["paged_kv_prefix_page_hits"] == 6272000
    assert summary[0]["paged_attention_actual_calls"] == 1568
    assert summary[0]["paged_attention_reuse_coverage"] == 1.0
    assert summary[0]["profile_vllm_cache_write_total_ms"] == 100.0


def test_resolve_reference_images_uses_user_paths(tmp_path):
    tool = _load_benchmark_tool()
    refs = []
    for idx in range(3):
        path = tmp_path / f"ref_{idx}.png"
        path.write_bytes(b"not decoded by the benchmark wrapper")
        refs.append(path)
    args = _args(reference_images=",".join(str(path) for path in refs))
    scenarios = [tool.scenario_catalog(5.0)["img2img-3ref-nocfg"]]

    resolved = tool.resolve_reference_images(args, tmp_path, scenarios)

    assert resolved == refs
