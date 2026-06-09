# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Hunyuan Image3 paged KV smoke stats gate.

The offline example imports the full vLLM-Omni runtime at module import time.
Load only the validation function from its AST so this gate can be tested in
minimal environments where the current vLLM package is not import-compatible.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
_END2END_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"


def _load_end2end_function(name: str):
    module = ast.parse(_END2END_SCRIPT.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(_END2END_SCRIPT), "exec"), namespace)
    return namespace[name]


def _load_validate_paged_kv_stats():
    return _load_end2end_function("validate_paged_kv_stats")


def _layer_stats(layer_idx: int, **overrides):
    stats = {
        "layer_idx": layer_idx,
        "paged_kv_cache_enabled": True,
        "paged_kv_cache_required": True,
        "paged_kv_cache_active": True,
        "paged_cache_builds": 1,
        "paged_cache_build_failures": 0,
        "paged_attention_calls": 1,
        "paged_attention_custom_mask_calls": 1,
        "paged_attention_fallbacks": 0,
        "paged_attention_runner_errors": 0,
        "paged_kv_prefix_page_hits": 4,
        "paged_kv_prefix_page_lookups": 4,
        "paged_kv_prefix_token_hits": 64,
        "paged_kv_prefix_token_lookups": 64,
    }
    stats.update(overrides)
    return stats


def _valid_stats():
    layers_detail = [_layer_stats(0), _layer_stats(1)]
    return {
        "layers": len(layers_detail),
        "enabled_layers": len(layers_detail),
        "required_layers": len(layers_detail),
        "active_layers": len(layers_detail),
        "paged_kv_cache_enabled": True,
        "paged_kv_cache_required": True,
        "paged_cache_builds": len(layers_detail),
        "paged_cache_build_failures": 0,
        "paged_attention_calls": len(layers_detail),
        "paged_attention_custom_mask_calls": len(layers_detail),
        "paged_attention_fallbacks": 0,
        "paged_attention_runner_errors": 0,
        "paged_kv_prefix_page_hits": 8,
        "paged_kv_prefix_page_lookups": 8,
        "paged_kv_prefix_token_hits": 128,
        "paged_kv_prefix_token_lookups": 128,
        "layers_detail": layers_detail,
    }


def test_validate_paged_kv_stats_accepts_complete_per_layer_stats():
    validate_paged_kv_stats = _load_validate_paged_kv_stats()

    validate_paged_kv_stats(_valid_stats(), require_custom_mask=True)


def test_enrich_paged_kv_stats_adds_expected_calls_and_page_hit_rate():
    enrich_paged_kv_stats = _load_end2end_function("enrich_paged_kv_stats")
    stats = _valid_stats()
    stats.update(
        {
            "paged_kv_prefix_page_hits": 8,
            "paged_kv_prefix_page_lookups": 8,
            "paged_kv_prefix_token_hits": 40,
            "paged_kv_prefix_token_lookups": 40,
        }
    )

    enriched = enrich_paged_kv_stats(stats, num_inference_steps=4)

    assert enriched is not stats
    assert enriched["paged_attention_expected_calls"] == 6
    assert enriched["paged_attention_reuse_coverage"] == 2 / 6
    assert enriched["paged_kv_prefix_page_hit_rate"] == 1.0
    assert enriched["paged_kv_prefix_token_hit_rate"] == 1.0
    assert enriched["paged_attention_hit_rate"] == 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda stats: stats.pop("layers_detail"), "layer stats were not returned"),
        (lambda stats: stats.update({"required_layers": 1}), "not required on all enabled layers"),
        (lambda stats: stats.update({"paged_cache_build_failures": 1}), "reported build failures"),
        (
            lambda stats: stats["layers_detail"][1].update({"paged_cache_build_failures": 1}),
            "cache build failed on layer 1",
        ),
        (
            lambda stats: stats["layers_detail"][1].update({"paged_attention_calls": 0}),
            "attention did not run on layer 1",
        ),
        (
            lambda stats: stats["layers_detail"][1].update({"paged_attention_fallbacks": 1}),
            "attention fell back on layer 1",
        ),
        (
            lambda stats: stats["layers_detail"][1].update({"paged_attention_custom_mask_calls": 0}),
            "did not use custom masks on layer 1",
        ),
        (
            lambda stats: stats.update({"paged_kv_prefix_page_hits": 0}),
            "did not report prefix page hits",
        ),
        (
            lambda stats: stats["layers_detail"][1].update({"paged_kv_prefix_page_hits": 3}),
            "prefix page misses on layer 1",
        ),
    ],
)
def test_validate_paged_kv_stats_rejects_incomplete_or_fallback_stats(mutation, message):
    validate_paged_kv_stats = _load_validate_paged_kv_stats()
    stats = copy.deepcopy(_valid_stats())
    mutation(stats)

    with pytest.raises(RuntimeError, match=message):
        validate_paged_kv_stats(stats, require_custom_mask=True)
