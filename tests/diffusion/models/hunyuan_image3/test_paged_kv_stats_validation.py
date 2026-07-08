# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from pathlib import Path

import pytest


def _load_validation_funcs():
    script = Path("examples/offline_inference/hunyuan_image3/end2end.py")
    module_ast = ast.parse(script.read_text())
    selected = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_validate_paged_kv_layer_stats", "_validate_paged_kv_stats"}
    ]
    namespace = {"json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_validate_paged_kv_stats"]


def _good_stats():
    return {
        "paged_kv_cache_enabled": True,
        "enabled_layers": 2,
        "paged_cache_builds": 2,
        "paged_attention_calls": 2,
        "paged_attention_custom_mask_calls": 2,
        "paged_attention_errors": 0,
        "layers_detail": [
            {
                "layer_idx": 0,
                "paged_kv_cache_enabled": True,
                "paged_cache_builds": 1,
                "paged_attention_calls": 1,
                "paged_attention_custom_mask_calls": 1,
                "paged_attention_errors": 0,
            },
            {
                "layer_idx": 1,
                "paged_kv_cache_enabled": True,
                "paged_cache_builds": 1,
                "paged_attention_calls": 1,
                "paged_attention_custom_mask_calls": 1,
                "paged_attention_errors": 0,
            },
        ],
    }


def test_validate_paged_kv_stats_accepts_custom_mask_layer_details():
    validate = _load_validation_funcs()

    validate(_good_stats(), require_custom_mask=True)


def test_validate_paged_kv_stats_rejects_missing_custom_mask_call():
    validate = _load_validation_funcs()
    stats = _good_stats()
    stats["paged_attention_custom_mask_calls"] = 0

    with pytest.raises(RuntimeError, match="custom-mask attention"):
        validate(stats, require_custom_mask=True)


def test_validate_paged_kv_stats_rejects_layer_without_custom_mask_call():
    validate = _load_validation_funcs()
    stats = _good_stats()
    stats["layers_detail"][1]["paged_attention_custom_mask_calls"] = 0

    with pytest.raises(RuntimeError, match="layer 1.*custom-mask"):
        validate(stats, require_custom_mask=True)
