# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Hunyuan Image3 paged KV environment parsing."""

from __future__ import annotations

import ast
import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRANSFORMER = _REPO_ROOT / "vllm_omni" / "diffusion" / "models" / "hunyuan_image3" / "hunyuan_image3_transformer.py"


class _DummyLogger:
    def warning(self, *args, **kwargs) -> None:
        pass


def _load_env_helpers():
    wanted_names = {
        "_HY3_PAGED_KV_CACHE_ENV",
        "_HY3_PAGED_KV_PAGE_SIZE_ENV",
        "_HY3_PAGED_KV_WORKSPACE_BYTES_ENV",
        "_HY3_PAGED_KV_VALIDATE_RUN_INPUTS_ENV",
        "_HY3_PAGED_KV_DEFAULT_PAGE_SIZE",
        "_HY3_PAGED_KV_DEFAULT_WORKSPACE_BYTES",
        "_ENABLED_VALUES",
        "_DISABLED_VALUES",
        "is_hunyuan_image3_paged_kv_cache_enabled",
        "is_hunyuan_image3_paged_kv_cache_required",
        "_parse_bool_env",
        "_parse_positive_int_env",
        "should_validate_hunyuan_image3_paged_kv_run_inputs",
    }
    module = ast.parse(_TRANSFORMER.read_text())
    body = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted_names:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_names:
            body.append(node)
    namespace = {"os": os, "logger": _DummyLogger()}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_TRANSFORMER), "exec"), namespace)
    return namespace


def test_required_mode_is_enabled_and_required(monkeypatch):
    helpers = _load_env_helpers()
    monkeypatch.setenv(helpers["_HY3_PAGED_KV_CACHE_ENV"], "required")

    assert helpers["is_hunyuan_image3_paged_kv_cache_enabled"]()
    assert helpers["is_hunyuan_image3_paged_kv_cache_required"]()


def test_paged_kv_cache_env_disabled_values(monkeypatch):
    helpers = _load_env_helpers()
    for value in ("0", "false", "no", "off", "disabled", "disable"):
        monkeypatch.setenv(helpers["_HY3_PAGED_KV_CACHE_ENV"], value)
        assert not helpers["is_hunyuan_image3_paged_kv_cache_enabled"]()
        assert not helpers["is_hunyuan_image3_paged_kv_cache_required"]()


def test_paged_kv_cache_env_unknown_value_is_not_enabled(monkeypatch):
    helpers = _load_env_helpers()
    monkeypatch.setenv(helpers["_HY3_PAGED_KV_CACHE_ENV"], "maybe")

    assert not helpers["is_hunyuan_image3_paged_kv_cache_enabled"]()
    assert not helpers["is_hunyuan_image3_paged_kv_cache_required"]()


def test_parse_positive_int_env_uses_default_for_missing_or_invalid_values(monkeypatch):
    helpers = _load_env_helpers()
    env_name = helpers["_HY3_PAGED_KV_PAGE_SIZE_ENV"]
    parse_positive_int_env = helpers["_parse_positive_int_env"]

    monkeypatch.delenv(env_name, raising=False)
    assert parse_positive_int_env(env_name, 16) == 16
    monkeypatch.setenv(env_name, "bad")
    assert parse_positive_int_env(env_name, 16) == 16
    monkeypatch.setenv(env_name, "0")
    assert parse_positive_int_env(env_name, 16) == 16
    monkeypatch.setenv(env_name, "-1")
    assert parse_positive_int_env(env_name, 16) == 16

    monkeypatch.setenv(env_name, "32")
    assert parse_positive_int_env(env_name, 16) == 32


def test_run_input_validation_defaults_to_required_mode(monkeypatch):
    helpers = _load_env_helpers()
    cache_env = helpers["_HY3_PAGED_KV_CACHE_ENV"]
    validate_env = helpers["_HY3_PAGED_KV_VALIDATE_RUN_INPUTS_ENV"]
    should_validate = helpers["should_validate_hunyuan_image3_paged_kv_run_inputs"]

    monkeypatch.delenv(validate_env, raising=False)
    monkeypatch.setenv(cache_env, "required")
    assert should_validate()

    monkeypatch.setenv(cache_env, "on")
    assert not should_validate()

    monkeypatch.setenv(validate_env, "1")
    assert should_validate()

    monkeypatch.setenv(validate_env, "0")
    assert not should_validate()
