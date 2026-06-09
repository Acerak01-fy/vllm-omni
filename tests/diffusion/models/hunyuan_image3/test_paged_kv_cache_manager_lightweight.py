# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight tests for Hunyuan Image3 paged KV cache metadata.

The full Hunyuan transformer module imports the active vLLM runtime. Extract
only the paged-KV dataclasses and manager so these metadata contracts can be
tested even when the local vLLM wheel is not import-compatible with the repo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRANSFORMER = _REPO_ROOT / "vllm_omni" / "diffusion" / "models" / "hunyuan_image3" / "hunyuan_image3_transformer.py"


def _load_paged_kv_manager():
    wanted_names = {
        "_ceil_div",
        "HunyuanImage3PagedKVAttentionMetadata",
        "_PagedPromptKVState",
        "HunyuanImage3PagedKVCacheManager",
    }
    module = ast.parse(_TRANSFORMER.read_text())
    body = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_names:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in wanted_names:
            body.append(node)
    namespace = {
        "dataclass": dataclass,
        "torch": torch,
        "should_profile_hunyuan_image3_paged_kv": lambda: False,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_TRANSFORMER), "exec"), namespace)
    return namespace["HunyuanImage3PagedKVCacheManager"]


def _load_paged_kv_runner_contract():
    wanted_names = {
        "HunyuanImage3PagedKVAttentionMetadata",
        "HunyuanImage3FlashInferPagedKVRunner",
    }
    module = ast.parse(_TRANSFORMER.read_text())
    body = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted_names:
            body.append(node)
    namespace = {
        "Any": object,
        "dataclass": dataclass,
        "torch": torch,
        "_HY3_PAGED_KV_DEFAULT_WORKSPACE_BYTES": 1,
        "_HY3_PAGED_KV_WORKSPACE_BYTES_ENV": "VLLM_OMNI_HY3_PAGED_KV_CACHE_WORKSPACE_BYTES",
        "_parse_positive_int_env": lambda _env_name, default: default,
        "should_profile_hunyuan_image3_paged_kv": lambda: False,
        "should_validate_hunyuan_image3_paged_kv_run_inputs": lambda: True,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_TRANSFORMER), "exec"), namespace)
    return namespace["HunyuanImage3FlashInferPagedKVRunner"]


def _load_position_validator_harness():
    module = ast.parse(_TRANSFORMER.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ImageKVCacheManager"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_current_position_ids"
    )
    harness = ast.ClassDef(
        name="_PositionValidatorHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    ast.fix_missing_locations(harness)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[harness], type_ignores=[]), str(_TRANSFORMER), "exec"), namespace)
    return namespace["_PositionValidatorHarness"]


def _load_paged_run_harness_base():
    module = ast.parse(_TRANSFORMER.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ImageKVCacheManager"
    )
    wanted_methods = {"_paged_kv_fallback", "_run_paged_prompt_kv_attention"}
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted_methods
    ]
    harness = ast.ClassDef(
        name="_PagedRunHarnessBase",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    ast.fix_missing_locations(harness)

    class _DummyLogger:
        @staticmethod
        def debug(*args, **kwargs) -> None:
            pass

    namespace = {
        "Any": object,
        "HunyuanImage3PagedKVAttentionMetadata": object,
        "logger": _DummyLogger(),
        "torch": torch,
    }
    exec(compile(ast.Module(body=[harness], type_ignores=[]), str(_TRANSFORMER), "exec"), namespace)
    return namespace["_PagedRunHarnessBase"]


def _known_kv(batch_size: int, seq_len: int, num_kv_heads: int = 2, head_dim: int = 4):
    values = torch.arange(batch_size * seq_len * num_kv_heads * head_dim, dtype=torch.float32)
    key = values.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    value = key + 1000.0
    return key, value


def _simulate_flashinfer_append(metadata, current_key: torch.Tensor, current_value: torch.Tensor) -> None:
    flat_key = current_key.reshape(-1, current_key.shape[2], current_key.shape[3])
    flat_value = current_value.reshape(-1, current_value.shape[2], current_value.shape[3])
    for token_idx, batch_idx in enumerate(metadata.append_batch_indices.tolist()):
        position = int(metadata.append_positions[token_idx].item())
        page_table_offset = position // metadata.page_size
        slot = position % metadata.page_size
        page_table_start = int(metadata.kv_indptr[batch_idx].item())
        page_idx = int(metadata.kv_indices[page_table_start + page_table_offset].item())
        metadata.key_cache[page_idx, slot] = flat_key[token_idx]
        metadata.value_cache[page_idx, slot] = flat_value[token_idx]


def _dense_from_pages(metadata, batch_idx: int, num_kv_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    page_start = int(metadata.kv_indptr[batch_idx].item())
    page_end = int(metadata.kv_indptr[batch_idx + 1].item())
    page_indices = metadata.kv_indices[page_start:page_end]
    seq_len = int(metadata.seq_lens[batch_idx].item())
    key = metadata.key_cache[page_indices].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    value = metadata.value_cache[page_indices].reshape(-1, num_kv_heads, head_dim)[:seq_len]
    return key, value


def _runner_contract_inputs(attention_mask: torch.Tensor | None = None):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    cached_key, cached_value = _known_kv(batch_size=2, seq_len=5)
    state = manager.build_prompt_state(cached_key, cached_value, torch.tensor([3, 5], dtype=torch.long))
    assert state is not None

    current_key, current_value = _known_kv(batch_size=2, seq_len=3)
    metadata = manager.build_attention_metadata(current_key, seq_len=8, attention_mask=attention_mask)
    query = torch.zeros(2, 3, 4, 4)
    return _load_paged_kv_runner_contract(), query, current_key, current_value, metadata


class _FallbackStats:
    def __init__(self) -> None:
        self.fallbacks = 0
        self.runner_errors = 0
        self.calls = 0

    def record_fallback(self) -> None:
        self.fallbacks += 1

    def record_runner_error(self) -> None:
        self.runner_errors += 1

    def record_attention_call(self, *, custom_mask_used: bool) -> None:
        del custom_mask_used
        self.calls += 1


class _NeverCalledRunner:
    def run(self, *args, **kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("runner must not run when metadata build fails")


def _make_metadata_failure_harness(*, required: bool):
    base_cls = _load_paged_run_harness_base()

    class Harness(base_cls):
        def __init__(self) -> None:
            self._paged_kv_cache_required = required
            self._paged_kv_cache_manager = _FallbackStats()

        def _can_use_paged_prompt_kv_attention(self, *args, **kwargs):
            return True, ""

        def _validate_current_position_ids(self, *args, **kwargs) -> None:
            return None

        def _build_paged_prompt_kv_attention_metadata(self, *args, **kwargs):
            raise ValueError("bad metadata")

        def _get_paged_kv_runner(self):
            return _NeverCalledRunner()

    return Harness()


def test_build_prompt_state_packs_prefix_pages_and_stats():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    cached_lens = torch.tensor([3, 5], dtype=torch.long)

    state = manager.build_prompt_state(key, value, cached_lens)

    assert state is not None
    assert state.key_cache.shape == (3, 4, 2, 4)
    assert state.value_cache.shape == (3, 4, 2, 4)
    assert state.prefix_page_indptr.tolist() == [0, 1, 3]
    assert state.prefix_page_indices.tolist() == [0, 1, 2]
    assert state.cached_lens.tolist() == [3, 5]
    assert manager.get_stats()["paged_cache_builds"] == 1
    assert torch.equal(state.key_cache[0, :3], key[0, :3])
    assert torch.equal(state.key_cache[1].reshape(-1, 2, 4), key[1, :4])
    assert torch.equal(state.key_cache[2, :1], key[1, 4:5])


def test_position_validator_accepts_none_and_valid_positions():
    harness = _load_position_validator_harness()()
    harness.image_kv_cache_lens = torch.tensor([3, 5], dtype=torch.long)

    harness._validate_current_position_ids(None, bs=2, q_len=3)
    harness._validate_current_position_ids(
        torch.tensor([[3, 4, 5], [5, 6, 7]], dtype=torch.long),
        bs=2,
        q_len=3,
    )


@pytest.mark.parametrize(
    ("cached_lens", "position_ids", "message"),
    [
        (None, torch.tensor([[3, 4, 5], [5, 6, 7]], dtype=torch.long), "requires cached prompt lengths"),
        (torch.tensor([3, 5], dtype=torch.long), torch.tensor([[3, 4, 5]], dtype=torch.long), "must equal"),
        (torch.tensor([3, 5], dtype=torch.long), torch.tensor([[4, 5, 6], [5, 6, 7]], dtype=torch.long), "first current position"),
    ],
)
def test_position_validator_rejects_inconsistent_positions(cached_lens, position_ids, message):
    harness = _load_position_validator_harness()()
    harness.image_kv_cache_lens = cached_lens

    with pytest.raises(ValueError, match=message):
        harness._validate_current_position_ids(position_ids, bs=2, q_len=3)


def test_constructor_rejects_invalid_page_size():
    manager_cls = _load_paged_kv_manager()

    with pytest.raises(ValueError, match="page_size must be positive"):
        manager_cls(enabled=True, required=False, page_size=0)


@pytest.mark.parametrize(
    "cached_lens",
    [
        torch.tensor([3], dtype=torch.long),
        torch.tensor([3, 6], dtype=torch.long),
        torch.tensor([3, 0], dtype=torch.long),
    ],
)
def test_build_prompt_state_rejects_invalid_cached_lens(cached_lens):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=False, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)

    state = manager.build_prompt_state(key, value, cached_lens)

    assert state is None
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


def test_build_prompt_state_rejects_mismatched_key_value_shapes():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=False, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)

    state = manager.build_prompt_state(key, value[:, :4], torch.tensor([3, 4], dtype=torch.long))

    assert state is None
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


@pytest.mark.parametrize(
    ("cached_key", "cached_value", "cached_lens", "message"),
    [
        (torch.zeros(2, 5, 2), torch.zeros(2, 5, 2), torch.tensor([3, 4]), "must be 4D"),
        (torch.zeros(2, 5, 2, 4), torch.zeros(2, 5, 2, 4), torch.tensor([[3], [4]]), "must be 1D"),
    ],
)
def test_build_prompt_state_rejects_invalid_prompt_cache_shapes(cached_key, cached_value, cached_lens, message):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=False, page_size=4)

    state = manager.build_prompt_state(cached_key, cached_value, cached_lens)

    assert state is None
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


@pytest.mark.parametrize(
    ("cached_lens", "message"),
    [
        (torch.tensor([3], dtype=torch.long), "lens count"),
        (torch.tensor([3, 6], dtype=torch.long), "must be <= cached KV length"),
        (torch.tensor([3, 0], dtype=torch.long), "must be positive"),
    ],
)
def test_build_prompt_state_required_mode_raises_on_invalid_cached_lens(cached_lens, message):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)

    with pytest.raises(RuntimeError, match=message):
        manager.build_prompt_state(key, value, cached_lens)
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


def test_build_prompt_state_required_mode_raises_on_mismatched_key_value_shapes():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)

    with pytest.raises(RuntimeError, match="cached key shape"):
        manager.build_prompt_state(key, value[:, :4], torch.tensor([3, 4], dtype=torch.long))
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


@pytest.mark.parametrize(
    ("cached_key", "cached_value", "cached_lens", "message"),
    [
        (torch.zeros(2, 5, 2), torch.zeros(2, 5, 2), torch.tensor([3, 4]), "must be 4D"),
        (torch.zeros(2, 5, 2, 4), torch.zeros(2, 5, 2, 4), torch.tensor([[3], [4]]), "must be 1D"),
    ],
)
def test_build_prompt_state_required_mode_raises_on_invalid_prompt_cache_shapes(
    cached_key,
    cached_value,
    cached_lens,
    message,
):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)

    with pytest.raises(RuntimeError, match=message):
        manager.build_prompt_state(cached_key, cached_value, cached_lens)
    assert manager.get_stats()["paged_kv_cache_active"] is False
    assert manager.get_stats()["paged_cache_build_failures"] == 1


@pytest.mark.parametrize(
    ("current_key", "seq_len", "message"),
    [
        (torch.zeros(2, 3, 2), 8, "current key must be 4D"),
        (torch.zeros(2, 0, 2, 4), 5, "q_len must be positive"),
        (torch.zeros(1, 3, 2, 4), 8, "batch size changed"),
        (torch.zeros(2, 3, 2, 4), 7, "must equal max cached prefix length"),
    ],
)
def test_build_attention_metadata_rejects_inconsistent_inputs(current_key, seq_len, message):
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    state = manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    assert state is not None

    with pytest.raises(ValueError, match=message):
        manager.build_attention_metadata(current_key, seq_len=seq_len, attention_mask=None)


def test_build_attention_metadata_extends_pages_for_current_tokens():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=4)
    state = manager.build_prompt_state(key, value, torch.tensor([4, 4], dtype=torch.long))
    assert state is not None

    current_key = torch.zeros(2, 3, 2, 4)
    metadata = manager.build_attention_metadata(current_key, seq_len=7, attention_mask=None)

    assert metadata.key_cache.shape == (4, 4, 2, 4)
    assert metadata.value_cache.shape == (4, 4, 2, 4)
    assert metadata.qo_indptr.tolist() == [0, 3, 6]
    assert metadata.kv_indptr.tolist() == [0, 2, 4]
    assert metadata.kv_indices.tolist() == [0, 2, 1, 3]
    assert metadata.kv_last_page_len.tolist() == [3, 3]
    assert metadata.append_batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert metadata.append_positions.tolist() == [4, 5, 6, 4, 5, 6]
    assert metadata.cached_lens.tolist() == [4, 4]
    assert metadata.seq_lens.tolist() == [7, 7]
    assert metadata.max_qo_len == 3
    assert metadata.max_kv_len == 7
    assert metadata.prefix_page_count == 2
    assert metadata.prefix_token_count == 8
    assert metadata.page_table_entry_count == 4
    assert metadata.current_page_count == 2
    assert metadata.custom_mask is None
    assert manager.get_stats()["paged_cache_expansions"] == 1
    assert manager.get_stats()["paged_kv_prefix_page_hits"] == 0

    manager.record_attention_call(custom_mask_used=False, metadata=metadata)

    stats = manager.get_stats()
    assert stats["paged_attention_calls"] == 1
    assert stats["paged_kv_prefix_page_hits"] == 2
    assert stats["paged_kv_prefix_page_lookups"] == 2
    assert stats["paged_kv_prefix_page_hit_rate"] == 1.0
    assert stats["paged_kv_prefix_token_hits"] == 8
    assert stats["paged_kv_prefix_token_lookups"] == 8
    assert stats["paged_kv_prefix_token_hit_rate"] == 1.0
    assert stats["paged_kv_page_table_entries"] == 4
    assert stats["paged_kv_current_page_entries"] == 2


def test_build_attention_metadata_supports_non_uniform_cached_lens():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    state = manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    assert state is not None

    current_key = torch.zeros(2, 3, 2, 4)
    metadata = manager.build_attention_metadata(current_key, seq_len=8, attention_mask=None)

    assert metadata.key_cache.shape == (4, 4, 2, 4)
    assert metadata.qo_indptr.tolist() == [0, 3, 6]
    assert metadata.kv_indptr.tolist() == [0, 2, 4]
    assert metadata.kv_indices.tolist() == [0, 3, 1, 2]
    assert metadata.kv_last_page_len.tolist() == [2, 4]
    assert metadata.append_batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert metadata.append_positions.tolist() == [3, 4, 5, 5, 6, 7]
    assert metadata.seq_lens.tolist() == [6, 8]
    assert metadata.max_kv_len == 8
    assert manager.get_stats()["paged_cache_expansions"] == 1


def test_flashinfer_append_layout_preserves_prefix_and_writes_current_pages():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    state = manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    assert state is not None

    current_key, current_value = _known_kv(batch_size=2, seq_len=3)
    current_key = current_key + 10_000.0
    current_value = current_value + 10_000.0
    metadata = manager.build_attention_metadata(current_key, seq_len=8, attention_mask=None)

    _simulate_flashinfer_append(metadata, current_key, current_value)

    assert torch.equal(metadata.key_cache[0, :3], key[0, :3])
    assert torch.equal(metadata.key_cache[0, 3], current_key[0, 0])
    assert torch.equal(metadata.key_cache[3, :2], current_key[0, 1:3])
    assert torch.equal(metadata.key_cache[1], key[1, :4])
    assert torch.equal(metadata.key_cache[2, 0], key[1, 4])
    assert torch.equal(metadata.key_cache[2, 1:4], current_key[1])

    dense0_key, dense0_value = _dense_from_pages(metadata, 0, num_kv_heads=2, head_dim=4)
    dense1_key, dense1_value = _dense_from_pages(metadata, 1, num_kv_heads=2, head_dim=4)
    assert torch.equal(dense0_key, torch.cat([key[0, :3], current_key[0]], dim=0))
    assert torch.equal(dense0_value, torch.cat([value[0, :3], current_value[0]], dim=0))
    assert torch.equal(dense1_key, torch.cat([key[1, :5], current_key[1]], dim=0))
    assert torch.equal(dense1_value, torch.cat([value[1, :5], current_value[1]], dim=0))


def test_later_denoise_append_overwrites_current_pages_and_preserves_prefix():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    state = manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    assert state is not None

    first_current_key, first_current_value = _known_kv(batch_size=2, seq_len=3)
    first_current_key = first_current_key + 10_000.0
    first_current_value = first_current_value + 10_000.0
    first_metadata = manager.build_attention_metadata(first_current_key, seq_len=8, attention_mask=None)
    _simulate_flashinfer_append(first_metadata, first_current_key, first_current_value)

    second_current_key, second_current_value = _known_kv(batch_size=2, seq_len=3)
    second_current_key = second_current_key + 20_000.0
    second_current_value = second_current_value + 20_000.0
    second_metadata = manager.build_attention_metadata(second_current_key, seq_len=8, attention_mask=None)
    _simulate_flashinfer_append(second_metadata, second_current_key, second_current_value)

    assert torch.equal(second_metadata.key_cache[0, :3], key[0, :3])
    assert torch.equal(second_metadata.key_cache[1], key[1, :4])
    assert torch.equal(second_metadata.key_cache[2, 0], key[1, 4])
    assert not torch.any(second_metadata.key_cache == 10_000.0)
    assert not torch.any(second_metadata.value_cache == 11_000.0)

    dense0_key, dense0_value = _dense_from_pages(second_metadata, 0, num_kv_heads=2, head_dim=4)
    dense1_key, dense1_value = _dense_from_pages(second_metadata, 1, num_kv_heads=2, head_dim=4)
    assert torch.equal(dense0_key, torch.cat([key[0, :3], second_current_key[0]], dim=0))
    assert torch.equal(dense0_value, torch.cat([value[0, :3], second_current_value[0]], dim=0))
    assert torch.equal(dense1_key, torch.cat([key[1, :5], second_current_key[1]], dim=0))
    assert torch.equal(dense1_value, torch.cat([value[1, :5], second_current_value[1]], dim=0))


def test_boolean_attention_mask_is_flattened_as_flashinfer_custom_mask():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=4)
    manager.build_prompt_state(key, value, torch.tensor([4, 4], dtype=torch.long))
    current_key = torch.zeros(2, 3, 2, 4)
    mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    mask[1, 0, 2, 6] = False

    metadata = manager.build_attention_metadata(current_key, seq_len=7, attention_mask=mask)

    assert metadata.custom_mask is not None
    assert metadata.custom_mask.shape == (2 * 3 * 7,)
    assert torch.equal(metadata.custom_mask.reshape(2, 3, 7), mask[:, 0])


def test_non_uniform_cached_lens_custom_mask_drops_dense_padding_columns():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    current_key = torch.zeros(2, 3, 2, 4)
    mask = torch.ones(2, 1, 3, 8, dtype=torch.bool)
    mask[0, 0, 1, 4] = False  # Dense padding column for sample 0; must be dropped.
    mask[0, 0, 0, 5] = False  # First current token for sample 0.
    mask[1, 0, 2, 4] = False  # Last prefix token for sample 1.
    mask[1, 0, 0, 7] = False  # Last current token for sample 1.

    metadata = manager.build_attention_metadata(current_key, seq_len=8, attention_mask=mask)

    assert metadata.custom_mask is not None
    assert metadata.custom_mask.shape == (3 * 6 + 3 * 8,)
    sample0 = metadata.custom_mask[: 3 * 6].reshape(3, 6)
    sample1 = metadata.custom_mask[3 * 6 :].reshape(3, 8)
    assert sample0[1, 4]  # Dense padding false at column 4 was not copied.
    assert not sample0[0, 3]  # Dense current column 5 maps after 3 prefix columns.
    assert not sample1[2, 4]
    assert not sample1[0, 7]


def test_non_uniform_cached_lens_padding_only_mask_becomes_all_keep():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=2, seq_len=5)
    manager.build_prompt_state(key, value, torch.tensor([3, 5], dtype=torch.long))
    current_key = torch.zeros(2, 3, 2, 4)
    mask = torch.ones(2, 1, 3, 8, dtype=torch.bool)
    mask[0, 0, :, 3:5] = False  # Dense padding columns for sample 0 only.

    metadata = manager.build_attention_metadata(current_key, seq_len=8, attention_mask=mask)

    assert metadata.custom_mask is None


def test_non_boolean_non_all_keep_mask_is_rejected():
    manager_cls = _load_paged_kv_manager()
    manager = manager_cls(enabled=True, required=True, page_size=4)
    key, value = _known_kv(batch_size=1, seq_len=4)
    manager.build_prompt_state(key, value, torch.tensor([4], dtype=torch.long))
    current_key = torch.zeros(1, 3, 2, 4)
    mask = torch.zeros(1, 1, 3, 7)
    mask[..., 0] = float("-inf")

    with pytest.raises(ValueError, match="only supports boolean custom masks"):
        manager.build_attention_metadata(current_key, seq_len=7, attention_mask=mask)


def test_run_paged_attention_falls_back_when_metadata_build_fails_in_optional_mode():
    harness = _make_metadata_failure_harness(required=False)
    query = torch.zeros(1, 3, 4, 4)
    key = torch.zeros(1, 3, 2, 4)
    value = torch.zeros(1, 3, 2, 4)

    result = harness._run_paged_prompt_kv_attention(
        query,
        key,
        value,
        seq_len=7,
        bs=1,
        attention_mask=None,
        full_attn_spans=None,
    )

    assert result is None
    assert harness._paged_kv_cache_manager.fallbacks == 1
    assert harness._paged_kv_cache_manager.calls == 0
    assert harness._paged_kv_cache_manager.runner_errors == 0


def test_run_paged_attention_required_mode_raises_when_metadata_build_fails():
    harness = _make_metadata_failure_harness(required=True)
    query = torch.zeros(1, 3, 4, 4)
    key = torch.zeros(1, 3, 2, 4)
    value = torch.zeros(1, 3, 2, 4)

    with pytest.raises(RuntimeError, match="metadata build failed"):
        harness._run_paged_prompt_kv_attention(
            query,
            key,
            value,
            seq_len=7,
            bs=1,
            attention_mask=None,
            full_attn_spans=None,
        )

    assert harness._paged_kv_cache_manager.fallbacks == 1
    assert harness._paged_kv_cache_manager.calls == 0
    assert harness._paged_kv_cache_manager.runner_errors == 0


def test_runner_contract_accepts_manager_metadata():
    runner_cls, query, key, value, metadata = _runner_contract_inputs()

    runner_cls._validate_run_inputs(query, key, value, metadata)


def test_runner_contract_rejects_qkv_shape_mismatches():
    runner_cls, query, key, value, metadata = _runner_contract_inputs()

    with pytest.raises(ValueError, match="query must be 4D"):
        runner_cls._validate_run_inputs(query.reshape(2, 3, 16), key, value, metadata)
    with pytest.raises(ValueError, match="key shape"):
        runner_cls._validate_run_inputs(query, key[:, :2], value[:, :2], metadata)
    with pytest.raises(ValueError, match="value shape"):
        runner_cls._validate_run_inputs(query, key, value[:, :, :1], metadata)
    with pytest.raises(ValueError, match="head_dim"):
        runner_cls._validate_run_inputs(torch.zeros(2, 3, 4, 5), key, value, metadata)


def test_runner_contract_rejects_cache_shape_mismatches():
    runner_cls, query, key, value, metadata = _runner_contract_inputs()

    bad_cache = metadata.key_cache[:, :, :, :3]
    bad_metadata = replace(metadata, key_cache=bad_cache, value_cache=bad_cache.clone())

    with pytest.raises(ValueError, match="cache KV shape"):
        runner_cls._validate_run_inputs(query, key, value, bad_metadata)


def test_runner_contract_rejects_invalid_index_metadata():
    runner_cls, query, key, value, metadata = _runner_contract_inputs()

    with pytest.raises(ValueError, match="qo_indptr length"):
        runner_cls._validate_run_inputs(query, key, value, replace(metadata, qo_indptr=metadata.qo_indptr[:-1]))

    bad_kv_indptr = metadata.kv_indptr.clone()
    bad_kv_indptr[-1] += 1
    with pytest.raises(ValueError, match="kv_indptr last value"):
        runner_cls._validate_run_inputs(query, key, value, replace(metadata, kv_indptr=bad_kv_indptr))

    bad_kv_indices = metadata.kv_indices.clone()
    bad_kv_indices[0] = metadata.key_cache.shape[0]
    with pytest.raises(ValueError, match="kv_indices"):
        runner_cls._validate_run_inputs(query, key, value, replace(metadata, kv_indices=bad_kv_indices))

    with pytest.raises(ValueError, match="append_positions length"):
        runner_cls._validate_run_inputs(
            query,
            key,
            value,
            replace(metadata, append_positions=metadata.append_positions[:-1]),
        )


def test_runner_contract_rejects_invalid_append_layout():
    runner_cls, query, key, value, metadata = _runner_contract_inputs()

    bad_batch_indices = metadata.append_batch_indices.clone()
    bad_batch_indices[0] = 1
    with pytest.raises(ValueError, match="append_batch_indices"):
        runner_cls._validate_run_inputs(query, key, value, replace(metadata, append_batch_indices=bad_batch_indices))

    bad_positions = metadata.append_positions.clone()
    bad_positions[0] += 1
    with pytest.raises(ValueError, match="append_positions"):
        runner_cls._validate_run_inputs(query, key, value, replace(metadata, append_positions=bad_positions))


def test_runner_contract_validates_custom_mask():
    mask = torch.ones(2, 1, 3, 8, dtype=torch.bool)
    mask[0, 0, 0, 5] = False
    runner_cls, query, key, value, metadata = _runner_contract_inputs(mask)
    assert metadata.custom_mask is not None

    runner_cls._validate_run_inputs(query, key, value, metadata)

    with pytest.raises(ValueError, match="custom_mask length"):
        runner_cls._validate_run_inputs(
            query,
            key,
            value,
            replace(metadata, custom_mask=metadata.custom_mask[:-1]),
        )
    with pytest.raises(ValueError, match="custom_mask must be bool"):
        runner_cls._validate_run_inputs(
            query,
            key,
            value,
            replace(metadata, custom_mask=metadata.custom_mask.to(torch.int32)),
        )
