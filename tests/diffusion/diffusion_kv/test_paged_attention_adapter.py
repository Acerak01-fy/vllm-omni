# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.diffusion_kv import paged_attention_adapter as adapter_module
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
    DiffusionPagedAttentionRow,
    DiffusionPagedAttentionRowBinding,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    override_paged_kv_adapter,
    set_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeBlockTables:
    def __init__(self) -> None:
        self.max_num_reqs = 4
        self.max_num_batched_tokens = 16
        self.cp_size = 1
        self.cp_rank = 0
        self.cp_interleave = 1
        self.block_sizes = [4]
        self.blocks_per_kv_block = [1]
        self.num_blocks = SimpleNamespace(np=np.full((1, self.max_num_reqs), 4, dtype=np.int32))
        self.gather_calls: list[tuple[torch.Tensor, int]] = []
        self.slot_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        self.gather_calls.append((idx_mapping.clone(), num_reqs_padded))
        return (torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        self.slot_calls.append(
            (
                idx_mapping.clone(),
                query_start_loc.clone(),
                positions.clone(),
                num_tokens_padded,
            )
        )
        return positions[:num_tokens_padded].to(torch.int64).unsqueeze(0)


class _FakeSpec:
    non_causal: bool
    block_size = 4

    def __init__(self, *, non_causal: bool) -> None:
        self.non_causal = non_causal

    def max_num_blocks_per_req(self, _vllm_config, max_len: int) -> int:
        return math.ceil(max_len / self.block_size)


class _FakeAttentionGroup:
    def __init__(self, reorder_batch_threshold: int | None) -> None:
        self.builder = SimpleNamespace(reorder_batch_threshold=reorder_batch_threshold)

    def get_metadata_builder(self, _index: int):
        return self.builder


class _FakeLayer:
    def __init__(self, *, non_causal: bool) -> None:
        self.num_heads = 2
        self.num_kv_heads = 2
        self.head_size = 4
        self.head_size_v = 4
        self.spec = _FakeSpec(non_causal=non_causal)
        self.updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]] = []
        self.native_events: list[str] = []

    def do_kv_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.native_events.append("update")
        self.updates.append((key, value, slot_mapping))

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: object,
    ) -> torch.Tensor:
        self.native_events.append("forward")
        self.calls.append((query, key, value, metadata))
        return query.reshape(query.shape[0], -1)


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    non_causal: bool = True,
    reorder_batch_threshold: int | None = None,
) -> tuple[DiffusionPagedAttentionAdapter, _FakeBlockTables, _FakeLayer, list[tuple]]:
    events: list[tuple] = []
    block_tables = _FakeBlockTables()
    layer = _FakeLayer(non_causal=non_causal)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer-0"], kv_cache_spec=layer.spec)],
    )
    row_map = {
        ("req-0", 0, None): DiffusionPagedAttentionRowBinding(2, 16),
        ("req-1", 1, None): DiffusionPagedAttentionRowBinding(3, 16),
        ("req-0", None, "text"): DiffusionPagedAttentionRowBinding(1, 16),
    }

    monkeypatch.setattr(adapter_module, "set_current_vllm_config", lambda _config: nullcontext())

    def build_metadata(**kwargs):
        events.append(("build", kwargs))
        return {"layer-0": "native-metadata"}

    monkeypatch.setattr(adapter_module, "build_attn_metadata", build_metadata)
    monkeypatch.setattr(
        adapter_module,
        "build_slot_mappings_by_layer",
        lambda slot_mappings, _config: {"layer-0": slot_mappings[0]},
    )

    config = SimpleNamespace(
        name="vllm-config",
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    adapter = DiffusionPagedAttentionAdapter(
        vllm_config=config,
        device=torch.device("cpu"),
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        attn_groups=[[_FakeAttentionGroup(reorder_batch_threshold)]],
        layers={"layer-0": layer},
        resolve_row=lambda request_id, sequence_id, context_id: row_map[(request_id, sequence_id, context_id)],
    )
    return adapter, block_tables, layer, events


def test_prepare_batch_reuses_native_block_table_and_metadata_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, block_tables, _, events = _make_adapter(monkeypatch)
    rows = (
        DiffusionPagedAttentionRow(
            request_id="req-0",
            sequence_id=0,
            query_len=3,
            seq_len=8,
            kv_start_pos=4,
        ),
        DiffusionPagedAttentionRow(
            request_id="req-1",
            sequence_id=1,
            query_len=2,
            seq_len=2,
        ),
    )

    batch = adapter.prepare_batch(rows)

    assert batch.num_tokens == 5
    assert batch.row_indices.tolist() == [2, 3]
    assert batch.query_start_loc.tolist() == [0, 3, 5]
    assert batch.seq_lens.tolist() == [8, 2]
    assert batch.positions.tolist() == [4, 5, 6, 0, 1]
    assert block_tables.gather_calls[0][1] == 2
    assert block_tables.gather_calls[0][0].tolist() == [2, 3]
    assert block_tables.slot_calls[0][3] == 5
    build_kwargs = events[0][1]
    assert build_kwargs["causal"] is False
    assert build_kwargs["max_query_len"] == 3
    assert build_kwargs["max_seq_len"] == 8
    assert build_kwargs["positions"] is batch.positions
    assert batch.attn_metadata == {"layer-0": "native-metadata"}
    assert batch.slot_mappings_by_layer["layer-0"].tolist() == [4, 5, 6, 0, 1]


def test_active_batch_runs_native_layer_and_restores_diffusion_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=5,
                seq_len=5,
            )
        ]
    )
    query = torch.randn(1, 5, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with adapter.activate(batch):
        output = adapter.forward("layer-0", query, key, value)

    assert output.shape == query.shape
    assert torch.equal(output, query)
    assert layer.calls[0][0].shape == (5, 2, 4)
    assert layer.native_events == ["update", "forward"]
    assert layer.calls[0][3] == "native-metadata"


def test_forward_extracts_non_aligned_suffix_from_full_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=5,
                kv_start_pos=3,
            )
        ]
    )
    query = torch.randn(1, 2, 2, 4)
    key = torch.arange(1 * 5 * 2 * 4, dtype=torch.float32).reshape(1, 5, 2, 4)
    value = key + 100

    with adapter.activate(batch):
        output = adapter.forward("layer-0", query, key, value)

    key_write, value_write, slot_mapping = layer.updates[0]
    assert output.shape == query.shape
    assert torch.equal(key_write, key[:, 3:5].reshape(2, 2, 4))
    assert torch.equal(value_write, value[:, 3:5].reshape(2, 2, 4))
    assert slot_mapping.tolist() == [3, 4]
    assert layer.native_events == ["update", "forward"]


def test_forward_rejects_untranslated_omni_metadata_before_cache_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    qkv = torch.randn(1, 2, 4)
    metadata = SimpleNamespace(attn_mask=torch.ones(1, 1), extra={})

    with adapter.activate(batch), pytest.raises(NotImplementedError, match="cannot translate"):
        adapter.forward("layer-0", qkv, qkv, qkv, omni_attn_metadata=metadata)

    assert layer.updates == []


def test_activate_exposes_adapter_only_for_active_omni_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    with set_forward_context():
        assert get_forward_context().paged_kv_adapter is None
        with adapter.activate(batch):
            assert get_forward_context().paged_kv_adapter is adapter
        assert get_forward_context().paged_kv_adapter is None


def test_forward_accepts_flattened_hidden_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    query = torch.randn(3, 8)

    with adapter.activate(batch):
        output = adapter.forward("layer-0", query, query, query)

    assert output.shape == query.shape
    assert torch.equal(output, query)


def test_forward_requires_active_prepared_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    qkv = torch.randn(1, 2, 4)

    with pytest.raises(RuntimeError, match="adapter.activate"):
        adapter.forward("layer-0", qkv, qkv, qkv)


def test_new_preparation_invalidates_native_buffer_views_in_older_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    old_batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    adapter.prepare_batch([DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1)])

    with pytest.raises(ValueError, match="stale"):
        with adapter.activate(old_batch):
            pass


def test_block_table_change_invalidates_prepared_native_buffer_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    adapter.invalidate_prepared_batches()

    with pytest.raises(ValueError, match="stale"):
        with adapter.activate(batch):
            pass


def test_block_table_change_is_rejected_during_active_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    with adapter.activate(batch), pytest.raises(RuntimeError, match="during an active forward"):
        adapter.invalidate_prepared_batches()


def test_prepare_batch_rejects_active_native_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    with adapter.activate(batch), pytest.raises(RuntimeError, match="during an active forward"):
        adapter.prepare_batch([DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1)])


def test_forward_rejects_tensor_count_different_from_prepared_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    qkv = torch.randn(2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="token count"):
        adapter.forward("layer-0", qkv, qkv, qkv)


def test_forward_rejects_dtype_different_from_model_activation_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    qkv = torch.randn(1, 2, 4, dtype=torch.float64)

    with adapter.activate(batch), pytest.raises(ValueError, match="model activation dtype"):
        adapter.forward("layer-0", qkv, qkv, qkv)


def test_forward_rejects_invalid_key_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    query = torch.randn(4, 2, 4)
    key = torch.randn(3, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="must contain either"):
        adapter.forward("layer-0", query, key, query)


def test_forward_rejects_batched_layout_that_does_not_match_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1),
            DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=3, seq_len=3),
        ]
    )
    qkv = torch.randn(2, 2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="must match prepared rows"):
        adapter.forward("layer-0", qkv, qkv, qkv)


def test_forward_rejects_uniform_full_kv_for_nonuniform_row_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=1,
                seq_len=3,
                kv_start_pos=2,
            ),
            DiffusionPagedAttentionRow(
                request_id="req-1",
                sequence_id=1,
                query_len=1,
                seq_len=5,
                kv_start_pos=4,
            ),
        ]
    )
    query = torch.randn(2, 1, 2, 4)
    full_kv = torch.randn(2, 4, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="full-sequence layout does not match"):
        adapter.forward("layer-0", query, full_kv, full_kv)


def test_causal_batch_rejects_non_suffix_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch, non_causal=False)
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=2,
        seq_len=8,
        kv_start_pos=3,
    )

    batch = adapter.prepare_batch([row])
    qkv = torch.randn(2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="requires each query/write span to end"):
        adapter.forward("layer-0", qkv, qkv, qkv)


def test_prepare_batch_rejects_native_decode_after_prefill_order(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, block_tables, _, _ = _make_adapter(
        monkeypatch,
        non_causal=False,
        reorder_batch_threshold=1,
    )
    rows = [
        DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=2, seq_len=2),
        DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1),
    ]

    with pytest.raises(ValueError, match="decode/short-query rows before"):
        adapter.prepare_batch(rows)

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_rejects_seq_len_beyond_installed_row_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, _ = _make_adapter(monkeypatch)
    block_tables.num_blocks.np[0, 2] = 1
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=5,
        seq_len=5,
    )

    with pytest.raises(ValueError, match="installed Worker row contains only 1 blocks"):
        adapter.prepare_batch([row])

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_rejects_seq_len_beyond_installed_logical_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, _ = _make_adapter(monkeypatch)
    adapter.resolve_row = lambda *_args: DiffusionPagedAttentionRowBinding(2, 4)
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=5,
        seq_len=5,
    )

    with pytest.raises(ValueError, match="installed allocation has logical length 4"):
        adapter.prepare_batch([row])

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_passes_dcp_local_seq_lens_to_native_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, events = _make_adapter(monkeypatch)
    block_tables.cp_size = 2
    block_tables.cp_rank = 1
    block_tables.cp_interleave = 2
    block_tables.num_blocks.np[0, 2] = 1

    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=5, seq_len=5)]
    )

    assert batch.row_indices.tolist() == [2]
    assert events[0][1]["dcp_local_seq_lens"].tolist() == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_id": 0, "context_id": "text"}, "exactly one"),
        ({}, "exactly one"),
        ({"sequence_id": 0, "query_len": 0}, "query_len"),
        ({"sequence_id": 0, "kv_start_pos": 4, "query_len": 2, "seq_len": 5}, "exceeds seq_len"),
    ],
)
def test_row_contract_rejects_invalid_identity_or_span(kwargs: dict, message: str) -> None:
    values = {
        "request_id": "req-0",
        "query_len": 1,
        "seq_len": 1,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        DiffusionPagedAttentionRow(**values)


def test_layer_adapter_forces_flash_attention_and_uses_rank_local_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_backends = []
    in_config_context = False

    class FakeImpl:
        def __init__(self, **kwargs) -> None:
            assert in_config_context
            self.kwargs = kwargs
            self.vllm_flash_attn_version = 3

    class FakeFlashAttentionBackend:
        @staticmethod
        def get_name() -> str:
            return "FLASH_ATTN"

        @staticmethod
        def indexes_kv_by_block_stride() -> bool:
            return True

        @staticmethod
        def get_impl_cls():
            return FakeImpl

    original_backend_per_kind = {"full": object()}
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None, backend_per_kind=original_backend_per_kind),
        model_config=SimpleNamespace(dtype=torch.float16),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )

    def select_backend(**_kwargs):
        assert in_config_context
        selected_backends.append((config.attention_config.backend, config.attention_config.backend_per_kind))
        return FakeFlashAttentionBackend

    @contextmanager
    def current_config(_config):
        nonlocal in_config_context
        previous = in_config_context
        in_config_context = True
        try:
            yield
        finally:
            in_config_context = previous

    monkeypatch.setattr(adapter_module, "get_attn_backend", select_backend)
    monkeypatch.setattr(adapter_module, "set_current_vllm_config", current_config)
    layer = SimpleNamespace(num_heads=8, softmax_scale=0.125)
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float16,
        non_causal=True,
    )

    native_layer = adapter_module.DiffusionPagedAttentionLayerAdapter(
        layer_name="layer-0",
        layer=layer,
        spec=spec,
        vllm_config=config,
        device=torch.device("cpu"),
        ulysses_degree=2,
    )

    assert selected_backends == [(adapter_module.AttentionBackendEnum.FLASH_ATTN, {})]
    assert config.attention_config.backend is None
    assert config.attention_config.backend_per_kind is original_backend_per_kind
    assert native_layer.num_heads == 4
    assert native_layer.num_kv_heads == 2
    assert native_layer.spec.num_kv_heads == 2
    assert native_layer.spec.indexes_kv_by_block_stride is True
    assert native_layer.impl.kwargs["num_heads"] == 4
    assert native_layer.impl.kwargs["num_kv_heads"] == 2


def test_layer_adapter_rejects_invalid_gqa_geometry() -> None:
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None, backend_per_kind={}),
        model_config=SimpleNamespace(dtype=torch.float16),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    layer = SimpleNamespace(num_heads=8, softmax_scale=0.125)
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=8,
        dtype=torch.float16,
        non_causal=True,
    )

    with pytest.raises(ValueError, match="num_heads divisible by num_kv_heads"):
        adapter_module.DiffusionPagedAttentionLayerAdapter(
            layer_name="layer-0",
            layer=layer,
            spec=spec,
            vllm_config=config,
            device=torch.device("cpu"),
        )


def test_omni_attention_wraps_paged_kernel_with_sp_hooks() -> None:
    events: list[str] = []

    class Strategy:
        name = "ulysses"

        def pre_attention(self, query, key, value, metadata):
            events.append("pre")
            return query + 1, key + 2, value + 3, metadata, object()

        def post_attention(self, output, _ctx):
            events.append("post")
            return output + 4

    class Adapter:
        def forward(self, layer_name, query, key, value, *, omni_attn_metadata):
            events.append("kernel")
            assert layer_name == "layer-0"
            assert omni_attn_metadata is None
            assert torch.equal(key, original + 2)
            assert torch.equal(value, original + 3)
            return query * 2

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.prefix = "layer-0"
    layer.paged_kv_cache_role = "primary"
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: Strategy()
    layer._with_kv_cache_dtype = lambda metadata: metadata
    original = torch.zeros(1, 2, 2, 4)

    with set_forward_context(), override_paged_kv_adapter(Adapter()):
        output = layer._forward_impl(original, original, original)

    assert events == ["pre", "kernel", "post"]
    assert torch.equal(output, torch.full_like(original, 6))


def test_omni_attention_keeps_dense_kernel_without_active_adapter() -> None:
    events: list[str] = []

    class Strategy:
        name = "ulysses"

        def pre_attention(self, query, key, value, metadata):
            events.append("pre")
            return query, key, value, metadata, object()

        def post_attention(self, output, _ctx):
            events.append("post")
            return output

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.paged_kv_cache_role = "primary"
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: Strategy()
    layer._with_kv_cache_dtype = lambda metadata: metadata
    layer._run_local_attention = lambda query, _key, _value, _metadata: events.append("dense") or query
    qkv = torch.zeros(1, 2, 2, 4)

    output = layer._forward_impl(qkv, qkv, qkv)

    assert events == ["pre", "dense", "post"]
    assert output is qkv
