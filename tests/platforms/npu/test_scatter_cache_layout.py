# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionRow,
    DiffusionPagedAttentionRowBinding,
    DiffusionPagedKVWritePlan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_pa_scatter(monkeypatch):
    from vllm_omni.platforms.npu.platform import (
        _logical_cache_to_pa_nz,
        _pa_nz_to_logical_cache,
    )

    calls = []

    def fake_scatter(*, key, value, key_cache, value_cache, slot_mapping):
        calls.append(slot_mapping.clone())
        head_size = key.shape[-1]
        block_ids = torch.arange(key_cache.shape[0])
        for source, cache_nz in ((key, key_cache), (value, value_cache)):
            logical = _pa_nz_to_logical_cache(cache_nz, head_size).clone()
            block_size = logical.shape[1]
            for token_index, slot in enumerate(slot_mapping.tolist()):
                if slot < 0:
                    continue
                logical[slot // block_size, slot % block_size] = source[token_index]
            cache_nz.copy_(_logical_cache_to_pa_nz(logical, block_ids, head_size))

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_scatter_pa_kv_cache=fake_scatter),
    )
    return calls


def _build_write_plan(rows, bindings, *, cp_size=1, cp_rank=0, cp_interleave=1):
    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    return NPUOmniPlatform.build_diffusion_paged_kv_write_plans(
        rows=rows,
        row_bindings=bindings,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer-0"])]),
        block_tables=SimpleNamespace(
            kernel_block_sizes=[4],
            blocks_per_kv_block=[1],
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_interleave=cp_interleave,
        ),
        device=torch.device("cpu"),
    )["layer-0"]


def test_logical_cache_pa_nz_round_trip_preserves_head_chunks() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import (
        _logical_cache_to_pa_nz,
        _pa_nz_to_logical_cache,
    )

    cache = torch.arange(4 * 128 * 2 * 128, dtype=torch.float32).reshape(4, 128, 2, 128)
    block_ids = torch.tensor([3, 1], dtype=torch.int64)
    selected = cache.index_select(0, block_ids)

    cache_nz = _logical_cache_to_pa_nz(cache, block_ids, head_size=128)
    restored = _pa_nz_to_logical_cache(cache_nz, head_size=128)

    assert cache_nz.shape == (2, 16, 128, 16)
    torch.testing.assert_close(restored, selected)


def test_paged_config_uses_ascend_kernel_block_size() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    NPUOmniPlatform.configure_diffusion_vllm_config(
        vllm_config,
        SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER),
    )

    assert vllm_config.cache_config.block_size == AscendAttentionBackend.get_supported_kernel_block_sizes()[0]


@pytest.mark.parametrize(
    ("row_specs", "cp_size", "expected_blocks", "expected_slots"),
    [
        pytest.param(
            [(0, 5, 7, 2, 8, (7, 3))],
            1,
            [3, 7],
            [6, 7, 0, 1, 2],
            id="cp1-noncontiguous",
        ),
        pytest.param(
            [
                (0, 6, 6, 0, 16, (5, 9)),
                (1, 4, 8, 4, 16, (2, 7)),
            ],
            2,
            [2, 5],
            [4, 5, -1, -1, 6, 7, 2, 3, -1, -1],
            id="cp2-multirow",
        ),
    ],
)
def test_diffusion_write_plan_maps_scheduler_blocks(row_specs, cp_size, expected_blocks, expected_slots) -> None:
    rows = [
        DiffusionPagedAttentionRow(
            request_id=f"req-{index}",
            sequence_id=sequence_id,
            query_len=query_len,
            seq_len=seq_len,
            kv_start_pos=start,
        )
        for index, (sequence_id, query_len, seq_len, start, _, _) in enumerate(row_specs)
    ]
    bindings = [
        DiffusionPagedAttentionRowBinding(row_index=index, max_seq_len=max_seq_len, block_ids=(blocks,))
        for index, (*_, max_seq_len, blocks) in enumerate(row_specs)
    ]
    plan = _build_write_plan(rows, bindings, cp_size=cp_size, cp_interleave=cp_size)

    assert plan.block_ids.tolist() == expected_blocks
    assert plan.local_slot_mapping.tolist() == expected_slots


@pytest.mark.parametrize("static_plan", [False, True], ids=["dynamic", "static"])
def test_cache_writer_skips_negative_slots_and_writes_back_logical_cache(monkeypatch, static_plan) -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import (
        _reshape_and_cache_without_cache_mode,
        _use_diffusion_paged_kv_write_plan,
    )

    calls = _install_fake_pa_scatter(monkeypatch)
    key_cache = torch.zeros(4, 4, 2, 16)
    value_cache = torch.zeros_like(key_cache)
    key = torch.arange(4 * 2 * 16, dtype=torch.float32).reshape(4, 2, 16) + 1
    value = key + 1000
    slots = torch.tensor([13, -1, 4, 7], dtype=torch.int64)
    plan = (
        DiffusionPagedKVWritePlan(
            block_ids=torch.tensor([3, 1], dtype=torch.int64),
            local_slot_mapping=torch.tensor([1, -1, 4, 7], dtype=torch.int64),
        )
        if static_plan
        else None
    )

    with _use_diffusion_paged_kv_write_plan(plan) if plan is not None else nullcontext():
        _reshape_and_cache_without_cache_mode(
            object,
            key,
            value,
            key_cache,
            value_cache,
            slots,
        )

    expected_local_slots = [1, -1, 4, 7] if static_plan else [5, -1, 0, 3]
    assert [call.tolist() for call in calls] == [expected_local_slots]
    for token_index in (0, 2, 3):
        block, offset = divmod(int(slots[token_index]), 4)
        torch.testing.assert_close(key_cache[block, offset], key[token_index])
        torch.testing.assert_close(value_cache[block, offset], value[token_index])
    assert torch.count_nonzero(key_cache) == 3 * key.shape[1] * key.shape[2]
