# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def test_static_diffusion_write_plan_compacts_noncontiguous_scheduler_blocks() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
        DiffusionPagedAttentionRow,
        DiffusionPagedAttentionRowBinding,
    )
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    rows = [
        DiffusionPagedAttentionRow(
            request_id="req-0",
            sequence_id=0,
            query_len=5,
            seq_len=7,
            kv_start_pos=2,
        )
    ]
    bindings = [
        DiffusionPagedAttentionRowBinding(
            row_index=0,
            max_seq_len=8,
            block_ids=((7, 3),),
        )
    ]
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer-0"])],
    )
    block_tables = SimpleNamespace(
        kernel_block_sizes=[4],
        blocks_per_kv_block=[1],
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )

    plan = NPUOmniPlatform.build_diffusion_paged_kv_write_plans(
        rows=rows,
        row_bindings=bindings,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        device=torch.device("cpu"),
    )["layer-0"]

    assert plan.block_ids.tolist() == [3, 7]
    assert plan.local_slot_mapping.tolist() == [6, 7, 0, 1, 2]
