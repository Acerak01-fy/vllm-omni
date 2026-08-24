# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
