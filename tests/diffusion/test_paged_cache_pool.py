# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from cache_dit.caching.cache_contexts.cache_context import CachedContext

from vllm_omni.diffusion.cache.paged_cache_context import PagedCacheContext
from vllm_omni.diffusion.cache.paged_cache_pool import (
    PagedCachePool,
    PagePoolExhaustedError,
    estimate_pool_size,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestPagedCachePool:
    def test_allocate_free_and_stats(self):
        pool = PagedCachePool(
            num_pages=4,
            page_size=2,
            hidden_dim=3,
            dtype=torch.float32,
            device="cpu",
        )

        pages = pool.allocate(num_tokens=3)

        assert pages == [3, 2]
        assert pool.num_used_pages == 2
        assert pool.num_free_pages == 2
        assert pool.stats()["used_bytes"] == 2 * 2 * 3 * 4

        pool.free(pages)

        assert pool.num_used_pages == 0
        assert pool.num_free_pages == 4
        assert pool.stats()["peak_used_pages"] == 2

    def test_exhaustion_does_not_partially_allocate(self):
        pool = PagedCachePool(
            num_pages=2,
            page_size=4,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
            pin_on_init=False,
        )

        with pytest.raises(PagePoolExhaustedError) as exc_info:
            pool.allocate(num_tokens=9)

        assert exc_info.value.num_needed == 3
        assert exc_info.value.num_available == 2
        assert pool.num_free_pages == 2
        assert pool.num_used_pages == 0

    def test_free_rejects_invalid_duplicate_and_double_free_ids(self):
        pool = PagedCachePool(
            num_pages=3,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        pages = pool.allocate(num_tokens=2)

        with pytest.raises(ValueError, match="duplicate page id"):
            pool.free([pages[0], pages[0]])
        with pytest.raises(ValueError, match="out of range"):
            pool.free([99])

        pool.free(pages)

        with pytest.raises(ValueError, match="double-free"):
            pool.free(pages)

    def test_page_table_scratch_reuses_storage(self):
        pool = PagedCachePool(
            num_pages=4,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )

        table = pool.to_gpu_page_table([3, 1])
        data_ptr = table.data_ptr()

        assert table.dtype == torch.int32
        assert table.tolist() == [3, 1]

        next_table = pool.to_gpu_page_table([2])

        assert next_table.data_ptr() == data_ptr
        assert next_table.tolist() == [2]

    def test_estimate_pool_size_uses_ceil_pages_and_safety_factor(self):
        assert (
            estimate_pool_size(
                max_concurrent_requests=2,
                max_seq_len=5,
                num_blocks=3,
                buffers_per_block=2,
                page_size=4,
                safety_factor=1.25,
            )
            == 30
        )


class TestPagedCacheContext:
    def test_set_get_and_clear_paged_tensor(self):
        pool = PagedCachePool(
            num_pages=4,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        context = PagedCacheContext(CachedContext(name="ctx"), pool)
        data = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

        context.set_buffer("hidden", data)

        assert context.page_tables["hidden"] == [3, 2, 1]
        assert pool.num_used_pages == 3
        assert "hidden" not in context.base_context.buffers
        assert torch.equal(context.get_buffer("hidden"), data)

        data.add_(100)
        assert not torch.equal(context.get_buffer("hidden"), data)

        context.clear_buffers()

        assert pool.num_used_pages == 0
        assert context.page_tables == {}
        assert context.get_buffer("hidden") is None

    def test_reuses_pages_for_same_shape_and_resizes_on_shape_change(self):
        pool = PagedCachePool(
            num_pages=4,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        context = PagedCacheContext(CachedContext(name="ctx"), pool)

        context.set_buffer("hidden", torch.ones((1, 4, 2), dtype=torch.float32))
        page_ids = context.page_tables["hidden"]
        context.set_buffer("hidden", torch.full((1, 4, 2), 2.0, dtype=torch.float32))

        assert context.page_tables["hidden"] == page_ids
        assert pool.num_used_pages == 2
        assert torch.equal(
            context.get_buffer("hidden"),
            torch.full((1, 4, 2), 2.0, dtype=torch.float32),
        )

        context.set_buffer("hidden", torch.full((1, 1, 2), 3.0, dtype=torch.float32))

        assert len(context.page_tables["hidden"]) == 1
        assert pool.num_used_pages == 1
        assert torch.equal(
            context.get_buffer("hidden"),
            torch.full((1, 1, 2), 3.0, dtype=torch.float32),
        )

    def test_non_matching_tensor_falls_back_to_wrapped_context(self):
        pool = PagedCachePool(
            num_pages=2,
            page_size=2,
            hidden_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        context = PagedCacheContext(CachedContext(name="ctx"), pool)
        metadata = torch.tensor([1.0], dtype=torch.float32)

        context.set_buffer("metadata", metadata)

        assert pool.num_used_pages == 0
        assert context.page_tables == {}
        assert context.get_buffer("metadata") is metadata
