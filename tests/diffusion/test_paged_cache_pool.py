# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from cache_dit.caching.cache_contexts.cache_context import CachedContext
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager

from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTBackend
from vllm_omni.diffusion.cache.cache_dit_driver import PagedCacheDiTStateDriver
from vllm_omni.diffusion.cache.paged_cache_context import PagedCacheContext
from vllm_omni.diffusion.cache.paged_cache_pool import (
    PagedCachePool,
    PagePoolExhaustedError,
    estimate_pool_size,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

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

    def test_estimate_pool_size_matches_wan_512_resolution_mapping(self):
        latent_seq_len = 16 * 16
        txt_seq_len = 77

        assert (
            estimate_pool_size(
                max_concurrent_requests=8,
                max_seq_len=max(latent_seq_len, txt_seq_len),
                num_blocks=40,
                buffers_per_block=3,
                page_size=16,
                safety_factor=1.0,
            )
            == 15360
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

    def test_apply_buffer_and_mean_abs_diff_cpu_fallback(self):
        pool = PagedCachePool(
            num_pages=4,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        context = PagedCacheContext(CachedContext(name="ctx"), pool)
        data = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)
        context.set_buffer("hidden", data)

        target = torch.ones_like(data)
        applied = context.apply_buffer("hidden", target, residual=True)

        assert torch.equal(applied, data + 1)
        assert torch.equal(target, data + 1)

        copied = context.apply_buffer("hidden", torch.empty_like(data), residual=False)

        assert torch.equal(copied, data)

        probe = data + 2
        expected_ratio = (probe - data).abs().sum() / data.abs().sum()

        assert context.mean_abs_diff_ratio("hidden", probe) == pytest.approx(expected_ratio.item())


class _ReplacingRefreshBackend:
    def force_refresh(self, pipeline, num_inference_steps: int, verbose: bool = False):
        del verbose
        manager = pipeline.transformer._context_manager
        for name in pipeline.transformer._context_names:
            current = manager.get_context(name)
            init_kwargs = dict(getattr(current, "_init_kwargs", {}))
            manager.remove_context(name)
            refreshed = manager.reset_context(name, **init_kwargs)
            refreshed.cache_config.num_inference_steps = num_inference_steps


def _make_context_pipeline(hidden_dim: int = 2):
    manager = CachedContextManager("paged-test", persistent_context=True)
    manager.new_context(name="ctx")
    transformer = torch.nn.Linear(hidden_dim, hidden_dim)
    transformer._context_manager = manager
    transformer._context_names = ("ctx",)
    transformer.inner_dim = hidden_dim
    return SimpleNamespace(transformer=transformer), manager


class TestPagedCacheDiTStateDriver:
    def test_paged_context_refresh_uses_wrapped_base_context(self):
        pipeline, manager = _make_context_pipeline()
        pool = PagedCachePool(
            num_pages=8,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )

        PagedCacheDiTStateDriver(_ReplacingRefreshBackend(), pipeline, pool)
        base_context = CachedContext(name="ctx")
        base_context.cache_config.num_inference_steps = 1
        base_context.executed_steps = 2
        paged_context = PagedCacheContext(base_context, pool)
        manager._current_context = paged_context

        should_refresh, reason = manager.maybe_refresh(paged_context)

        assert should_refresh is True
        assert reason == "num_inference_steps"

        should_refresh, reason = manager.maybe_refresh()

        assert should_refresh is True
        assert reason == "num_inference_steps"

    def test_initialize_rewraps_contexts_replaced_by_force_refresh(self):
        pipeline, manager = _make_context_pipeline()
        pool = PagedCachePool(
            num_pages=8,
            page_size=2,
            hidden_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        driver = PagedCacheDiTStateDriver(_ReplacingRefreshBackend(), pipeline, pool)
        slot = driver.create_empty_slot()

        driver.initialize_fresh_slot(slot, num_inference_steps=4)
        payload = PagedCacheDiTStateDriver._get_payload(slot)
        context = payload[0]["ctx"]

        assert isinstance(context, PagedCacheContext)
        assert manager.get_context("ctx") is context
        assert context.cache_config.num_inference_steps == 4

        data = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        context.set_buffer("hidden", data)

        assert torch.equal(context.get_buffer("hidden"), data)
        assert pool.num_used_pages == 3
        assert driver.estimate_slot_bytes(slot) == 3 * 2 * 2 * 4

        driver.clear_slot(slot)

        assert pool.num_used_pages == 0
        assert payload[0] == {}

    def test_cache_dit_backend_creates_paged_driver_when_enabled(self):
        pipeline, _ = _make_context_pipeline()
        backend = CacheDiTBackend(
            DiffusionCacheConfig(
                enable_paged_cache=True,
                paged_cache_num_pages=4,
                paged_cache_page_size=2,
                paged_cache_hidden_dim=2,
            )
        )
        backend.enabled = True

        driver = backend.create_state_driver(pipeline)

        assert isinstance(driver, PagedCacheDiTStateDriver)
        assert driver.pool.num_pages == 4
        assert driver.pool.page_size == 2
        assert driver.pool.hidden_dim == 2

    def test_cache_dit_backend_estimates_pool_size_from_model_shape(self):
        pipeline, _ = _make_context_pipeline(hidden_dim=4)
        pipeline.transformer.blocks = [object(), object(), object()]
        backend = CacheDiTBackend(
            DiffusionCacheConfig(
                enable_paged_cache=True,
                paged_cache_page_size=4,
                paged_cache_max_seq_len=5,
                paged_cache_max_concurrent_requests=2,
                paged_cache_buffers_per_block=3,
                paged_cache_safety_factor=1.0,
            )
        )
        backend.enabled = True

        driver = backend.create_state_driver(pipeline)

        assert isinstance(driver, PagedCacheDiTStateDriver)
        assert driver.pool.num_pages == 36
        assert driver.pool.page_size == 4
        assert driver.pool.hidden_dim == 4
