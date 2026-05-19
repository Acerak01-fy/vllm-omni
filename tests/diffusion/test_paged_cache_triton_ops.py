# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from cache_dit.caching.cache_contexts.cache_context import CachedContext

from vllm_omni.diffusion.cache.kernels.paged_cache_ops import (
    get_paged_cache_kernel_stats,
    paged_abs_diff_stats,
    paged_l2_diff,
    paged_residual_add_,
    paged_scatter_write,
    reset_paged_cache_kernel_stats,
)
from vllm_omni.diffusion.cache.paged_cache_context import PagedCacheContext
from vllm_omni.diffusion.cache.paged_cache_pool import PagedCachePool

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def test_paged_cache_triton_ops_match_torch_gather_baseline():
    reset_paged_cache_kernel_stats()
    device = torch.device("cuda")
    page_size = 2
    hidden_dim = 5
    num_tokens = 5
    pool = torch.empty((8, page_size, hidden_dim), dtype=torch.float32, device=device)
    page_table = torch.tensor([7, 3, 5], dtype=torch.int32, device=device)
    data = torch.arange(num_tokens * hidden_dim, dtype=torch.float32, device=device).reshape(num_tokens, hidden_dim)

    paged_scatter_write(
        data,
        pool,
        page_table,
        num_tokens=num_tokens,
        page_size=page_size,
        hidden_dim=hidden_dim,
    )

    gathered = torch.index_select(pool, 0, page_table).reshape(-1, hidden_dim)[:num_tokens]
    torch.testing.assert_close(gathered, data)

    residual_target = torch.ones_like(data)
    paged_residual_add_(
        residual_target,
        pool,
        page_table,
        num_tokens=num_tokens,
        page_size=page_size,
        hidden_dim=hidden_dim,
        add_input=True,
    )
    torch.testing.assert_close(residual_target, data + 1)

    copy_target = torch.empty_like(data)
    paged_residual_add_(
        copy_target,
        pool,
        page_table,
        num_tokens=num_tokens,
        page_size=page_size,
        hidden_dim=hidden_dim,
        add_input=False,
    )
    torch.testing.assert_close(copy_target, data)

    probe = data + 0.5
    l2 = paged_l2_diff(
        probe,
        pool,
        page_table,
        num_tokens=num_tokens,
        page_size=page_size,
        hidden_dim=hidden_dim,
    )
    torch.testing.assert_close(l2, ((probe - data) ** 2).sum())

    stats = paged_abs_diff_stats(
        probe,
        pool,
        page_table,
        num_tokens=num_tokens,
        page_size=page_size,
        hidden_dim=hidden_dim,
    )
    expected_stats = torch.stack([(probe - data).abs().sum(), data.abs().sum()])
    torch.testing.assert_close(stats, expected_stats)

    assert get_paged_cache_kernel_stats() == {
        "scatter_write_triton": 1,
        "residual_add_triton": 1,
        "copy_from_cache_triton": 1,
        "l2_diff_triton": 1,
        "abs_diff_triton": 1,
    }


def test_paged_cache_context_uses_triton_ops_on_cuda():
    pool = PagedCachePool(
        num_pages=8,
        page_size=2,
        hidden_dim=4,
        dtype=torch.float32,
        device="cuda",
    )
    context = PagedCacheContext(CachedContext(name="ctx"), pool)
    data = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)

    context.set_buffer("hidden", data)

    torch.testing.assert_close(context.get_buffer("hidden"), data)

    target = torch.ones_like(data)
    applied = context.apply_buffer("hidden", target, residual=True)

    torch.testing.assert_close(applied, data + 1)
    torch.testing.assert_close(target, data + 1)

    probe = data + 2
    expected_ratio = ((probe - data).abs().sum() / data.abs().sum()).item()

    assert context.mean_abs_diff_ratio("hidden", probe) == pytest.approx(expected_ratio)
