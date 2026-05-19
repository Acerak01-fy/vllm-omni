# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Cache-DiT storage kernels.

The public functions in this module are intentionally small wrappers:
CUDA tensors use Triton kernels, while CPU tensors keep a torch fallback
so unit tests and non-CUDA development paths remain usable.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on minimal envs.
    triton = None
    tl = None

_DEFAULT_BLOCK_SIZE = 1024
_KERNEL_STATS: dict[str, int] = {}


def reset_paged_cache_kernel_stats() -> None:
    """Reset per-process paged cache kernel counters."""
    _KERNEL_STATS.clear()


def get_paged_cache_kernel_stats() -> dict[str, int]:
    """Return per-process paged cache kernel counters."""
    return dict(_KERNEL_STATS)


def _count(name: str) -> None:
    _KERNEL_STATS[name] = _KERNEL_STATS.get(name, 0) + 1


if triton is not None and tl is not None:

    @triton.jit
    def _paged_scatter_write_kernel(
        src_ptr,
        page_pool_ptr,
        page_table_ptr,
        total_elements,
        page_size: tl.constexpr,
        hidden_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_elements
        token_ids = offsets // hidden_dim
        hidden_offsets = offsets - token_ids * hidden_dim
        page_table_offsets = token_ids // page_size
        offsets_in_page = token_ids - page_table_offsets * page_size
        page_ids = tl.load(page_table_ptr + page_table_offsets, mask=mask, other=0).to(tl.int64)
        dst_offsets = (page_ids * page_size + offsets_in_page) * hidden_dim + hidden_offsets
        values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(page_pool_ptr + dst_offsets, values, mask=mask)

    @triton.jit
    def _paged_residual_add_kernel(
        dst_ptr,
        page_pool_ptr,
        page_table_ptr,
        total_elements,
        page_size: tl.constexpr,
        hidden_dim: tl.constexpr,
        add_input: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_elements
        token_ids = offsets // hidden_dim
        hidden_offsets = offsets - token_ids * hidden_dim
        page_table_offsets = token_ids // page_size
        offsets_in_page = token_ids - page_table_offsets * page_size
        page_ids = tl.load(page_table_ptr + page_table_offsets, mask=mask, other=0).to(tl.int64)
        cache_offsets = (page_ids * page_size + offsets_in_page) * hidden_dim + hidden_offsets
        cache_values = tl.load(page_pool_ptr + cache_offsets, mask=mask, other=0.0)
        if add_input:
            dst_values = tl.load(dst_ptr + offsets, mask=mask, other=0.0)
            cache_values += dst_values
        tl.store(dst_ptr + offsets, cache_values, mask=mask)

    @triton.jit
    def _paged_l2_diff_kernel(
        src_ptr,
        page_pool_ptr,
        page_table_ptr,
        partials_ptr,
        total_elements,
        page_size: tl.constexpr,
        hidden_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_elements
        token_ids = offsets // hidden_dim
        hidden_offsets = offsets - token_ids * hidden_dim
        page_table_offsets = token_ids // page_size
        offsets_in_page = token_ids - page_table_offsets * page_size
        page_ids = tl.load(page_table_ptr + page_table_offsets, mask=mask, other=0).to(tl.int64)
        cache_offsets = (page_ids * page_size + offsets_in_page) * hidden_dim + hidden_offsets
        src_values = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cache_values = tl.load(page_pool_ptr + cache_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = src_values - cache_values
        partial = tl.sum(tl.where(mask, diff * diff, 0.0))
        tl.store(partials_ptr + tl.program_id(0), partial)

    @triton.jit
    def _paged_abs_diff_stats_kernel(
        src_ptr,
        page_pool_ptr,
        page_table_ptr,
        partials_ptr,
        total_elements,
        num_blocks,
        page_size: tl.constexpr,
        hidden_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_elements
        token_ids = offsets // hidden_dim
        hidden_offsets = offsets - token_ids * hidden_dim
        page_table_offsets = token_ids // page_size
        offsets_in_page = token_ids - page_table_offsets * page_size
        page_ids = tl.load(page_table_ptr + page_table_offsets, mask=mask, other=0).to(tl.int64)
        cache_offsets = (page_ids * page_size + offsets_in_page) * hidden_dim + hidden_offsets
        src_values = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cache_values = tl.load(page_pool_ptr + cache_offsets, mask=mask, other=0.0).to(tl.float32)
        abs_diff = tl.abs(src_values - cache_values)
        abs_cache = tl.abs(cache_values)
        pid = tl.program_id(0)
        tl.store(partials_ptr + pid, tl.sum(tl.where(mask, abs_diff, 0.0)))
        tl.store(partials_ptr + num_blocks + pid, tl.sum(tl.where(mask, abs_cache, 0.0)))


def paged_scatter_write(
    src: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> None:
    """Write contiguous ``src`` tokens into ``page_pool`` pages."""
    _validate_common(src, page_pool, page_table, num_tokens, page_size, hidden_dim)
    if num_tokens == 0:
        return
    src_flat = _flatten_token_tensor(src, num_tokens, hidden_dim)
    if _use_triton(src_flat, page_pool, page_table):
        _count("scatter_write_triton")
        total_elements = num_tokens * hidden_dim
        grid = (triton.cdiv(total_elements, _DEFAULT_BLOCK_SIZE),)
        _paged_scatter_write_kernel[grid](
            src_flat,
            page_pool,
            page_table,
            total_elements,
            page_size,
            hidden_dim,
            block_size=_DEFAULT_BLOCK_SIZE,
        )
        return
    _count("scatter_write_torch")
    _torch_scatter_write(src_flat, page_pool, page_table, num_tokens, page_size, hidden_dim)


def paged_residual_add_(
    dst: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
    add_input: bool = True,
) -> None:
    """Apply paged cache data to ``dst``.

    With ``add_input=True`` this computes ``dst += cache`` in-place.
    With ``add_input=False`` this computes ``dst = cache`` in-place.
    """
    _validate_common(dst, page_pool, page_table, num_tokens, page_size, hidden_dim)
    if num_tokens == 0:
        return
    dst_flat = _flatten_token_tensor(dst, num_tokens, hidden_dim, require_contiguous=True)
    if _use_triton(dst_flat, page_pool, page_table):
        _count("residual_add_triton" if add_input else "copy_from_cache_triton")
        total_elements = num_tokens * hidden_dim
        grid = (triton.cdiv(total_elements, _DEFAULT_BLOCK_SIZE),)
        _paged_residual_add_kernel[grid](
            dst_flat,
            page_pool,
            page_table,
            total_elements,
            page_size,
            hidden_dim,
            add_input,
            block_size=_DEFAULT_BLOCK_SIZE,
        )
        return
    _count("residual_add_torch" if add_input else "copy_from_cache_torch")
    _torch_apply_cache(dst_flat, page_pool, page_table, num_tokens, page_size, hidden_dim, add_input)


def paged_l2_diff(
    src: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Return ``sum((src - paged_cache) ** 2)`` as a scalar tensor."""
    _validate_common(src, page_pool, page_table, num_tokens, page_size, hidden_dim)
    src_flat = _flatten_token_tensor(src, num_tokens, hidden_dim)
    if num_tokens == 0:
        return torch.zeros((), dtype=torch.float32, device=src.device)
    if _use_triton(src_flat, page_pool, page_table):
        _count("l2_diff_triton")
        total_elements = num_tokens * hidden_dim
        num_blocks = triton.cdiv(total_elements, _DEFAULT_BLOCK_SIZE)
        partials = torch.empty((num_blocks,), dtype=torch.float32, device=src.device)
        _paged_l2_diff_kernel[(num_blocks,)](
            src_flat,
            page_pool,
            page_table,
            partials,
            total_elements,
            page_size,
            hidden_dim,
            block_size=_DEFAULT_BLOCK_SIZE,
        )
        return partials.sum()
    _count("l2_diff_torch")
    cache = _torch_gather(page_pool, page_table, num_tokens, page_size, hidden_dim)
    diff = src_flat.float() - cache.float()
    return (diff * diff).sum()


def paged_abs_diff_stats(
    src: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Return ``[sum(abs(src-cache)), sum(abs(cache))]`` as fp32."""
    _validate_common(src, page_pool, page_table, num_tokens, page_size, hidden_dim)
    src_flat = _flatten_token_tensor(src, num_tokens, hidden_dim)
    if num_tokens == 0:
        return torch.zeros((2,), dtype=torch.float32, device=src.device)
    if _use_triton(src_flat, page_pool, page_table):
        _count("abs_diff_triton")
        total_elements = num_tokens * hidden_dim
        num_blocks = triton.cdiv(total_elements, _DEFAULT_BLOCK_SIZE)
        partials = torch.empty((2, num_blocks), dtype=torch.float32, device=src.device)
        _paged_abs_diff_stats_kernel[(num_blocks,)](
            src_flat,
            page_pool,
            page_table,
            partials,
            total_elements,
            num_blocks,
            page_size,
            hidden_dim,
            block_size=_DEFAULT_BLOCK_SIZE,
        )
        return partials.sum(dim=1)
    _count("abs_diff_torch")
    cache = _torch_gather(page_pool, page_table, num_tokens, page_size, hidden_dim)
    src_float = src_flat.float()
    cache_float = cache.float()
    return torch.stack(
        [
            (src_float - cache_float).abs().sum(),
            cache_float.abs().sum(),
        ]
    )


def _validate_common(
    tensor: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> None:
    if tensor.device != page_pool.device or page_table.device != page_pool.device:
        raise ValueError("tensor, page_pool, and page_table must be on the same device")
    if int(page_pool.shape[1]) != page_size or int(page_pool.shape[2]) != hidden_dim:
        raise ValueError("page_pool shape does not match page_size/hidden_dim")
    if int(tensor.numel()) < num_tokens * hidden_dim:
        raise ValueError("tensor does not contain enough elements for num_tokens * hidden_dim")
    required_pages = math.ceil(num_tokens / page_size) if num_tokens else 0
    if int(page_table.numel()) < required_pages:
        raise ValueError("page_table does not contain enough page ids")


def _flatten_token_tensor(
    tensor: torch.Tensor,
    num_tokens: int,
    hidden_dim: int,
    *,
    require_contiguous: bool = False,
) -> torch.Tensor:
    flat = tensor.reshape(-1, hidden_dim)
    if int(flat.shape[0]) < num_tokens:
        raise ValueError("tensor has fewer flattened tokens than num_tokens")
    flat = flat[:num_tokens]
    if not flat.is_contiguous():
        if require_contiguous:
            raise ValueError("in-place paged cache operations require a contiguous target tensor")
        flat = flat.contiguous()
    return flat.reshape(-1)


def _use_triton(tensor: torch.Tensor, page_pool: torch.Tensor, page_table: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and tensor.is_cuda
        and page_pool.is_cuda
        and page_table.is_cuda
        and tensor.is_contiguous()
        and page_pool.is_contiguous()
        and page_table.is_contiguous()
    )


def _torch_gather(
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    del page_size
    pages = torch.index_select(page_pool, 0, page_table)
    return pages.reshape(-1, hidden_dim)[:num_tokens].reshape(-1)


def _torch_scatter_write(
    src_flat: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
) -> None:
    src_2d = src_flat.reshape(-1, hidden_dim)
    for page_idx, page_id_tensor in enumerate(page_table):
        start = page_idx * page_size
        end = min(start + page_size, num_tokens)
        if start >= end:
            break
        page_pool[int(page_id_tensor.item()), : end - start].copy_(src_2d[start:end])


def _torch_apply_cache(
    dst_flat: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: torch.Tensor,
    num_tokens: int,
    page_size: int,
    hidden_dim: int,
    add_input: bool,
) -> None:
    cache = _torch_gather(page_pool, page_table, num_tokens, page_size, hidden_dim)
    if add_input:
        dst_flat[: num_tokens * hidden_dim].add_(cache)
    else:
        dst_flat[: num_tokens * hidden_dim].copy_(cache)
