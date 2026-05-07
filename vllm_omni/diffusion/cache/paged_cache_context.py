# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather-baseline context wrapper for paged Cache-DiT buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.cache.paged_cache_pool import PagedCachePool

__all__ = ["PagedBufferEntry", "PagedCacheContext"]


@dataclass
class PagedBufferEntry:
    """Page-table metadata for one logical Cache-DiT buffer."""

    page_ids: list[int]
    shape: torch.Size
    num_tokens: int
    page_table: torch.Tensor


class PagedCacheContext:
    """Wrap a cache-dit context and store eligible tensor buffers in pages.

    Phase A intentionally keeps cache-dit's higher-level logic untouched:
    warmup, SCM, CFG step accounting, and similarity decisions still run
    through ``CachedContextManager``.  This wrapper only replaces the
    low-level ``set_buffer`` / ``get_buffer`` / ``clear_buffers`` storage path.
    """

    def __init__(self, base_context: Any, pool: PagedCachePool) -> None:
        object.__setattr__(self, "_base_context", base_context)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_paged_buffers", {})

    @property
    def base_context(self) -> Any:
        return self._base_context

    @property
    def pool(self) -> PagedCachePool:
        return self._pool

    @property
    def paged_buffers(self) -> dict[str, PagedBufferEntry]:
        return self._paged_buffers

    @property
    def page_tables(self) -> dict[str, list[int]]:
        return {name: list(entry.page_ids) for name, entry in self._paged_buffers.items()}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_context, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_base_context", "_pool", "_paged_buffers"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base_context, name, value)

    def set_buffer(self, name: str, buffer: Any) -> None:
        """Store ``buffer`` either in the page pool or in the wrapped context."""
        if not self._can_page(buffer):
            self._free_paged_buffer(name)
            self._set_base_buffer(name, buffer)
            return

        assert isinstance(buffer, torch.Tensor)
        entry = self._ensure_entry(name, buffer)
        if name in self._base_context.buffers:
            self._remove_base_buffer(name)

        flat = buffer.detach().reshape(-1, self._pool.hidden_dim)
        if not flat.is_contiguous():
            flat = flat.contiguous()

        for page_idx, page_id in enumerate(entry.page_ids):
            start = page_idx * self._pool.page_size
            end = min(start + self._pool.page_size, entry.num_tokens)
            if start >= end:
                break
            self._pool.get_page_tensor(page_id)[: end - start].copy_(flat[start:end])

    def get_buffer(self, name: str) -> torch.Tensor | Any | None:
        """Return a contiguous tensor view of a paged logical buffer."""
        entry = self._paged_buffers.get(name)
        if entry is None:
            return self._get_base_buffer(name)
        if entry.num_tokens == 0:
            return torch.empty(tuple(entry.shape), dtype=self._pool.dtype, device=self._pool.device)

        pages = torch.index_select(self._pool.page_pool_tensor, 0, entry.page_table)
        flat = pages.reshape(-1, self._pool.hidden_dim)[: entry.num_tokens]
        return flat.reshape(tuple(entry.shape)).contiguous()

    def remove_buffer(self, name: str) -> None:
        self._free_paged_buffer(name)
        self._remove_base_buffer(name)

    def clear_buffers(self) -> None:
        for entry in list(self._paged_buffers.values()):
            self._pool.free(entry.page_ids)
        self._paged_buffers.clear()
        self._base_context.clear_buffers()

    def resident_bytes(self) -> int:
        page_bytes = (
            self._pool.page_size * self._pool.hidden_dim * torch.empty(0, dtype=self._pool.dtype).element_size()
        )
        return sum(len(entry.page_ids) * page_bytes for entry in self._paged_buffers.values())

    def _can_page(self, buffer: Any) -> bool:
        if not isinstance(buffer, torch.Tensor) or buffer.ndim == 0:
            return False
        if int(buffer.shape[-1]) != self._pool.hidden_dim:
            return False
        if buffer.dtype != self._pool.dtype or buffer.device != self._pool.device:
            return False
        return True

    def _ensure_entry(self, name: str, buffer: torch.Tensor) -> PagedBufferEntry:
        num_tokens = buffer.numel() // self._pool.hidden_dim
        required_pages = math.ceil(num_tokens / self._pool.page_size) if num_tokens else 0
        entry = self._paged_buffers.get(name)

        if entry is None:
            page_ids = self._pool.allocate(num_tokens)
            entry = PagedBufferEntry(
                page_ids=page_ids,
                shape=torch.Size(buffer.shape),
                num_tokens=num_tokens,
                page_table=self._make_page_table(page_ids),
            )
            self._paged_buffers[name] = entry
            return entry

        if required_pages > len(entry.page_ids):
            extra_pages = self._pool.allocate((required_pages - len(entry.page_ids)) * self._pool.page_size)
            entry.page_ids.extend(extra_pages)
        elif required_pages < len(entry.page_ids):
            self._pool.free(entry.page_ids[required_pages:])
            entry.page_ids = entry.page_ids[:required_pages]

        entry.shape = torch.Size(buffer.shape)
        entry.num_tokens = num_tokens
        entry.page_table = self._make_page_table(entry.page_ids)
        return entry

    def _free_paged_buffer(self, name: str) -> None:
        entry = self._paged_buffers.pop(name, None)
        if entry is not None:
            self._pool.free(entry.page_ids)

    def _make_page_table(self, page_ids: list[int]) -> torch.Tensor:
        return torch.tensor(page_ids, dtype=torch.long, device=self._pool.device)

    def _set_base_buffer(self, name: str, buffer: Any) -> None:
        if hasattr(self._base_context, "set_buffer"):
            self._base_context.set_buffer(name, buffer)
        else:
            self._base_context.buffers[name] = buffer

    def _get_base_buffer(self, name: str) -> Any | None:
        if hasattr(self._base_context, "get_buffer"):
            return self._base_context.get_buffer(name)
        return self._base_context.buffers.get(name)

    def _remove_base_buffer(self, name: str) -> None:
        if hasattr(self._base_context, "remove_buffer"):
            self._base_context.remove_buffer(name)
        else:
            self._base_context.buffers.pop(name, None)
