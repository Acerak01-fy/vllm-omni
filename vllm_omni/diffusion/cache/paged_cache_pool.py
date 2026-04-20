# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-allocated GPU page pool for DiT Fn/Bn cache buffers.

This is the low-level primitive of the paged cache-dit design
(see ``paged_cache_dit_design.md`` §3.2).

The pool holds a single contiguous tensor of shape
``[num_pages, page_size, hidden_dim]`` that is allocated once and never
freed during its lifetime. All subsequent cache storage is served by
handing out page indices from a free list, so there is no further
interaction with the PyTorch CUDA caching allocator for these buffers.
This is what turns external fragmentation into zero.

Only the pool primitive lives here. Triton kernels
(``paged_scatter_write`` / ``paged_residual_add`` / ``paged_l2_diff``),
the ``PagedCacheContext`` duck-type, the ``PagedCacheDiTStateDriver``
adapter, and the batched-forward rewiring all live in later phases and
depend on the API declared in this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

__all__ = [
    "PagePoolExhaustedError",
    "PagedCachePool",
    "estimate_pool_size",
]


class PagePoolExhaustedError(RuntimeError):
    """Raised when ``PagedCachePool.allocate`` cannot satisfy a request.

    Upper layers (admission control in the driver, Module 4) are expected
    to catch this and either reject / queue the incoming request or
    evict an existing slot. At the pool level we never block and never
    partially allocate.
    """

    def __init__(self, num_needed: int, num_available: int) -> None:
        super().__init__(f"PagedCachePool exhausted: need {num_needed} pages, only {num_available} available.")
        self.num_needed = num_needed
        self.num_available = num_available


class PagedCachePool:
    """Fixed-size GPU page pool for Fn/Bn cache buffers.

    The pool owns a single large tensor allocated at construction time
    and hands out integer page ids through a stack-based free list.
    Callers are responsible for returning page ids via :meth:`free` when
    the owning request slot is released.

    Thread-safety: not synchronized. The design assumes a single
    DiT worker thread drives forward passes (consistent with the rest
    of the diffusion worker). Add locking only if that assumption
    changes.
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        pin_on_init: bool = True,
    ) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be > 0, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self._num_pages = int(num_pages)
        self._page_size = int(page_size)
        self._hidden_dim = int(hidden_dim)
        self._dtype = dtype
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

        # One-shot allocation. ``torch.empty`` is intentional — the pool is
        # always fully overwritten before it is read, and zeroing 2+ GB on
        # every worker start-up is pure latency tax.
        if pin_on_init:
            self._page_pool = torch.empty(
                (self._num_pages, self._page_size, self._hidden_dim),
                dtype=self._dtype,
                device=self._device,
            )
        else:
            # Deferred allocation is useful for tests that only exercise the
            # free-list bookkeeping without needing the backing storage.
            self._page_pool = None  # type: ignore[assignment]

        # Free list: stack of available page ids. Pop from the tail so the
        # most-recently-freed page is re-used first (keeps working set
        # small and cache-friendly). Pre-populate with all ids.
        self._free_list: list[int] = list(range(self._num_pages))

        # Tracks which ids are currently allocated — only used for
        # double-free detection. Kept as a set for O(1) membership.
        self._allocated: set[int] = set()

        # GPU-side scratch buffer for page-table uploads. Callers that
        # need a ``torch.IntTensor`` of page ids for Triton kernels should
        # go through :meth:`to_gpu_page_table` instead of creating new
        # tensors every step — that would defeat the whole point of the
        # pool.
        self._page_id_scratch: torch.Tensor | None = None

        # Observability
        self._peak_used: int = 0

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def page_pool_tensor(self) -> torch.Tensor:
        """Full pool tensor for use as a Triton kernel base pointer.

        Module 2 (Triton kernels) reads this as the single storage
        operand — every ``page_id`` addresses a slab
        ``[page_size, hidden_dim]`` inside it.
        """
        if self._page_pool is None:
            raise RuntimeError(
                "PagedCachePool was constructed with pin_on_init=False; backing tensor is not materialized."
            )
        return self._page_pool

    @property
    def num_free_pages(self) -> int:
        return len(self._free_list)

    @property
    def num_used_pages(self) -> int:
        return self._num_pages - len(self._free_list)

    # ── Core API ────────────────────────────────────────────────────────

    def allocate(self, num_tokens: int) -> list[int]:
        """Reserve enough pages to hold ``num_tokens`` contiguous tokens.

        Pages are not cleared — callers must scatter-write before reading.

        Raises:
            ValueError: ``num_tokens`` is negative.
            PagePoolExhaustedError: insufficient free pages. No partial
                allocation is performed: on failure the pool state is
                unchanged.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        if num_tokens == 0:
            return []

        num_needed = math.ceil(num_tokens / self._page_size)
        if num_needed > len(self._free_list):
            raise PagePoolExhaustedError(num_needed, len(self._free_list))

        # Atomic-looking allocation: pop exactly ``num_needed`` ids from
        # the tail of the free list. Because the pre-check above
        # guarantees capacity, we can pop without worrying about rollback.
        page_ids = [self._free_list.pop() for _ in range(num_needed)]
        self._allocated.update(page_ids)

        used = self.num_used_pages
        if used > self._peak_used:
            self._peak_used = used
        return page_ids

    def free(self, page_ids: Iterable[int]) -> None:
        """Return ``page_ids`` to the free list.

        Double-free and out-of-range ids raise ``ValueError`` rather than
        being silently accepted — a leaked or duplicate id would manifest
        as cross-request memory corruption, which is worse than failing
        loud.
        """
        # Materialize once so we can validate before mutating state.
        ids = list(page_ids)
        if not ids:
            return

        seen: set[int] = set()
        for pid in ids:
            if not 0 <= pid < self._num_pages:
                raise ValueError(f"page id {pid} is out of range [0, {self._num_pages})")
            if pid in seen:
                raise ValueError(f"duplicate page id {pid} in free() call")
            seen.add(pid)
            if pid not in self._allocated:
                raise ValueError(f"page id {pid} is not currently allocated (double-free?)")

        for pid in ids:
            self._allocated.discard(pid)
            self._free_list.append(pid)

    def get_page_tensor(self, page_id: int) -> torch.Tensor:
        """Return a zero-copy view of page ``page_id``.

        Shape: ``[page_size, hidden_dim]``.
        """
        if not 0 <= page_id < self._num_pages:
            raise ValueError(f"page id {page_id} is out of range [0, {self._num_pages})")
        return self.page_pool_tensor[page_id]

    # ── GPU page-table helper (used by Module 2 kernels) ────────────────

    def to_gpu_page_table(self, page_ids: list[int]) -> torch.Tensor:
        """Materialize ``page_ids`` as a GPU ``int32`` tensor.

        Uses a persistent device-side scratch buffer so repeated calls
        don't allocate. The returned tensor is a view into the scratch
        and is only valid until the next call on this pool — callers
        that need stable storage across steps should keep their own
        per-slot tensor (this is what ``PagedCacheContext`` will do in
        Module 3).
        """
        n = len(page_ids)
        if n > self._num_pages:
            raise ValueError(f"page_ids of length {n} exceeds pool capacity {self._num_pages}")
        if self._page_id_scratch is None:
            self._page_id_scratch = torch.empty(self._num_pages, dtype=torch.int32, device=self._device)
        if n == 0:
            return self._page_id_scratch[:0]
        src = torch.as_tensor(page_ids, dtype=torch.int32)
        self._page_id_scratch[:n].copy_(src, non_blocking=True)
        return self._page_id_scratch[:n]

    # ── Observability ───────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of pool occupancy.

        ``fragmentation`` is always 0.0 by construction — this is the
        whole reason the pool exists, and the field is kept in the
        dict so downstream metrics plumbing does not need to special-case
        the paged backend.
        """
        used = self.num_used_pages
        page_bytes = self._page_size * self._hidden_dim * _dtype_bytes(self._dtype)
        return {
            "num_pages": self._num_pages,
            "free_pages": self.num_free_pages,
            "used_pages": used,
            "peak_used_pages": self._peak_used,
            "page_size": self._page_size,
            "hidden_dim": self._hidden_dim,
            "dtype": str(self._dtype),
            "bytes_per_page": page_bytes,
            "total_bytes": self._num_pages * page_bytes,
            "used_bytes": used * page_bytes,
            "fragmentation": 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"PagedCachePool(num_pages={self._num_pages}, "
            f"page_size={self._page_size}, hidden_dim={self._hidden_dim}, "
            f"dtype={self._dtype}, device={self._device}, "
            f"free={self.num_free_pages}, used={self.num_used_pages})"
        )


# ── Sizing helper ───────────────────────────────────────────────────────


def estimate_pool_size(
    max_concurrent_requests: int,
    max_seq_len: int,
    num_blocks: int,
    buffers_per_block: int = 3,
    page_size: int = 16,
    safety_factor: float = 1.15,
) -> int:
    """Estimate ``num_pages`` needed to host a worst-case workload.

    The default assumptions match the design doc's Wan2.2 example:
      * 3 buffers per block (Fn_residual, Bn_residual, Bn_encoder)
      * page_size 16 tokens
      * 15% slack to absorb encoder seq-len rounding and peak concurrency

    Args:
        max_concurrent_requests: Expected worst-case in-flight requests.
        max_seq_len: Largest per-buffer token count any single request
            will need (upper bound of ``latent_seq_len`` and
            ``txt_seq_len`` if pools are shared).
        num_blocks: Transformer depth.
        buffers_per_block: How many distinct buffers are cached per
            block (Fn_res + Bn_res + Bn_enc → 3 by default).
        page_size: Page granularity in tokens.
        safety_factor: Multiplicative slack on top of the tight bound.

    Returns:
        An integer ``num_pages`` suitable for :class:`PagedCachePool`.
    """
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if buffers_per_block <= 0:
        raise ValueError("buffers_per_block must be > 0")
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0")

    pages_per_buffer = math.ceil(max_seq_len / page_size)
    pages_per_request = pages_per_buffer * num_blocks * buffers_per_block
    tight = pages_per_request * max_concurrent_requests
    return int(math.ceil(tight * safety_factor))


# ── Internals ───────────────────────────────────────────────────────────


def _dtype_bytes(dtype: torch.dtype) -> int:
    # ``torch.tensor`` of size 0 is the cheapest portable way to read
    # ``element_size`` without constructing a real tensor with storage.
    return torch.empty(0, dtype=dtype).element_size()
