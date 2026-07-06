# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image3 prompt/ref KV cache backed by AR-Diffusion paging primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    allocate_kv_pool_with_views,
    compute_slot_mapping,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ar_diffusion_paged_attention,
)

_HY3_PAGED_KV_CACHE_ENV = "VLLM_OMNI_HY3_PAGED_KV_CACHE"
_HY3_PAGED_KV_PAGE_SIZE_ENV = "VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"
_HY3_PAGED_KV_DEFAULT_PAGE_SIZE = 16
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "enabled", "enable", "required"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})


def is_hunyuan_image3_paged_kv_cache_enabled() -> bool:
    value = os.environ.get(_HY3_PAGED_KV_CACHE_ENV)
    if value is None:
        return False
    value = value.strip().lower()
    if value in _DISABLED_VALUES:
        return False
    return value == "" or value in _ENABLED_VALUES


def is_hunyuan_image3_paged_kv_cache_required() -> bool:
    return os.environ.get(_HY3_PAGED_KV_CACHE_ENV, "").strip().lower() == "required"


def hunyuan_image3_paged_kv_page_size() -> int:
    value = os.environ.get(_HY3_PAGED_KV_PAGE_SIZE_ENV)
    if value is None or value.strip() == "":
        return _HY3_PAGED_KV_DEFAULT_PAGE_SIZE
    try:
        page_size = int(value)
    except ValueError:
        return _HY3_PAGED_KV_DEFAULT_PAGE_SIZE
    return page_size if page_size > 0 else _HY3_PAGED_KV_DEFAULT_PAGE_SIZE


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class HunyuanPromptKVRowRef:
    owner: HunyuanPromptKVPagePool
    block_ids: tuple[int, ...]
    lens: int


@dataclass(frozen=True)
class HunyuanPromptKVLayerRows:
    """Request-local view over one layer's persistent prompt/ref prefix pages."""

    owner: HunyuanPromptKVPagePool
    rows_by_branch: dict[int, HunyuanPromptKVRowRef]

    @property
    def lens(self) -> torch.Tensor:
        branches = sorted(self.rows_by_branch)
        values = [self.rows_by_branch[branch].lens for branch in branches]
        device = self.owner.device or torch.device("cpu")
        return torch.tensor(values, dtype=torch.long, device=device)

    def select_branch(self, branch: int) -> HunyuanPromptKVRowRef:
        try:
            return self.rows_by_branch[int(branch)]
        except KeyError as exc:
            raise KeyError(f"Hunyuan prompt KV branch {branch} was not captured.") from exc


@dataclass
class HunyuanPromptKVBatch:
    owner: HunyuanPromptKVPagePool
    row_refs: list[HunyuanPromptKVRowRef]

    @property
    def lens(self) -> torch.Tensor:
        device = self.owner.device or torch.device("cpu")
        return torch.tensor([row.lens for row in self.row_refs], dtype=torch.long, device=device)

    def view_rows(self, row_indices: list[int], branches: list[int]) -> HunyuanPromptKVLayerRows:
        if len(row_indices) != len(branches):
            raise ValueError("row_indices and branches must have the same length.")
        rows_by_branch: dict[int, HunyuanPromptKVRowRef] = {}
        for row_idx, branch in zip(row_indices, branches):
            rows_by_branch[int(branch)] = self.row_refs[int(row_idx)]
        return HunyuanPromptKVLayerRows(owner=self.owner, rows_by_branch=rows_by_branch)


@dataclass
class HunyuanPagedAttentionInputs:
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    prefix_blocks: int
    current_blocks: int


class HunyuanPromptKVPagePool:
    """Per-layer prompt/ref prefix page pool for Hunyuan Image3.

    The pool deliberately reuses the AR-Diffusion paged cache tensor layout and
    slot mapping helpers. Persistent blocks store first-step prompt/ref prefix
    KV. Later denoise steps reuse those block ids and reserve scratch pages for
    the current image KV of this forward.
    """

    def __init__(self, *, page_size: int | None = None, enabled: bool, required: bool) -> None:
        self.page_size = int(page_size or hunyuan_image3_paged_kv_page_size())
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        self.enabled = bool(enabled)
        self.required = bool(required)
        self._kv_pool: torch.Tensor | None = None
        self._k_pool: torch.Tensor | None = None
        self._v_pool: torch.Tensor | None = None
        self._num_blocks = 0
        self._persistent_blocks = 0
        self._current_batch: HunyuanPromptKVBatch | None = None
        self._shape: tuple[int, int] | None = None
        self.dtype: torch.dtype | None = None
        self.device: torch.device | None = None
        self.stats: dict[str, int] = {
            "paged_cache_builds": 0,
            "paged_attention_calls": 0,
            "paged_attention_fallbacks": 0,
            "paged_attention_errors": 0,
            "paged_prefix_blocks": 0,
            "paged_current_blocks": 0,
        }

    @property
    def active(self) -> bool:
        return self._current_batch is not None

    @property
    def current_batch(self) -> HunyuanPromptKVBatch | None:
        return self._current_batch

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> dict[str, int | bool]:
        return {
            "paged_kv_cache_enabled": self.enabled,
            "paged_kv_cache_required": self.required,
            "paged_kv_cache_active": self.active,
            "paged_kv_page_size": self.page_size,
            "paged_kv_num_blocks": self._num_blocks,
            "paged_kv_persistent_blocks": self._persistent_blocks,
            **self.stats,
        }

    def record_fallback(self) -> None:
        self.stats["paged_attention_fallbacks"] += 1

    def record_error(self) -> None:
        self.stats["paged_attention_errors"] += 1

    def _init_pool(self, *, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self._shape = (int(heads), int(head_dim))
        self.dtype = dtype
        self.device = device
        self._ensure_capacity(max(1, self._num_blocks))

    def _ensure_compatible(self, key: torch.Tensor) -> None:
        if key.dim() != 4:
            raise ValueError(f"Hunyuan paged KV expects 4D key/value tensors, got {tuple(key.shape)}")
        _, _, heads, head_dim = key.shape
        shape = (int(heads), int(head_dim))
        if self._kv_pool is None:
            self._init_pool(heads=heads, head_dim=head_dim, dtype=key.dtype, device=key.device)
            return
        if self._shape != shape:
            raise ValueError(f"Hunyuan paged KV shape changed from {self._shape} to {shape}.")
        if self.dtype != key.dtype:
            raise ValueError(f"Hunyuan paged KV dtype changed from {self.dtype} to {key.dtype}.")
        if self.device != key.device:
            raise ValueError(f"Hunyuan paged KV device changed from {self.device} to {key.device}.")

    def _ensure_capacity(self, num_blocks: int) -> None:
        if num_blocks <= self._num_blocks and self._kv_pool is not None:
            return
        if self._shape is None or self.dtype is None or self.device is None:
            raise RuntimeError("Hunyuan paged KV pool shape is not initialized.")

        new_num_blocks = max(int(num_blocks), max(1, self._num_blocks * 2))
        heads, head_dim = self._shape
        kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
            new_num_blocks,
            self.page_size,
            1,
            heads,
            head_dim,
            self.dtype,
            self.device,
        )
        if self._kv_pool is not None and self._num_blocks > 0:
            kv_pools[0][:, : self._num_blocks].copy_(self._kv_pool[:, : self._num_blocks])
        self._kv_pool = kv_pools[0]
        self._k_pool = k_pools[0]
        self._v_pool = v_pools[0]
        self._num_blocks = new_num_blocks

    def clear_current(self) -> None:
        self._current_batch = None

    def capture_prefix(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        lens: torch.Tensor,
    ) -> HunyuanPromptKVBatch:
        if not self.enabled:
            raise RuntimeError("Hunyuan paged KV capture called while disabled.")
        if key.shape != value.shape:
            raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
        self._ensure_compatible(key)
        assert self._k_pool is not None and self._v_pool is not None

        if lens.dim() != 1 or lens.numel() != key.shape[0]:
            raise ValueError("lens must be 1D and match the key/value batch size.")
        if torch.any(lens <= 0):
            raise ValueError("Hunyuan paged KV prefix lens must be positive.")
        if torch.any(lens > key.shape[1]):
            raise ValueError("Hunyuan paged KV prefix lens exceeds key/value length.")

        row_refs: list[HunyuanPromptKVRowRef] = []
        next_block = self._persistent_blocks
        total_new_blocks = 0
        for row in range(key.shape[0]):
            row_len = int(lens[row].item())
            num_blocks = _ceil_div(row_len, self.page_size)
            block_ids = tuple(range(next_block, next_block + num_blocks))
            next_block += num_blocks
            total_new_blocks += num_blocks
            row_refs.append(HunyuanPromptKVRowRef(owner=self, block_ids=block_ids, lens=row_len))

        self._ensure_capacity(next_block)
        for row, row_ref in enumerate(row_refs):
            positions = torch.arange(row_ref.lens, dtype=torch.long, device=key.device)
            slots = compute_slot_mapping(row_ref.block_ids, positions, self.page_size).to(device=key.device)
            self._k_pool[slots] = key[row, : row_ref.lens]
            self._v_pool[slots] = value[row, : row_ref.lens]

        self._persistent_blocks = next_block
        self.stats["paged_cache_builds"] += 1
        self.stats["paged_prefix_blocks"] += total_new_blocks
        self._current_batch = HunyuanPromptKVBatch(owner=self, row_refs=row_refs)
        return self._current_batch

    def restore_batch(self, row_refs: list[HunyuanPromptKVRowRef]) -> None:
        if any(row.owner is not self for row in row_refs):
            raise ValueError("Cannot restore Hunyuan prompt KV rows from a different page pool.")
        self._current_batch = HunyuanPromptKVBatch(owner=self, row_refs=list(row_refs))

    def materialize_rows(
        self, row_refs: list[HunyuanPromptKVRowRef]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not row_refs:
            raise ValueError("Cannot materialize an empty Hunyuan prompt KV batch.")
        assert self._kv_pool is not None
        heads, head_dim = self._kv_pool.shape[3], self._kv_pool.shape[4]
        max_len = max(row.lens for row in row_refs)
        key = self._kv_pool.new_zeros(len(row_refs), max_len, heads, head_dim)
        value = self._kv_pool.new_zeros(len(row_refs), max_len, heads, head_dim)
        lens = torch.tensor([row.lens for row in row_refs], dtype=torch.long, device=self._kv_pool.device)
        for i, row_ref in enumerate(row_refs):
            blocks = torch.tensor(row_ref.block_ids, dtype=torch.long, device=self._kv_pool.device)
            row_key = self._kv_pool[0, blocks].reshape(-1, heads, head_dim)[: row_ref.lens]
            row_value = self._kv_pool[1, blocks].reshape(-1, heads, head_dim)[: row_ref.lens]
            key[i, : row_ref.lens] = row_key
            value[i, : row_ref.lens] = row_value
        return key, value, lens

    @staticmethod
    def attention_mask_is_all_keep(attention_mask: torch.Tensor | None) -> bool:
        if attention_mask is None or attention_mask.numel() == 0:
            return True
        if attention_mask.dtype == torch.bool:
            return bool(torch.all(attention_mask).item())
        if torch.is_floating_point(attention_mask):
            return bool(torch.all(attention_mask == 0).item())
        return bool(torch.all(attention_mask != 0).item())

    def _build_attention_inputs(self, key: torch.Tensor, seq_len: int) -> HunyuanPagedAttentionInputs:
        batch = self._current_batch
        if batch is None:
            raise RuntimeError("Hunyuan paged KV attention has no current prompt KV batch.")
        if key.dim() != 4:
            raise ValueError(f"current key must be 4D, got {tuple(key.shape)}")
        if key.shape[0] != len(batch.row_refs):
            raise ValueError("current key batch size does not match prompt KV rows.")

        bs, q_len = key.shape[:2]
        block_rows: list[list[int]] = []
        seq_lens: list[int] = []
        scratch_cursor = self._persistent_blocks
        current_blocks = 0
        for row_ref in batch.row_refs:
            row_seq_len = int(row_ref.lens + q_len)
            if row_seq_len > seq_len:
                raise ValueError("row sequence length exceeds dense sequence length.")
            row_page_count = _ceil_div(row_seq_len, self.page_size)
            extra_blocks = row_page_count - len(row_ref.block_ids)
            if extra_blocks < 0:
                raise AssertionError("paged metadata would drop prefix blocks.")
            scratch_blocks = list(range(scratch_cursor, scratch_cursor + extra_blocks))
            scratch_cursor += extra_blocks
            current_blocks += extra_blocks
            block_rows.append(list(row_ref.block_ids) + scratch_blocks)
            seq_lens.append(row_seq_len)

        self._ensure_capacity(scratch_cursor)

        max_pages = max(len(row) for row in block_rows)
        padded_rows = [row + [0] * (max_pages - len(row)) for row in block_rows]
        device = key.device
        block_table = torch.tensor(padded_rows, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(0, (bs + 1) * q_len, q_len, dtype=torch.int32, device=device)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        return HunyuanPagedAttentionInputs(
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_tensor,
            max_query_len=int(q_len),
            max_seq_len=max(seq_lens),
            prefix_blocks=sum(len(row.block_ids) for row in batch.row_refs),
            current_blocks=current_blocks,
        )

    def run_paged_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        seq_len: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        if key.shape != value.shape:
            raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
        self._ensure_compatible(key)
        inputs = self._build_attention_inputs(key, seq_len)
        assert self._kv_pool is not None and self._k_pool is not None and self._v_pool is not None

        batch = self._current_batch
        assert batch is not None
        q_len = key.shape[1]
        for row, row_ref in enumerate(batch.row_refs):
            positions = torch.arange(row_ref.lens, row_ref.lens + q_len, dtype=torch.long, device=key.device)
            slots = compute_slot_mapping(inputs.block_table[row].tolist(), positions, self.page_size).to(
                device=key.device
            )
            self._k_pool[slots] = key[row]
            self._v_pool[slots] = value[row]

        out = ar_diffusion_paged_attention(
            query,
            self._kv_pool[0],
            self._kv_pool[1],
            block_table=inputs.block_table,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            max_query_len=inputs.max_query_len,
            max_seq_len=inputs.max_seq_len,
            softmax_scale=softmax_scale,
            causal=False,
        )
        self.stats["paged_attention_calls"] += 1
        self.stats["paged_current_blocks"] += int(inputs.current_blocks)
        return out
