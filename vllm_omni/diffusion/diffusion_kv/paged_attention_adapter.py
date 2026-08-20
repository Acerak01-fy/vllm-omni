# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import Segment, build_segments
from vllm_omni.diffusion.forward_context import override_paged_kv_adapter


@dataclass(frozen=True)
class DiffusionPagedAttentionRowBinding:
    """Worker row and logical length installed for one allocation identity."""

    row_index: int
    max_seq_len: int


DiffusionKVRowResolver = Callable[
    [str, int | None, str | None],
    DiffusionPagedAttentionRowBinding,
]


class DiffusionPagedAttentionLayerAdapter(AttentionLayerBase):
    """Register a diffusion layer with vLLM's native cache machinery.

    This object deliberately does *not* subclass ``vllm.Attention``.  The
    latter owns a second execution path and would bypass Omni's sequence
    parallel pre/post hooks.  The wrapper only supplies the small
    ``AttentionLayerBase`` contract needed by ``init_attn_backend`` and keeps
    the native FlashAttention implementation/cache view available to the
    diffusion adapter.
    """

    def __init__(
        self,
        *,
        layer_name: str,
        layer: Any,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
        ulysses_degree: int = 1,
    ) -> None:
        if type(ulysses_degree) is not int or ulysses_degree <= 0:
            raise ValueError(f"ulysses_degree must be a positive integer, got {ulysses_degree!r}")
        num_heads = int(layer.num_heads)
        num_kv_heads = int(spec.num_kv_heads)
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "Paged attention requires positive Q/KV heads with num_heads divisible by num_kv_heads: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}"
            )
        if num_heads % ulysses_degree != 0 or num_kv_heads % ulysses_degree != 0:
            raise ValueError(
                "Paged attention requires Q/KV heads divisible by ulysses_degree: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, ulysses_degree={ulysses_degree}"
            )
        num_heads //= ulysses_degree
        num_kv_heads //= ulysses_degree

        attention_config = vllm_config.attention_config
        previous_backend = attention_config.backend
        previous_backend_per_kind = attention_config.backend_per_kind
        try:
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN
            attention_config.backend_per_kind = {}
            with set_current_vllm_config(vllm_config):
                attn_backend = get_attn_backend(
                    head_size=spec.head_size,
                    dtype=vllm_config.model_config.dtype,
                    kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                    attn_type=AttentionType.DECODER,
                    num_heads=num_heads,
                )
        finally:
            attention_config.backend = previous_backend
            attention_config.backend_per_kind = previous_backend_per_kind
        if attn_backend.get_name() != "FLASH_ATTN":
            raise RuntimeError(
                f"Diffusion paged attention requires vLLM's FLASH_ATTN backend; selected {attn_backend.get_name()!r}"
            )
        canonical_spec = replace(
            spec,
            num_kv_heads=num_kv_heads,
            indexes_kv_by_block_stride=attn_backend.indexes_kv_by_block_stride(),
        )
        self.layer_name = layer_name
        self.spec = canonical_spec
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = int(canonical_spec.head_size)
        self.head_size_v = int(getattr(canonical_spec, "head_size_v", canonical_spec.head_size))
        if self.head_size_v != self.head_size:
            raise ValueError(
                "Diffusion paged FlashAttention requires equal key/value head sizes; "
                f"got head_size={self.head_size}, head_size_v={self.head_size_v}"
            )
        self.softmax_scale = float(layer.softmax_scale)
        self.attn_backend = attn_backend
        self.kv_cache: torch.Tensor | None = None
        # Native FlashAttention uses these fields for KV cache quantization.
        # They are harmless for the normal (unquantized) path and avoid making
        # the adapter depend on the concrete vLLM Attention module.
        self._q_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._k_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._v_scale = torch.ones(1, device=device, dtype=torch.float32)
        with set_current_vllm_config(vllm_config):
            self.impl = self._create_native_impl(vllm_config)

    def get_attn_backend(self):
        return self.attn_backend

    def _create_native_impl(self, vllm_config: VllmConfig):
        impl_cls = self.attn_backend.get_impl_cls()
        impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        if getattr(impl, "vllm_flash_attn_version", None) is None:
            raise RuntimeError(f"FlashAttention paged-KV kernel is unavailable for diffusion layer {self.layer_name!r}")
        return impl

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> AttentionSpec:
        del vllm_config
        return self.spec


@dataclass(frozen=True)
class DiffusionPagedAttentionRow:
    """One logical BlockTable row participating in a paged attention call."""

    request_id: str
    query_len: int
    seq_len: int
    kv_start_pos: int = 0
    sequence_id: int | None = None
    context_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("Paged attention request_id must be a non-empty string")
        if (self.sequence_id is None) == (self.context_id is None):
            raise ValueError("Paged attention row requires exactly one of sequence_id or context_id")
        if self.sequence_id is not None and (type(self.sequence_id) is not int or self.sequence_id < 0):
            raise ValueError("Paged attention sequence_id must be a non-negative integer")
        if self.context_id is not None and (type(self.context_id) is not str or not self.context_id):
            raise ValueError("Paged attention context_id must be a non-empty string")
        if type(self.query_len) is not int or self.query_len <= 0:
            raise ValueError("Paged attention query_len must be a positive integer")
        if type(self.seq_len) is not int or self.seq_len <= 0:
            raise ValueError("Paged attention seq_len must be a positive integer")
        if type(self.kv_start_pos) is not int or self.kv_start_pos < 0:
            raise ValueError("Paged attention kv_start_pos must be a non-negative integer")
        if self.kv_start_pos + self.query_len > self.seq_len:
            raise ValueError(
                "Paged attention write span exceeds seq_len: "
                f"start={self.kv_start_pos}, query_len={self.query_len}, seq_len={self.seq_len}"
            )

    @property
    def identity(self) -> tuple[str, int | None, str | None]:
        return (self.request_id, self.sequence_id, self.context_id)


@dataclass(frozen=True)
class PreparedDiffusionPagedAttentionBatch:
    """Native metadata shared by all paged attention layers in one forward."""

    rows: tuple[DiffusionPagedAttentionRow, ...]
    row_indices: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    attn_metadata: dict[str, Any]
    slot_mappings_by_layer: dict[str, torch.Tensor]
    num_tokens: int
    _owner: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DiffusionPagedAttentionSegmentContext:
    """One piecewise query selection and its native attention metadata."""

    query_indices: torch.Tensor
    native_metadata: Any


@dataclass(frozen=True, slots=True)
class DiffusionPagedAttentionContext:
    """One layer's native inputs for an Omni paged-backend invocation."""

    layer: DiffusionPagedAttentionLayerAdapter
    query: torch.Tensor
    key_write: torch.Tensor
    value_write: torch.Tensor
    slot_mapping: torch.Tensor
    native_metadata: Any
    piecewise_segments: tuple[DiffusionPagedAttentionSegmentContext, ...]
    query_token_shape: tuple[int, ...]
    query_has_head_dims: bool

    def restore_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.query_has_head_dims:
            return output.reshape(*self.query_token_shape, self.layer.num_heads, self.layer.head_size_v)
        return output.reshape(*self.query_token_shape, self.layer.num_heads * self.layer.head_size_v)


@dataclass(frozen=True, slots=True)
class _PreparedDiffusionPiecewiseSegment:
    query_indices: torch.Tensor
    attn_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedDiffusionPiecewisePlan:
    spans: tuple[tuple[tuple[int, int], ...], ...]
    segments: tuple[_PreparedDiffusionPiecewiseSegment, ...]


class DiffusionPagedAttentionAdapter:
    """Translate diffusion rows into vLLM metadata for an Omni paged backend."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        device: torch.device,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        attn_groups: list[list[Any]],
        layers: Mapping[str, DiffusionPagedAttentionLayerAdapter],
        resolve_row: DiffusionKVRowResolver,
    ) -> None:
        if not layers:
            raise ValueError("Paged attention requires at least one native attention layer")
        if len(attn_groups) != len(kv_cache_config.kv_cache_groups):
            raise ValueError(
                "Paged attention group mismatch: "
                f"builders={len(attn_groups)}, cache_groups={len(kv_cache_config.kv_cache_groups)}"
            )
        self.vllm_config = vllm_config
        self.device = torch.device(device)
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = attn_groups
        self.layers = dict(layers)
        self.resolve_row = resolve_row
        self._owner = object()
        self._prepare_generation = 0
        self._active_batch: PreparedDiffusionPagedAttentionBatch | None = None
        self._active_piecewise_plan: _PreparedDiffusionPiecewisePlan | None = None
        self._causal_by_group = self._resolve_group_causality()
        self._reorder_batch_threshold = self._resolve_reorder_batch_threshold()

    def _resolve_group_causality(self) -> dict[int, bool]:
        causal_by_group: dict[int, bool] = {}
        for group_index, cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_causality = {
                not bool(getattr(self.layers[layer_name].spec, "non_causal", False))
                for layer_name in cache_group.layer_names
            }
            if len(layer_causality) != 1:
                raise ValueError(f"Paged attention cache group {group_index} mixes causal and non-causal layers")
            causal_by_group[group_index] = layer_causality.pop()
        return causal_by_group

    def _resolve_reorder_batch_threshold(self) -> int | None:
        thresholds = [
            group.get_metadata_builder(0).reorder_batch_threshold
            for group_index, groups in enumerate(self.attn_groups)
            if self._causal_by_group[group_index]
            for group in groups
        ]
        concrete_thresholds = [threshold for threshold in thresholds if threshold is not None]
        return min(concrete_thresholds, default=None)

    def _validate_native_batch_order(self, rows: tuple[DiffusionPagedAttentionRow, ...]) -> None:
        threshold = self._reorder_batch_threshold
        if threshold is None or not any(self._causal_by_group.values()):
            return

        found_long_query = False
        for row in rows:
            if row.query_len > threshold:
                found_long_query = True
            elif found_long_query:
                raise ValueError(
                    "Causal paged attention rows must place native decode/short-query rows before "
                    f"prefill/long-query rows (decode threshold={threshold})"
                )

    def _validate_row_capacity(
        self,
        rows: tuple[DiffusionPagedAttentionRow, ...],
        row_indices: list[int],
    ) -> None:
        native_num_blocks = self.block_tables.num_blocks.np
        blocks_per_kv_block = self.block_tables.blocks_per_kv_block
        for row, row_index in zip(rows, row_indices, strict=True):
            if type(row_index) is not int or not 0 <= row_index < self.block_tables.max_num_reqs:
                raise ValueError(f"Paged attention row {row.identity!r} resolved to invalid Worker row {row_index!r}")
            for group_index, (block_size, block_multiplier) in enumerate(
                zip(
                    self.block_tables.block_sizes,
                    blocks_per_kv_block,
                    strict=True,
                )
            ):
                required_blocks = cdiv(
                    row.seq_len,
                    block_size * self.block_tables.cp_size,
                )
                required_kernel_blocks = required_blocks * block_multiplier
                installed_kernel_blocks = int(native_num_blocks[group_index, row_index])
                if required_kernel_blocks > installed_kernel_blocks:
                    raise ValueError(
                        f"Paged attention row {row.identity!r} requires {required_blocks} blocks in "
                        f"cache group {group_index} for seq_len={row.seq_len}, but its installed Worker "
                        f"row contains only {installed_kernel_blocks // block_multiplier} blocks"
                    )

    def _build_native_metadata(
        self,
        *,
        query_lens: Sequence[int],
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        positions: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
        slot_mappings: torch.Tensor,
        causal: bool | Mapping[int, bool],
    ) -> dict[str, Any]:
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            dcp_local_seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=self.block_tables.cp_size,
                dcp_rank=self.block_tables.cp_rank,
                cp_kv_cache_interleave_size=self.block_tables.cp_interleave,
            )
        with set_current_vllm_config(self.vllm_config):
            return build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=len(query_lens),
                num_tokens=sum(query_lens),
                query_start_loc_gpu=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max(query_lens),
                seq_lens=seq_lens,
                max_seq_len=max(int(seq_len) for seq_len in seq_lens_cpu),
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=self.kv_cache_config,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                dcp_local_seq_lens=dcp_local_seq_lens,
                positions=positions,
                causal=causal,
            )

    def prepare_batch(
        self,
        rows: Sequence[DiffusionPagedAttentionRow],
    ) -> PreparedDiffusionPagedAttentionBatch:
        rows = tuple(rows)
        if not rows:
            raise ValueError("Paged attention batch must contain at least one row")
        if self._active_batch is not None:
            raise RuntimeError("Cannot prepare paged attention metadata during an active forward")
        if len(rows) > self.block_tables.max_num_reqs:
            raise ValueError(
                f"Paged attention batch has {len(rows)} rows; Worker capacity is {self.block_tables.max_num_reqs}"
            )

        identities = [row.identity for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("Paged attention batch contains duplicate row identities")
        row_bindings = [self.resolve_row(row.request_id, row.sequence_id, row.context_id) for row in rows]
        row_indices_list = [binding.row_index for binding in row_bindings]
        if len(set(row_indices_list)) != len(row_indices_list):
            raise ValueError("Paged attention batch resolves multiple inputs to the same Worker row")
        for row, binding in zip(rows, row_bindings, strict=True):
            if type(binding.max_seq_len) is not int or binding.max_seq_len < 0:
                raise ValueError(
                    f"Paged attention row {row.identity!r} resolved to invalid logical capacity {binding.max_seq_len!r}"
                )
            if row.seq_len > binding.max_seq_len:
                raise ValueError(
                    f"Paged attention row {row.identity!r} uses seq_len={row.seq_len}, but its installed "
                    f"allocation has logical length {binding.max_seq_len}"
                )
        self._validate_native_batch_order(rows)
        self._validate_row_capacity(rows, row_indices_list)

        query_lens = [row.query_len for row in rows]
        num_tokens = sum(query_lens)
        if num_tokens > self.block_tables.max_num_batched_tokens:
            raise ValueError(
                f"Paged attention batch has {num_tokens} tokens; Worker capacity is "
                f"{self.block_tables.max_num_batched_tokens}"
            )
        query_offsets = [0]
        for query_len in query_lens:
            query_offsets.append(query_offsets[-1] + query_len)

        # Native BlockTables reuses persistent gather/slot buffers. Starting a
        # new preparation invalidates every batch prepared before it, even if a
        # later metadata builder raises.
        self._prepare_generation += 1
        generation = self._prepare_generation
        row_indices = torch.tensor(row_indices_list, dtype=torch.int32, device=self.device)
        query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        seq_lens_cpu = torch.tensor([row.seq_len for row in rows], dtype=torch.int32)
        seq_lens = seq_lens_cpu.to(self.device)
        positions = torch.cat(
            [
                torch.arange(
                    row.kv_start_pos,
                    row.kv_start_pos + row.query_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                for row in rows
            ]
        )

        block_tables = self.block_tables.gather_block_tables(
            row_indices,
            num_reqs_padded=len(rows),
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            row_indices,
            query_start_loc,
            positions,
            num_tokens_padded=num_tokens,
        )
        causal: bool | Mapping[int, bool]
        causal_values = set(self._causal_by_group.values())
        causal = causal_values.pop() if len(causal_values) == 1 else self._causal_by_group
        attn_metadata = self._build_native_metadata(
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            causal=causal,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings,
            self.kv_cache_config,
        )
        return PreparedDiffusionPagedAttentionBatch(
            rows=rows,
            row_indices=row_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            positions=positions,
            block_tables=tuple(block_tables),
            slot_mappings=slot_mappings,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            num_tokens=num_tokens,
            _owner=self._owner,
            _generation=generation,
        )

    def invalidate_prepared_batches(self) -> None:
        """Invalidate native buffer views after BlockTable state changes."""

        if self._active_batch is not None:
            raise RuntimeError("Cannot change paged attention BlockTables during an active forward")
        self._prepare_generation += 1

    @contextmanager
    def activate(
        self,
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> Iterator[DiffusionPagedAttentionAdapter]:
        if batch._owner is not self._owner:
            raise ValueError("Prepared paged attention batch belongs to a different adapter")
        if batch._generation != self._prepare_generation:
            raise ValueError("Prepared paged attention batch is stale after a newer batch preparation")
        if self._active_batch is not None:
            raise RuntimeError("Paged attention adapter already has an active forward")
        self._active_batch = batch
        self._active_piecewise_plan = None
        try:
            with override_paged_kv_adapter(self), set_current_vllm_config(self.vllm_config):
                yield self
        finally:
            self._active_piecewise_plan = None
            self._active_batch = None

    @staticmethod
    def _flatten_tensor(
        tensor: torch.Tensor,
        *,
        num_heads: int,
        head_size: int,
        name: str,
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if tensor.ndim >= 2 and tensor.shape[-2:] == (num_heads, head_size):
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-2]), True
        hidden_size = num_heads * head_size
        if tensor.ndim >= 1 and tensor.shape[-1] == hidden_size:
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-1]), False
        raise ValueError(
            f"Paged attention {name} must end in ({num_heads}, {head_size}) or ({hidden_size},); "
            f"got shape={tuple(tensor.shape)}"
        )

    @staticmethod
    def _validate_query_layout(
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> None:
        if len(token_shape) == 1:
            return
        if len(token_shape) == 2:
            batch_size, tokens_per_row = token_shape
            query_lens = tuple(row.query_len for row in batch.rows)
            if batch_size == len(batch.rows) and all(query_len == tokens_per_row for query_len in query_lens):
                return
            raise ValueError(
                "Paged attention batched Q/K/V layout must match prepared rows: "
                f"shape={token_shape}, row_query_lens={query_lens}"
            )
        raise ValueError(
            "Paged attention supports packed [T, ...] or uniform batched [B, T, ...] token layouts; "
            f"got token shape={token_shape}"
        )

    @staticmethod
    def _extract_write_tokens(
        tensor: torch.Tensor,
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
        *,
        name: str,
    ) -> torch.Tensor:
        """Return the K/V tokens covered by the current page-table write.

        A diffusion step commonly passes ``K/V=[prefix | current image]``
        while the query contains only ``current image``.  The native cache
        writer must receive only the write span, and that span may begin in
        the middle of a physical block.  This helper accepts both packed and
        per-row full-sequence layouts and slices with the logical
        ``kv_start_pos`` from the prepared rows.
        """

        query_tokens = batch.num_tokens
        if tensor.shape[0] == query_tokens:
            if len(token_shape) == 2:
                batch_size, tokens_per_row = token_shape
                if batch_size != len(batch.rows) or any(row.query_len != tokens_per_row for row in batch.rows):
                    raise ValueError(
                        f"Paged attention {name} query-aligned layout does not match prepared rows: "
                        f"shape={token_shape}, row_query_lens={tuple(row.query_len for row in batch.rows)}"
                    )
            return tensor

        full_tokens = sum(row.seq_len for row in batch.rows)
        if tensor.shape[0] != full_tokens:
            raise ValueError(
                f"Paged attention {name} must contain either {query_tokens} write tokens or "
                f"{full_tokens} full-sequence tokens; got {tensor.shape[0]}"
            )

        pieces: list[torch.Tensor] = []
        if len(token_shape) == 2:
            batch_size, tokens_per_row = token_shape
            if batch_size != len(batch.rows):
                raise ValueError(
                    f"Paged attention {name} batch dimension {batch_size} does not match rows {len(batch.rows)}"
                )
            row_seq_lens = tuple(row.seq_len for row in batch.rows)
            if any(seq_len != tokens_per_row for seq_len in row_seq_lens):
                raise ValueError(
                    f"Paged attention {name} full-sequence layout does not match prepared rows: "
                    f"shape={token_shape}, row_seq_lens={row_seq_lens}"
                )
            row_offset = 0
            for row in batch.rows:
                if tokens_per_row < row.kv_start_pos + row.query_len:
                    raise ValueError(
                        f"Paged attention {name} row {row.identity!r} has only {tokens_per_row} tokens, "
                        f"but write span ends at {row.kv_start_pos + row.query_len}"
                    )
                pieces.append(tensor[row_offset + row.kv_start_pos : row_offset + row.kv_start_pos + row.query_len])
                row_offset += tokens_per_row
        elif len(token_shape) == 1:
            row_offset = 0
            for row in batch.rows:
                pieces.append(tensor[row_offset + row.kv_start_pos : row_offset + row.kv_start_pos + row.query_len])
                row_offset += row.seq_len
        else:
            raise ValueError(
                f"Paged attention {name} supports packed [T, ...] or batched [B, S, ...] layouts; "
                f"got token shape={token_shape}"
            )
        result = torch.cat(pieces, dim=0)
        if result.shape[0] != query_tokens:
            raise RuntimeError(
                f"Paged attention {name} write extraction produced {result.shape[0]} tokens; expected {query_tokens}"
            )
        return result

    @staticmethod
    def _normalize_piecewise_spans(
        full_attn_spans: list[list[tuple[int, int]]],
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Validate model metadata and return an immutable cache key."""

        if len(full_attn_spans) != len(batch.rows):
            raise ValueError(
                "Paged piecewise attention requires one full_attn_spans entry per prepared row: "
                f"spans={len(full_attn_spans)}, rows={len(batch.rows)}"
            )

        normalized_rows: list[tuple[tuple[int, int], ...]] = []
        for row, row_spans in zip(batch.rows, full_attn_spans, strict=True):
            if not isinstance(row_spans, list):
                raise TypeError(f"Paged piecewise spans for row {row.identity!r} must be a list")
            normalized_row: list[tuple[int, int]] = []
            previous_end = 0
            for span_index, span in enumerate(row_spans):
                if not isinstance(span, (tuple, list)) or len(span) != 2:
                    raise TypeError(
                        f"Paged piecewise span {span_index} for row {row.identity!r} must be a (start, end) pair"
                    )
                start, end = span
                if type(start) is not int or type(end) is not int or start < 0 or start >= end:
                    raise ValueError(
                        f"Paged piecewise span {span!r} for row {row.identity!r} must satisfy 0 <= start < end"
                    )
                if start < previous_end:
                    raise ValueError(
                        f"Paged piecewise spans for row {row.identity!r} must be sorted and non-overlapping"
                    )
                if end > row.seq_len:
                    raise ValueError(
                        f"Paged piecewise span {span!r} exceeds row {row.identity!r} seq_len={row.seq_len}"
                    )
                normalized_row.append((start, end))
                previous_end = end
            normalized_rows.append(tuple(normalized_row))

        normalized_spans = tuple(normalized_rows)
        reference_spans = normalized_spans[0]
        for row_index, row_spans in enumerate(normalized_spans[1:], start=1):
            if row_spans != reference_spans:
                raise ValueError(
                    "Paged piecewise attention requires homogeneous batch spans: "
                    f"row 0 has {reference_spans!r}, row {row_index} has {row_spans!r}"
                )
        return normalized_spans

    @staticmethod
    def _build_piecewise_row_segments(
        spans: tuple[tuple[tuple[int, int], ...], ...],
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> tuple[tuple[Segment, ...], ...]:
        """Partition every row's current query into causal/full segments."""

        non_suffix_rows = [row.identity for row in batch.rows if row.kv_start_pos + row.query_len != row.seq_len]
        if non_suffix_rows:
            raise ValueError(
                "Paged piecewise attention requires each query/write span to end at seq_len; "
                f"invalid rows={non_suffix_rows!r}"
            )

        segments_by_row = tuple(
            tuple(build_segments(row_spans, row.kv_start_pos, row.query_len))
            for row, row_spans in zip(batch.rows, spans, strict=True)
        )
        segment_count = len(segments_by_row[0])
        if any(len(row_segments) != segment_count for row_segments in segments_by_row[1:]):
            raise ValueError("Paged piecewise attention requires rows to produce the same segment count")
        return segments_by_row

    def _prepare_piecewise_segment(
        self,
        batch: PreparedDiffusionPagedAttentionBatch,
        row_query_offsets: Sequence[int],
        row_segments: Sequence[Segment],
    ) -> _PreparedDiffusionPiecewiseSegment:
        """Build one native batch from the corresponding segment of each row."""

        segment_mode = row_segments[0].mode
        if any(segment.mode != segment_mode for segment in row_segments[1:]):
            raise ValueError("Paged piecewise attention requires segment causality to match across rows")

        query_lens: list[int] = []
        seq_lens_values: list[int] = []
        query_indices_values: list[int] = []
        query_offsets = [0]
        for row_index, (row, segment) in enumerate(zip(batch.rows, row_segments, strict=True)):
            local_start = segment.q_start - row.kv_start_pos
            query_len = segment.q_end - segment.q_start
            query_lens.append(query_len)
            seq_lens_values.append(segment.kv_end)
            query_offsets.append(query_offsets[-1] + query_len)
            packed_start = row_query_offsets[row_index] + local_start
            query_indices_values.extend(range(packed_start, packed_start + query_len))

        query_indices = torch.tensor(query_indices_values, dtype=torch.long, device=self.device)
        query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
        seq_lens_cpu = torch.tensor(seq_lens_values, dtype=torch.int32)
        attn_metadata = self._build_native_metadata(
            query_lens=query_lens,
            query_start_loc=query_start_loc_cpu.to(self.device),
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_cpu.to(self.device),
            seq_lens_cpu=seq_lens_cpu,
            positions=batch.positions.index_select(0, query_indices),
            block_tables=batch.block_tables,
            slot_mappings=batch.slot_mappings.index_select(-1, query_indices),
            causal=(segment_mode == "causal"),
        )
        return _PreparedDiffusionPiecewiseSegment(
            query_indices=query_indices,
            attn_metadata=attn_metadata,
        )

    def _get_piecewise_plan(
        self,
        full_attn_spans: list[list[tuple[int, int]]],
    ) -> _PreparedDiffusionPiecewisePlan:
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("Piecewise paged attention requires an active prepared batch")
        normalized_spans = self._normalize_piecewise_spans(full_attn_spans, batch)

        cached_plan = self._active_piecewise_plan
        if cached_plan is not None:
            if cached_plan.spans != normalized_spans:
                raise ValueError("Paged piecewise attention metadata changed between layers in one active batch")
            return cached_plan

        segments_by_row = self._build_piecewise_row_segments(normalized_spans, batch)
        row_query_offsets = [0]
        for row in batch.rows:
            row_query_offsets.append(row_query_offsets[-1] + row.query_len)
        plan = _PreparedDiffusionPiecewisePlan(
            spans=normalized_spans,
            segments=tuple(
                self._prepare_piecewise_segment(batch, row_query_offsets, row_segments)
                for row_segments in zip(*segments_by_row, strict=True)
            ),
        )
        self._active_piecewise_plan = plan
        return plan

    def prepare_layer_context(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        omni_attn_metadata: Any | None = None,
    ) -> DiffusionPagedAttentionContext:
        if self._active_batch is None:
            raise RuntimeError("Paged attention forward must run inside adapter.activate(batch)")
        try:
            layer = self.layers[layer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown diffusion paged attention layer {layer_name!r}") from exc
        batch = self._active_batch
        full_attn_spans = self._validate_omni_attn_metadata(omni_attn_metadata)

        query_flat, query_token_shape, query_has_head_dims = self._flatten_tensor(
            query,
            num_heads=layer.num_heads,
            head_size=layer.head_size,
            name="query",
        )
        key_flat, key_token_shape, _ = self._flatten_tensor(
            key,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            name="key",
        )
        value_flat, value_token_shape, _ = self._flatten_tensor(
            value,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size_v,
            name="value",
        )
        self._validate_query_layout(query_token_shape, batch)
        if query_flat.shape[0] != batch.num_tokens:
            raise ValueError(
                "Paged attention query token count must match the prepared batch: "
                f"query={query_flat.shape[0]}, prepared={batch.num_tokens}"
            )
        if query.device != self.device or key.device != self.device or value.device != self.device:
            raise ValueError(
                f"Paged attention Q/K/V must be on {self.device}; "
                f"got query={query.device}, key={key.device}, value={value.device}"
            )
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(f"Paged attention Q/K/V dtypes must match; got {query.dtype}, {key.dtype}, {value.dtype}")
        expected_dtype = self.vllm_config.model_config.dtype
        if query.dtype != expected_dtype:
            raise ValueError(
                f"Paged attention Q/K/V dtype must match model activation dtype {expected_dtype}; got {query.dtype}"
            )
        if not bool(getattr(layer.spec, "non_causal", False)):
            non_suffix_rows = [row.identity for row in batch.rows if row.kv_start_pos + row.query_len != row.seq_len]
            if non_suffix_rows:
                raise ValueError(
                    "Causal paged attention requires each query/write span to end at seq_len; "
                    f"invalid rows={non_suffix_rows!r}"
                )

        key_write = self._extract_write_tokens(key_flat, key_token_shape, batch, name="key")
        value_write = self._extract_write_tokens(value_flat, value_token_shape, batch, name="value")
        if key_write.shape[0] != value_write.shape[0]:
            raise ValueError(
                f"Paged attention key/value write lengths differ: {key_write.shape[0]} vs {value_write.shape[0]}"
            )
        slot_mapping = batch.slot_mappings_by_layer.get(layer_name)
        if slot_mapping is None:
            raise KeyError(f"No native slot mapping was built for diffusion layer {layer_name!r}")
        slot_mapping = slot_mapping.reshape(-1)[: batch.num_tokens]
        if slot_mapping.numel() != batch.num_tokens:
            raise ValueError(
                f"Paged attention slot mapping for {layer_name!r} has {slot_mapping.numel()} entries; "
                f"expected {batch.num_tokens}"
            )

        # vLLM's writer consumes slot_mapping directly, so a span that starts
        # inside a physical block naturally writes only the requested suffix.
        key_write = key_write.contiguous()
        value_write = value_write.contiguous()
        try:
            native_metadata = batch.attn_metadata[layer_name]
        except KeyError as exc:
            raise KeyError(f"No native attention metadata was built for diffusion layer {layer_name!r}") from exc
        piecewise_segments: tuple[DiffusionPagedAttentionSegmentContext, ...] = ()
        if full_attn_spans is not None:
            plan = self._get_piecewise_plan(full_attn_spans)
            try:
                piecewise_segments = tuple(
                    DiffusionPagedAttentionSegmentContext(
                        query_indices=segment.query_indices,
                        native_metadata=segment.attn_metadata[layer_name],
                    )
                    for segment in plan.segments
                )
            except KeyError as exc:
                raise KeyError(
                    f"No piecewise native attention metadata was built for diffusion layer {layer_name!r}"
                ) from exc
        return DiffusionPagedAttentionContext(
            layer=layer,
            query=query_flat.contiguous(),
            key_write=key_write,
            value_write=value_write,
            slot_mapping=slot_mapping,
            native_metadata=native_metadata,
            piecewise_segments=piecewise_segments,
            query_token_shape=query_token_shape,
            query_has_head_dims=query_has_head_dims,
        )

    @staticmethod
    def _validate_omni_attn_metadata(
        metadata: Any | None,
    ) -> list[list[tuple[int, int]]] | None:
        if metadata is None:
            return None
        full_attn_spans = getattr(metadata, "full_attn_spans", None)
        attn_mask = getattr(metadata, "attn_mask", None)
        unsupported_fields = [
            field_name
            for field_name in (
                "joint_attn_mask",
                "query_ranges",
                "video_layout",
            )
            if getattr(metadata, field_name, None) is not None
        ]
        if attn_mask is not None:
            if full_attn_spans is None:
                unsupported_fields.append("attn_mask")
            elif not isinstance(attn_mask, torch.Tensor) or attn_mask.ndim != 4:
                unsupported_fields.append("attn_mask (piecewise paging requires a 4D tensor)")
        extra = getattr(metadata, "extra", None)
        if extra:
            unsupported_fields.append(f"extra={sorted(extra)}")
        if unsupported_fields:
            raise NotImplementedError(
                "Diffusion paged attention cannot translate Omni attention metadata fields "
                f"{unsupported_fields!r} to native FlashAttention metadata"
            )
        return full_attn_spans
