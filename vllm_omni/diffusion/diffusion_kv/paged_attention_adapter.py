# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context as set_vllm_forward_context
from vllm.model_executor.layers.attention import Attention as VllmAttention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables


@dataclass(frozen=True)
class DiffusionPagedAttentionRowBinding:
    """Worker row and logical length installed for one allocation identity."""

    row_index: int
    max_seq_len: int


DiffusionKVRowResolver = Callable[
    [str, int | None, str | None],
    DiffusionPagedAttentionRowBinding,
]


class DiffusionPagedAttentionLayerAdapter(VllmAttention):
    """A native vLLM attention layer backed by a diffusion Attention spec."""

    def __init__(
        self,
        *,
        layer_name: str,
        layer: Any,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        with set_current_vllm_config(vllm_config):
            attn_backend = get_attn_backend(
                head_size=spec.head_size,
                dtype=vllm_config.model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                num_heads=layer.num_heads,
            )
            canonical_spec = replace(
                spec,
                indexes_kv_by_block_stride=attn_backend.indexes_kv_by_block_stride(),
            )
            super().__init__(
                num_heads=layer.num_heads,
                head_size=canonical_spec.head_size,
                scale=layer.softmax_scale,
                num_kv_heads=canonical_spec.num_kv_heads,
                cache_config=vllm_config.cache_config,
                prefix=layer_name,
                attn_type=AttentionType.DECODER,
                attn_backend=attn_backend,
                head_size_v=getattr(canonical_spec, "head_size_v", canonical_spec.head_size),
            )
        self._diffusion_layer_ref = weakref.ref(layer)
        self.spec = canonical_spec
        self.to(device=device)

    @property
    def layer(self) -> Any:
        layer = self._diffusion_layer_ref()
        if layer is None:
            raise RuntimeError("The diffusion Attention layer backing this native adapter no longer exists")
        return layer

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


class DiffusionPagedAttentionAdapter:
    """Translate diffusion rows into vLLM metadata and execute native attention."""

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
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            dcp_local_seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=self.block_tables.cp_size,
                dcp_rank=self.block_tables.cp_rank,
                cp_kv_cache_interleave_size=self.block_tables.cp_interleave,
            )
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
        with set_current_vllm_config(self.vllm_config):
            attn_metadata = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=len(rows),
                num_tokens=num_tokens,
                query_start_loc_gpu=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max(query_lens),
                seq_lens=seq_lens,
                max_seq_len=max(row.seq_len for row in rows),
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=self.kv_cache_config,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                dcp_local_seq_lens=dcp_local_seq_lens,
                positions=positions,
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
        try:
            with (
                set_current_vllm_config(self.vllm_config),
                set_vllm_forward_context(
                    batch.attn_metadata,
                    self.vllm_config,
                    num_tokens=batch.num_tokens,
                    slot_mapping=batch.slot_mappings_by_layer,
                ),
            ):
                yield self
        finally:
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
    def _validate_token_layout(
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

    def forward(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self._active_batch is None:
            raise RuntimeError("Paged attention forward must run inside adapter.activate(batch)")
        try:
            layer = self.layers[layer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown diffusion paged attention layer {layer_name!r}") from exc
        batch = self._active_batch

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
        if key_token_shape != query_token_shape or value_token_shape != query_token_shape:
            raise ValueError(
                "Paged attention Q/K/V token layouts must match; "
                f"got query={query_token_shape}, key={key_token_shape}, value={value_token_shape}"
            )
        self._validate_token_layout(query_token_shape, batch)
        token_counts = (query_flat.shape[0], key_flat.shape[0], value_flat.shape[0])
        if token_counts != (batch.num_tokens, batch.num_tokens, batch.num_tokens):
            raise ValueError(
                "Paged attention Q/K/V token count must match the prepared batch: "
                f"qkv={token_counts}, prepared={batch.num_tokens}"
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

        output = layer(query_flat, key_flat, value_flat)
        if query_has_head_dims:
            return output.reshape(*query_token_shape, layer.num_heads, layer.head_size_v)
        return output.reshape(*query_token_shape, layer.num_heads * layer.head_size_v)
