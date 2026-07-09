# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image3 prompt/ref KV cache backed by AR-Diffusion paging primitives."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    allocate_kv_pool_with_views,
    compute_slot_mapping,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ar_diffusion_paged_attention,
)
from vllm_omni.profiler import profile_range

_HY3_PAGED_KV_CACHE_ENV = "VLLM_OMNI_HY3_PAGED_KV_CACHE"
_HY3_PAGED_KV_PAGE_SIZE_ENV = "VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE"
_HY3_PAGED_KV_PROFILE_ENV = "VLLM_OMNI_HY3_PAGED_KV_PROFILE"
_HY3_PAGED_KV_WORKSPACE_BYTES_ENV = "VLLM_OMNI_HY3_PAGED_KV_CACHE_WORKSPACE_BYTES"
_HY3_PAGED_KV_DEFAULT_PAGE_SIZE = 16
_HY3_PAGED_KV_DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "enabled", "enable", "required"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_RESHAPE_AND_CACHE_FLASH: Any | None = None
_RESHAPE_AND_CACHE_FLASH_LOAD_ERROR: Exception | None = None
_FLASHINFER_SEGMENT_PACKBITS: Any | None = None
_FLASHINFER_SEGMENT_PACKBITS_LOAD_ERROR: Exception | None = None
_PACKED_CUSTOM_MASK_CACHE_MAX_ENTRIES = 4
_MASK_NONE_OR_EMPTY = "none_or_empty"
_MASK_INPUT_ALL_KEEP = "input_all_keep"
_MASK_EFFECTIVE_ALL_KEEP = "effective_all_keep"
_MASK_CUSTOM = "custom"


def _load_reshape_and_cache_flash() -> Any:
    global _RESHAPE_AND_CACHE_FLASH, _RESHAPE_AND_CACHE_FLASH_LOAD_ERROR
    if _RESHAPE_AND_CACHE_FLASH is not None:
        return _RESHAPE_AND_CACHE_FLASH
    if _RESHAPE_AND_CACHE_FLASH_LOAD_ERROR is not None:
        raise RuntimeError(
            "vLLM reshape_and_cache_flash is unavailable for Hunyuan Image3 paged KV writes."
        ) from _RESHAPE_AND_CACHE_FLASH_LOAD_ERROR
    try:
        from vllm import _custom_ops as vllm_custom_ops
    except Exception as exc:
        _RESHAPE_AND_CACHE_FLASH_LOAD_ERROR = exc
        raise RuntimeError("vLLM reshape_and_cache_flash is unavailable for Hunyuan Image3 paged KV writes.") from exc
    _RESHAPE_AND_CACHE_FLASH = vllm_custom_ops.reshape_and_cache_flash
    return _RESHAPE_AND_CACHE_FLASH


def _load_flashinfer_segment_packbits() -> Any:
    global _FLASHINFER_SEGMENT_PACKBITS, _FLASHINFER_SEGMENT_PACKBITS_LOAD_ERROR
    if _FLASHINFER_SEGMENT_PACKBITS is not None:
        return _FLASHINFER_SEGMENT_PACKBITS
    if _FLASHINFER_SEGMENT_PACKBITS_LOAD_ERROR is not None:
        raise RuntimeError(
            "FlashInfer segment_packbits is unavailable for Hunyuan Image3 custom masks."
        ) from _FLASHINFER_SEGMENT_PACKBITS_LOAD_ERROR
    try:
        from flashinfer.quantization import segment_packbits
    except Exception as exc:
        _FLASHINFER_SEGMENT_PACKBITS_LOAD_ERROR = exc
        raise RuntimeError("FlashInfer segment_packbits is unavailable for Hunyuan Image3 custom masks.") from exc
    _FLASHINFER_SEGMENT_PACKBITS = segment_packbits
    return _FLASHINFER_SEGMENT_PACKBITS


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


def is_hunyuan_image3_paged_kv_profile_enabled() -> bool:
    value = os.environ.get(_HY3_PAGED_KV_PROFILE_ENV)
    if value is None:
        return False
    value = value.strip().lower()
    if value in _DISABLED_VALUES:
        return False
    return value == "" or value in _ENABLED_VALUES


def hunyuan_image3_paged_kv_page_size() -> int:
    return _parse_positive_int_env(_HY3_PAGED_KV_PAGE_SIZE_ENV, _HY3_PAGED_KV_DEFAULT_PAGE_SIZE)


def _parse_positive_int_env(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tensor_identity_key(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    try:
        version = int(tensor._version)
    except RuntimeError:
        # Inference-mode tensors do not expose a version counter.
        version = 0
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        version,
    )


def _profile_start(tensor: torch.Tensor) -> float | None:
    if not is_hunyuan_image3_paged_kv_profile_enabled():
        return None
    if tensor.is_cuda:
        torch.accelerator.synchronize(tensor.device)
    return time.perf_counter()


def _profile_finish(
    stats: dict[str, int | float] | None,
    key: str,
    start: float | None,
    tensor: torch.Tensor,
) -> None:
    if start is None or stats is None:
        return
    if tensor.is_cuda:
        torch.accelerator.synchronize(tensor.device)
    stats[key] = float(stats.get(key, 0.0)) + (time.perf_counter() - start) * 1000.0
    stats["paged_profile_syncs"] = int(stats.get("paged_profile_syncs", 0)) + 1


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
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    custom_mask: torch.Tensor | None
    packed_custom_mask: torch.Tensor | None
    plan_cache_key: tuple[Any, ...] | None
    max_query_len: int
    max_seq_len: int
    prefix_blocks: int
    current_blocks: int
    block_rows: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _CustomMaskBuildResult:
    mask: torch.Tensor | None
    reason: str


class HunyuanFlashInferPagedKVRunner:
    """FlashInfer paged prefill runner for Hunyuan custom-mask reuse."""

    def __init__(self, workspace_bytes: int | None = None) -> None:
        self.workspace_bytes = int(
            workspace_bytes
            or _parse_positive_int_env(
                _HY3_PAGED_KV_WORKSPACE_BYTES_ENV,
                _HY3_PAGED_KV_DEFAULT_WORKSPACE_BYTES,
            )
        )
        self._wrapper_cls: Any | None = None
        self._load_error: Exception | None = None
        self._workspace_by_device: dict[tuple[str, int | None], torch.Tensor] = {}
        self._wrapper_by_device: dict[tuple[str, int | None], Any] = {}
        self._plan_key_by_device: dict[tuple[str, int | None], tuple[Any, ...]] = {}

    def _load(self) -> bool:
        if self._wrapper_cls is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        except Exception as exc:
            self._load_error = exc
            return False
        self._wrapper_cls = BatchPrefillWithPagedKVCacheWrapper
        return True

    @staticmethod
    def _device_key(device: torch.device) -> tuple[str, int | None]:
        return device.type, device.index

    def _get_wrapper(self, device: torch.device) -> Any:
        if not self._load():
            raise ImportError("FlashInfer is unavailable for Hunyuan Image3 paged KV attention") from self._load_error
        device_key = self._device_key(device)
        wrapper = self._wrapper_by_device.get(device_key)
        if wrapper is not None:
            return wrapper
        workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device=device)
        assert self._wrapper_cls is not None
        wrapper = self._wrapper_cls(workspace, "NHD")
        self._workspace_by_device[device_key] = workspace
        self._wrapper_by_device[device_key] = wrapper
        return wrapper

    @staticmethod
    def _runtime_plan_cache_key(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        inputs: HunyuanPagedAttentionInputs,
        *,
        softmax_scale: float,
    ) -> tuple[Any, ...] | None:
        if inputs.plan_cache_key is None:
            return None
        return (
            inputs.plan_cache_key,
            str(query.dtype),
            str(key_cache.dtype),
            int(query.shape[2]),
            int(key_cache.shape[2]),
            int(query.shape[3]),
            int(key_cache.shape[1]),
            float(softmax_scale),
        )

    def run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        inputs: HunyuanPagedAttentionInputs,
        *,
        softmax_scale: float,
        profile_stats: dict[str, int | float] | None = None,
    ) -> torch.Tensor:
        wrapper = self._get_wrapper(query.device)
        device_key = self._device_key(query.device)
        plan_cache_key = self._runtime_plan_cache_key(query, key_cache, inputs, softmax_scale=softmax_scale)
        if plan_cache_key is not None and self._plan_key_by_device.get(device_key) == plan_cache_key:
            if profile_stats is not None:
                profile_stats["paged_profile_flashinfer_plan_cache_hits"] = (
                    int(profile_stats.get("paged_profile_flashinfer_plan_cache_hits", 0)) + 1
                )
        else:
            profile_start = _profile_start(query)
            with profile_range("hy3.paged_kv.flashinfer.plan"):
                wrapper.plan(
                    inputs.query_start_loc,
                    inputs.kv_indptr,
                    inputs.kv_indices,
                    inputs.kv_last_page_len,
                    query.shape[2],
                    key_cache.shape[2],
                    query.shape[3],
                    key_cache.shape[1],
                    custom_mask=None if inputs.packed_custom_mask is not None else inputs.custom_mask,
                    packed_custom_mask=inputs.packed_custom_mask,
                    causal=False,
                    sm_scale=float(softmax_scale),
                    q_data_type=query.dtype,
                    kv_data_type=key_cache.dtype,
                )
            _profile_finish(profile_stats, "paged_profile_flashinfer_plan_ms", profile_start, query)
            if plan_cache_key is not None:
                self._plan_key_by_device[device_key] = plan_cache_key
            else:
                self._plan_key_by_device.pop(device_key, None)
        profile_start = _profile_start(query)
        with profile_range("hy3.paged_kv.flashinfer.run"):
            out = wrapper.run(
                query.reshape(-1, query.shape[2], query.shape[3]).contiguous(),
                (key_cache, value_cache),
                return_lse=False,
            )
        _profile_finish(profile_stats, "paged_profile_flashinfer_run_ms", profile_start, query)
        return out.reshape(query.shape)


class HunyuanPromptKVPagePool:
    """Per-layer prompt/ref prefix page pool for Hunyuan Image3.

    The pool deliberately reuses the AR-Diffusion paged cache tensor layout and
    slot mapping helpers. Persistent blocks store first-step prompt/ref prefix
    KV. Later denoise steps reuse those block ids and reserve scratch pages for
    the current image KV of this forward.
    """

    _shared_flashinfer_runner: ClassVar[HunyuanFlashInferPagedKVRunner | None] = None
    _packed_custom_mask_cache: ClassVar[OrderedDict[tuple[Any, ...], torch.Tensor]] = OrderedDict()

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
        self._kv_cache_scale: torch.Tensor | None = None
        self._flashinfer_runner: HunyuanFlashInferPagedKVRunner | None = None
        self.stats: dict[str, int | float] = {
            "paged_cache_builds": 0,
            "paged_attention_calls": 0,
            "paged_attention_custom_mask_calls": 0,
            "paged_attention_errors": 0,
            "paged_mask_none_or_empty_skips": 0,
            "paged_mask_input_all_keep_skips": 0,
            "paged_mask_effective_all_keep_skips": 0,
            "paged_mask_custom_builds": 0,
            "paged_mask_packed_cache_hits": 0,
            "paged_mask_packed_cache_misses": 0,
            "paged_mask_packbits_calls": 0,
            "paged_prefix_blocks": 0,
            "paged_current_blocks": 0,
            "paged_profile_page_write_ms": 0.0,
            "paged_profile_mask_build_ms": 0.0,
            "paged_profile_packed_mask_build_ms": 0.0,
            "paged_profile_flashinfer_plan_ms": 0.0,
            "paged_profile_flashinfer_plan_cache_hits": 0,
            "paged_profile_flashinfer_run_ms": 0.0,
            "paged_profile_fast_attention_ms": 0.0,
            "paged_profile_syncs": 0,
        }

    @property
    def active(self) -> bool:
        return self._current_batch is not None

    @property
    def current_batch(self) -> HunyuanPromptKVBatch | None:
        return self._current_batch

    def clear_cache(self) -> None:
        self._current_batch = None
        self._persistent_blocks = 0

    def reset_stats(self, *, clear_cache: bool = False) -> None:
        for key in self.stats:
            self.stats[key] = 0
        if clear_cache:
            self.clear_cache()

    def get_stats(self) -> dict[str, int | float | bool | dict[str, int]]:
        return {
            "paged_kv_cache_enabled": self.enabled,
            "paged_kv_cache_required": self.required,
            "paged_kv_cache_active": self.active,
            "paged_kv_page_size": self.page_size,
            "paged_kv_num_blocks": self._num_blocks,
            "paged_kv_persistent_blocks": self._persistent_blocks,
            **self.stats,
        }

    def record_error(self) -> None:
        self.stats["paged_attention_errors"] += 1

    def _init_pool(self, *, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self._shape = (int(heads), int(head_dim))
        self.dtype = dtype
        self.device = device
        self._kv_cache_scale = None
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

        new_num_blocks = max(int(num_blocks), 1)
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

    def _cache_scale_tensor(self) -> torch.Tensor:
        if self.device is None:
            raise RuntimeError("Hunyuan paged KV pool device is not initialized.")
        scale = self._kv_cache_scale
        if scale is None or scale.device != self.device:
            scale = torch.ones((), dtype=torch.float32, device=self.device)
            self._kv_cache_scale = scale
        return scale

    def _write_paged_kv_slots(self, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        if key.shape != value.shape:
            raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
        if key.dim() != 3:
            raise ValueError(f"Hunyuan paged KV slot writer expects 3D key/value tensors, got {tuple(key.shape)}")
        if key.dtype != value.dtype:
            raise ValueError(f"key dtype {key.dtype} != value dtype {value.dtype}")
        if slots.dim() != 1 or slots.numel() != key.shape[0]:
            raise ValueError("slot_mapping must be 1D and match the key/value token count.")
        if slots.numel() == 0:
            return
        assert self._kv_pool is not None and self._k_pool is not None and self._v_pool is not None

        slot_mapping = slots.to(device=key.device, dtype=torch.long).contiguous()
        profile_start = _profile_start(key)
        if key.is_cuda:
            try:
                reshape_and_cache_flash = _load_reshape_and_cache_flash()
                scale = self._cache_scale_tensor()
                with profile_range("hy3.paged_kv.write_slots.flash"):
                    reshape_and_cache_flash(
                        key.contiguous(),
                        value.contiguous(),
                        self._kv_pool[0],
                        self._kv_pool[1],
                        slot_mapping,
                        "auto",
                        scale,
                        scale,
                    )
            finally:
                _profile_finish(self.stats, "paged_profile_page_write_ms", profile_start, key)
            return

        try:
            with profile_range("hy3.paged_kv.write_slots.torch"):
                self._k_pool[slot_mapping] = key.to(dtype=self._k_pool.dtype)
                self._v_pool[slot_mapping] = value.to(dtype=self._v_pool.dtype)
        finally:
            _profile_finish(self.stats, "paged_profile_page_write_ms", profile_start, key)

    def clear_current(self) -> None:
        self._current_batch = None

    def capture_prefix(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        lens: torch.Tensor,
        *,
        reserve_current_tokens: int = 0,
    ) -> HunyuanPromptKVBatch:
        with profile_range("hy3.paged_kv.capture_prefix"):
            if not self.enabled:
                raise RuntimeError("Hunyuan paged KV capture called while disabled.")
            if key.shape != value.shape:
                raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
            self._ensure_compatible(key)
            assert self._k_pool is not None and self._v_pool is not None

            if lens.dim() != 1 or lens.numel() != key.shape[0]:
                raise ValueError("lens must be 1D and match the key/value batch size.")
            lens_cpu = lens.detach().to(device="cpu", dtype=torch.long)
            lens_values = [int(v) for v in lens_cpu.tolist()]
            if any(row_len <= 0 for row_len in lens_values):
                raise ValueError("Hunyuan paged KV prefix lens must be positive.")
            if any(row_len > key.shape[1] for row_len in lens_values):
                raise ValueError("Hunyuan paged KV prefix lens exceeds key/value length.")

            row_refs: list[HunyuanPromptKVRowRef] = []
            next_block = self._persistent_blocks
            total_new_blocks = 0
            for row_len in lens_values:
                num_blocks = _ceil_div(row_len, self.page_size)
                block_ids = tuple(range(next_block, next_block + num_blocks))
                next_block += num_blocks
                total_new_blocks += num_blocks
                row_refs.append(HunyuanPromptKVRowRef(owner=self, block_ids=block_ids, lens=row_len))

            reserve_blocks = 0
            reserve_current_tokens = max(int(reserve_current_tokens), 0)
            if reserve_current_tokens:
                for row_ref in row_refs:
                    row_seq_len = row_ref.lens + reserve_current_tokens
                    reserve_blocks += max(0, _ceil_div(row_seq_len, self.page_size) - len(row_ref.block_ids))

            self._ensure_capacity(next_block + reserve_blocks)
            for row, row_ref in enumerate(row_refs):
                start_slot = int(row_ref.block_ids[0]) * self.page_size
                slots = torch.arange(start_slot, start_slot + row_ref.lens, dtype=torch.long, device=key.device)
                self._write_paged_kv_slots(key[row, : row_ref.lens], value[row, : row_ref.lens], slots)

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

    @classmethod
    def _build_custom_attention_mask_result(
        cls,
        attention_mask: torch.Tensor | None,
        *,
        row_refs: list[HunyuanPromptKVRowRef],
        q_len: int,
        seq_len: int,
    ) -> _CustomMaskBuildResult:
        if attention_mask is None or attention_mask.numel() == 0:
            return _CustomMaskBuildResult(None, _MASK_NONE_OR_EMPTY)
        if attention_mask.dtype != torch.bool:
            if cls.attention_mask_is_all_keep(attention_mask):
                return _CustomMaskBuildResult(None, _MASK_INPUT_ALL_KEEP)
            raise ValueError(
                f"Hunyuan Image3 paged KV attention only supports boolean custom masks, got {attention_mask.dtype}."
            )

        bs = len(row_refs)
        mask = attention_mask
        if mask.dim() == 4:
            mask = mask[:, 0]
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((bs, q_len, seq_len))
            else:
                mask = mask.broadcast_to((q_len, seq_len)).unsqueeze(0).expand(bs, -1, -1)
        except RuntimeError as exc:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} cannot broadcast to "
                f"(batch={bs}, q_len={q_len}, seq_len={seq_len})."
            ) from exc

        dense_prefix_len = seq_len - q_len
        if dense_prefix_len < 0:
            raise ValueError(f"seq_len({seq_len}) must be >= q_len({q_len}).")

        mask_parts: list[torch.Tensor] = []
        for row, row_ref in enumerate(row_refs):
            if row_ref.lens > dense_prefix_len:
                raise ValueError(
                    f"cached prefix length({row_ref.lens}) cannot exceed dense prefix length({dense_prefix_len})."
                )
            prefix_mask = mask[row, :, : row_ref.lens]
            current_mask = mask[row, :, dense_prefix_len : dense_prefix_len + q_len]
            mask_parts.append(torch.cat([prefix_mask, current_mask], dim=1).contiguous().reshape(-1))

        packed = torch.cat(mask_parts, dim=0)
        with profile_range("hy3.paged_kv.custom_mask.effective_all_keep"):
            if bool(torch.all(packed).item()):
                return _CustomMaskBuildResult(None, _MASK_EFFECTIVE_ALL_KEEP)
        return _CustomMaskBuildResult(packed.contiguous(), _MASK_CUSTOM)

    @classmethod
    def build_custom_attention_mask(
        cls,
        attention_mask: torch.Tensor | None,
        *,
        row_refs: list[HunyuanPromptKVRowRef],
        q_len: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        return cls._build_custom_attention_mask_result(
            attention_mask,
            row_refs=row_refs,
            q_len=q_len,
            seq_len=seq_len,
        ).mask

    @classmethod
    def _build_full_attention_mask_result(
        cls,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        q_len: int,
        seq_len: int,
    ) -> _CustomMaskBuildResult:
        if attention_mask is None or attention_mask.numel() == 0:
            return _CustomMaskBuildResult(None, _MASK_NONE_OR_EMPTY)
        if attention_mask.dtype != torch.bool:
            if cls.attention_mask_is_all_keep(attention_mask):
                return _CustomMaskBuildResult(None, _MASK_INPUT_ALL_KEEP)
            raise ValueError(
                f"Hunyuan Image3 paged KV attention only supports boolean custom masks, got {attention_mask.dtype}."
            )

        mask = attention_mask
        if mask.dim() == 4:
            mask = mask[:, 0]
        if mask.dim() == 3 and mask.shape[0] == 1 and batch_size != 1:
            mask = mask[0]
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((batch_size, q_len, seq_len))
            else:
                mask = mask.broadcast_to((q_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
        except RuntimeError as exc:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} cannot broadcast to "
                f"(batch={batch_size}, q_len={q_len}, seq_len={seq_len})."
            ) from exc

        packed = mask.contiguous().reshape(-1)
        with profile_range("hy3.paged_kv.full_mask.effective_all_keep"):
            if bool(torch.all(packed).item()):
                return _CustomMaskBuildResult(None, _MASK_EFFECTIVE_ALL_KEEP)
        return _CustomMaskBuildResult(packed.contiguous(), _MASK_CUSTOM)

    @classmethod
    def build_full_attention_mask(
        cls,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        q_len: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        return cls._build_full_attention_mask_result(
            attention_mask,
            batch_size=batch_size,
            q_len=q_len,
            seq_len=seq_len,
        ).mask

    @staticmethod
    def _build_mask_pack_indptr(
        *,
        query_start_loc: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        page_counts = kv_indptr[1:] - kv_indptr[:-1]
        kv_lens = (page_counts - 1).clamp_min(0) * int(page_size) + kv_last_page_len
        mask_lens = q_lens * kv_lens
        mask_indptr = torch.empty_like(query_start_loc)
        mask_indptr[0] = 0
        mask_indptr[1:] = torch.cumsum(mask_lens.to(dtype=mask_indptr.dtype), dim=0)
        return mask_indptr

    @classmethod
    def _lookup_packed_custom_mask(cls, cache_key: tuple[Any, ...] | None) -> torch.Tensor | None:
        if cache_key is None:
            return None
        packed = cls._packed_custom_mask_cache.get(cache_key)
        if packed is None:
            return None
        cls._packed_custom_mask_cache.move_to_end(cache_key)
        return packed

    @classmethod
    def _store_packed_custom_mask(cls, cache_key: tuple[Any, ...] | None, packed: torch.Tensor) -> None:
        if cache_key is None:
            return
        cls._packed_custom_mask_cache[cache_key] = packed
        cls._packed_custom_mask_cache.move_to_end(cache_key)
        while len(cls._packed_custom_mask_cache) > _PACKED_CUSTOM_MASK_CACHE_MAX_ENTRIES:
            cls._packed_custom_mask_cache.popitem(last=False)

    def _record_custom_mask_result(self, result: _CustomMaskBuildResult) -> None:
        if result.reason == _MASK_NONE_OR_EMPTY:
            self.stats["paged_mask_none_or_empty_skips"] += 1
        elif result.reason == _MASK_INPUT_ALL_KEEP:
            self.stats["paged_mask_input_all_keep_skips"] += 1
        elif result.reason == _MASK_EFFECTIVE_ALL_KEEP:
            self.stats["paged_mask_effective_all_keep_skips"] += 1
        elif result.reason == _MASK_CUSTOM:
            self.stats["paged_mask_custom_builds"] += 1

    def _lookup_packed_custom_mask_with_stats(self, cache_key: tuple[Any, ...] | None) -> torch.Tensor | None:
        cached = self._lookup_packed_custom_mask(cache_key)
        if cache_key is None:
            return cached
        if cached is None:
            self.stats["paged_mask_packed_cache_misses"] += 1
        else:
            self.stats["paged_mask_packed_cache_hits"] += 1
        return cached

    def _pack_custom_mask(
        self,
        custom_mask: torch.Tensor | None,
        mask_indptr: torch.Tensor,
        *,
        cache_key: tuple[Any, ...] | None,
    ) -> torch.Tensor | None:
        if custom_mask is None or not custom_mask.is_cuda:
            return None
        cached = self._lookup_packed_custom_mask(cache_key)
        if cached is not None:
            self.stats["paged_mask_packed_cache_hits"] += 1
            return cached

        self.stats["paged_mask_packbits_calls"] += 1
        profile_start = _profile_start(custom_mask)
        try:
            segment_packbits = _load_flashinfer_segment_packbits()
            with profile_range("hy3.paged_kv.custom_mask.packbits"):
                packed_custom_mask, _ = segment_packbits(
                    custom_mask.contiguous().view(-1),
                    mask_indptr,
                    bitorder="little",
                )
        finally:
            _profile_finish(self.stats, "paged_profile_packed_mask_build_ms", profile_start, custom_mask)
        self._store_packed_custom_mask(cache_key, packed_custom_mask)
        return packed_custom_mask

    def _get_flashinfer_runner(self) -> HunyuanFlashInferPagedKVRunner:
        if self._flashinfer_runner is None:
            if HunyuanPromptKVPagePool._shared_flashinfer_runner is None:
                HunyuanPromptKVPagePool._shared_flashinfer_runner = HunyuanFlashInferPagedKVRunner()
            return HunyuanPromptKVPagePool._shared_flashinfer_runner
        return self._flashinfer_runner

    def _run_attention_from_inputs(
        self,
        query: torch.Tensor,
        inputs: HunyuanPagedAttentionInputs,
        *,
        softmax_scale: float,
    ) -> torch.Tensor:
        assert self._kv_pool is not None
        if inputs.custom_mask is not None or inputs.packed_custom_mask is not None:
            runner = self._get_flashinfer_runner()
            if is_hunyuan_image3_paged_kv_profile_enabled():
                with profile_range("hy3.paged_kv.attention.flashinfer"):
                    return runner.run(
                        query,
                        self._kv_pool[0],
                        self._kv_pool[1],
                        inputs,
                        softmax_scale=softmax_scale,
                        profile_stats=self.stats,
                    )
            with profile_range("hy3.paged_kv.attention.flashinfer"):
                return runner.run(
                    query,
                    self._kv_pool[0],
                    self._kv_pool[1],
                    inputs,
                    softmax_scale=softmax_scale,
                )
        profile_start = _profile_start(query)
        try:
            with profile_range("hy3.paged_kv.attention.fast"):
                return ar_diffusion_paged_attention(
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
        finally:
            _profile_finish(self.stats, "paged_profile_fast_attention_ms", profile_start, query)

    def run_first_step_paged_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cached_prompt_lens: torch.Tensor,
        *,
        seq_len: int,
        softmax_scale: float,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Hunyuan paged KV first-step attention called while disabled.")
        if key.shape != value.shape:
            raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
        if query.dim() != 4 or key.dim() != 4:
            raise ValueError("Hunyuan first-step paged KV attention expects 4D query/key/value tensors.")
        self._ensure_compatible(key)

        batch_size, key_len = key.shape[:2]
        q_len = query.shape[1]
        if query.shape[0] != batch_size:
            raise ValueError("query batch size does not match key/value batch size.")
        if int(seq_len) != key_len:
            raise ValueError(f"seq_len({seq_len}) must match first-step KV length({key_len}).")
        if q_len > key_len:
            raise ValueError(f"query length({q_len}) cannot exceed KV length({key_len}).")
        if cached_prompt_lens.dim() != 1 or cached_prompt_lens.numel() != batch_size:
            raise ValueError("cached_prompt_lens must be 1D and match the key/value batch size.")
        if torch.any(cached_prompt_lens <= 0):
            raise ValueError("Hunyuan paged KV prefix lens must be positive.")
        if torch.any(cached_prompt_lens > key_len):
            raise ValueError("Hunyuan paged KV prefix lens exceeds key/value length.")

        prefix_row_refs: list[HunyuanPromptKVRowRef] = []
        block_rows: list[list[int]] = []
        row_prefix_blocks: list[int] = []
        row_full_blocks: list[int] = []
        next_block = self._persistent_blocks
        total_prefix_blocks = 0
        for row in range(batch_size):
            prefix_len = int(cached_prompt_lens[row].item())
            prefix_blocks = _ceil_div(prefix_len, self.page_size)
            full_blocks = _ceil_div(key_len, self.page_size)
            block_ids = tuple(range(next_block, next_block + prefix_blocks))
            next_block += prefix_blocks
            total_prefix_blocks += prefix_blocks
            row_prefix_blocks.append(prefix_blocks)
            row_full_blocks.append(full_blocks)
            prefix_row_refs.append(HunyuanPromptKVRowRef(owner=self, block_ids=block_ids, lens=prefix_len))

        scratch_cursor = next_block
        current_blocks = 0
        kv_indices: list[int] = []
        kv_indptr = [0]
        kv_last_page_len: list[int] = []
        seq_lens: list[int] = []
        for row_ref, prefix_blocks, full_blocks in zip(prefix_row_refs, row_prefix_blocks, row_full_blocks):
            extra_blocks = full_blocks - prefix_blocks
            if extra_blocks < 0:
                raise AssertionError("paged first-step metadata would drop prefix blocks.")
            scratch_blocks = list(range(scratch_cursor, scratch_cursor + extra_blocks))
            scratch_cursor += extra_blocks
            current_blocks += extra_blocks
            row_blocks = list(row_ref.block_ids) + scratch_blocks
            block_rows.append(row_blocks)
            kv_indices.extend(row_blocks)
            kv_indptr.append(kv_indptr[-1] + len(row_blocks))
            last_page_len = key_len % self.page_size
            kv_last_page_len.append(self.page_size if last_page_len == 0 else last_page_len)
            seq_lens.append(key_len)

        self._ensure_capacity(scratch_cursor)
        with profile_range("hy3.paged_kv.first_step_write_all"):
            for row, row_blocks in enumerate(block_rows):
                positions = torch.arange(key_len, dtype=torch.long, device=key.device)
                slots = compute_slot_mapping(row_blocks, positions, self.page_size).to(device=key.device)
                self._write_paged_kv_slots(key[row], value[row], slots)

        with profile_range("hy3.paged_kv.first_step_build_inputs"):
            max_pages = max(len(row) for row in block_rows)
            padded_rows = [row + [0] * (max_pages - len(row)) for row in block_rows]
            device = key.device
            block_table = torch.tensor(padded_rows, dtype=torch.int32, device=device)
            query_start_loc = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device)
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
            kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
            kv_indices_tensor = torch.tensor(kv_indices, dtype=torch.int32, device=device)
            kv_last_page_len_tensor = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)
        mask_cache_key = (
            "first",
            _tensor_identity_key(attention_mask),
            int(batch_size),
            int(q_len),
            int(key_len),
        )
        packed_custom_mask = (
            self._lookup_packed_custom_mask_with_stats(mask_cache_key)
            if attention_mask is not None and attention_mask.numel() != 0
            else None
        )
        custom_mask: torch.Tensor | None = None
        if packed_custom_mask is None:
            profile_start = _profile_start(key)
            try:
                with profile_range("hy3.paged_kv.first_step_mask_build"):
                    mask_result = self._build_full_attention_mask_result(
                        attention_mask,
                        batch_size=batch_size,
                        q_len=int(q_len),
                        seq_len=int(key_len),
                    )
                    custom_mask = mask_result.mask
                    self._record_custom_mask_result(mask_result)
            finally:
                _profile_finish(self.stats, "paged_profile_mask_build_ms", profile_start, key)
            if custom_mask is not None:
                mask_indptr = self._build_mask_pack_indptr(
                    query_start_loc=query_start_loc,
                    kv_indptr=kv_indptr_tensor,
                    kv_last_page_len=kv_last_page_len_tensor,
                    page_size=self.page_size,
                )
                packed_custom_mask = self._pack_custom_mask(
                    custom_mask,
                    mask_indptr,
                    cache_key=mask_cache_key,
                )
        has_custom_mask = custom_mask is not None or packed_custom_mask is not None
        plan_cache_key = (
            "first",
            int(batch_size),
            int(q_len),
            int(key_len),
            int(self.page_size),
            tuple(kv_indptr),
            tuple(kv_indices),
            tuple(kv_last_page_len),
            tuple(seq_lens),
            mask_cache_key if has_custom_mask else None,
        )
        inputs = HunyuanPagedAttentionInputs(
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_tensor,
            kv_indptr=kv_indptr_tensor,
            kv_indices=kv_indices_tensor,
            kv_last_page_len=kv_last_page_len_tensor,
            custom_mask=custom_mask,
            packed_custom_mask=packed_custom_mask,
            plan_cache_key=plan_cache_key,
            max_query_len=int(q_len),
            max_seq_len=max(seq_lens),
            prefix_blocks=total_prefix_blocks,
            current_blocks=current_blocks,
        )

        self._persistent_blocks = next_block
        self._current_batch = HunyuanPromptKVBatch(owner=self, row_refs=prefix_row_refs)
        with profile_range("hy3.paged_kv.first_step_run_attention"):
            out = self._run_attention_from_inputs(query, inputs, softmax_scale=softmax_scale)
        self.stats["paged_cache_builds"] += 1
        self.stats["paged_attention_calls"] += 1
        if inputs.custom_mask is not None or inputs.packed_custom_mask is not None:
            self.stats["paged_attention_custom_mask_calls"] += 1
        self.stats["paged_prefix_blocks"] += total_prefix_blocks
        self.stats["paged_current_blocks"] += int(current_blocks)
        return out

    def _build_attention_inputs(
        self,
        key: torch.Tensor,
        seq_len: int,
        attention_mask: torch.Tensor | None = None,
    ) -> HunyuanPagedAttentionInputs:
        batch = self._current_batch
        if batch is None:
            raise RuntimeError("Hunyuan paged KV attention has no current prompt KV batch.")
        if key.dim() != 4:
            raise ValueError(f"current key must be 4D, got {tuple(key.shape)}")
        if key.shape[0] != len(batch.row_refs):
            raise ValueError("current key batch size does not match prompt KV rows.")

        bs, q_len = key.shape[:2]
        block_rows: list[list[int]] = []
        kv_indices: list[int] = []
        kv_indptr = [0]
        kv_last_page_len: list[int] = []
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
            row_blocks = list(row_ref.block_ids) + scratch_blocks
            block_rows.append(row_blocks)
            kv_indices.extend(row_blocks)
            kv_indptr.append(kv_indptr[-1] + len(row_blocks))
            last_page_len = row_seq_len % self.page_size
            kv_last_page_len.append(self.page_size if last_page_len == 0 else last_page_len)
            seq_lens.append(row_seq_len)

        self._ensure_capacity(scratch_cursor)

        max_pages = max(len(row) for row in block_rows)
        padded_rows = [row + [0] * (max_pages - len(row)) for row in block_rows]
        device = key.device
        block_table = torch.tensor(padded_rows, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(0, (bs + 1) * q_len, q_len, dtype=torch.int32, device=device)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
        kv_indices_tensor = torch.tensor(kv_indices, dtype=torch.int32, device=device)
        kv_last_page_len_tensor = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)
        mask_cache_key = (
            "reuse",
            _tensor_identity_key(attention_mask),
            tuple(row.lens for row in batch.row_refs),
            int(q_len),
            int(seq_len),
        )
        packed_custom_mask = (
            self._lookup_packed_custom_mask_with_stats(mask_cache_key)
            if attention_mask is not None and attention_mask.numel() != 0
            else None
        )
        custom_mask: torch.Tensor | None = None
        if packed_custom_mask is None:
            profile_start = _profile_start(key)
            try:
                with profile_range("hy3.paged_kv.reuse_mask_build"):
                    mask_result = self._build_custom_attention_mask_result(
                        attention_mask,
                        row_refs=batch.row_refs,
                        q_len=int(q_len),
                        seq_len=int(seq_len),
                    )
                    custom_mask = mask_result.mask
                    self._record_custom_mask_result(mask_result)
            finally:
                _profile_finish(self.stats, "paged_profile_mask_build_ms", profile_start, key)
            if custom_mask is not None:
                mask_indptr = self._build_mask_pack_indptr(
                    query_start_loc=query_start_loc,
                    kv_indptr=kv_indptr_tensor,
                    kv_last_page_len=kv_last_page_len_tensor,
                    page_size=self.page_size,
                )
                packed_custom_mask = self._pack_custom_mask(
                    custom_mask,
                    mask_indptr,
                    cache_key=mask_cache_key,
                )
        has_custom_mask = custom_mask is not None or packed_custom_mask is not None
        plan_cache_key = (
            "reuse",
            int(bs),
            int(q_len),
            int(seq_len),
            int(self.page_size),
            tuple(kv_indptr),
            tuple(kv_indices),
            tuple(kv_last_page_len),
            tuple(seq_lens),
            mask_cache_key if has_custom_mask else None,
        )
        return HunyuanPagedAttentionInputs(
            block_table=block_table,
            block_rows=tuple(tuple(row) for row in block_rows),
            query_start_loc=query_start_loc,
            seq_lens=seq_lens_tensor,
            kv_indptr=kv_indptr_tensor,
            kv_indices=kv_indices_tensor,
            kv_last_page_len=kv_last_page_len_tensor,
            custom_mask=custom_mask,
            packed_custom_mask=packed_custom_mask,
            plan_cache_key=plan_cache_key,
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
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key.shape != value.shape:
            raise ValueError(f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}")
        self._ensure_compatible(key)
        with profile_range("hy3.paged_kv.reuse_build_inputs"):
            inputs = self._build_attention_inputs(key, seq_len, attention_mask)
        assert self._kv_pool is not None and self._k_pool is not None and self._v_pool is not None

        batch = self._current_batch
        assert batch is not None
        if len(inputs.block_rows) != len(batch.row_refs):
            raise AssertionError("paged reuse inputs must include one block row per batch row.")
        q_len = key.shape[1]
        with profile_range("hy3.paged_kv.reuse_write_current"):
            for row, row_ref in enumerate(batch.row_refs):
                positions = torch.arange(row_ref.lens, row_ref.lens + q_len, dtype=torch.long, device=key.device)
                slots = compute_slot_mapping(inputs.block_rows[row], positions, self.page_size).to(device=key.device)
                self._write_paged_kv_slots(key[row], value[row], slots)

        with profile_range("hy3.paged_kv.reuse_run_attention"):
            out = self._run_attention_from_inputs(query, inputs, softmax_scale=softmax_scale)
        self.stats["paged_attention_calls"] += 1
        if inputs.custom_mask is not None or inputs.packed_custom_mask is not None:
            self.stats["paged_attention_custom_mask_calls"] += 1
        self.stats["paged_current_blocks"] += int(inputs.current_blocks)
        return out
