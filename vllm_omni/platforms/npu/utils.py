# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

_diffusion_paged_kv_write_plan: ContextVar[Any | None] = ContextVar(
    "diffusion_paged_kv_write_plan",
    default=None,
)


def _logical_cache_to_pa_nz(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    head_size: int,
    *,
    last_dim: int = 16,
) -> torch.Tensor:
    """Convert selected logical ``[N, B, H, D]`` pages to PA_NZ."""

    selected = cache.index_select(0, block_ids)
    block_size = selected.shape[1]
    return (
        selected.permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), cache.shape[2], block_size, head_size // last_dim, last_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(block_ids.numel(), cache.shape[2] * head_size // last_dim, block_size, last_dim)
    )


def _pa_nz_to_logical_cache(
    cache_nz: torch.Tensor,
    head_size: int,
    *,
    last_dim: int = 16,
) -> torch.Tensor:
    """Convert PA_NZ pages back to logical ``[N, B, H, D]`` order."""

    num_heads = cache_nz.shape[1] * last_dim // head_size
    return (
        cache_nz.reshape(cache_nz.shape[0], num_heads, head_size // last_dim, cache_nz.shape[2], last_dim)
        .permute(0, 3, 1, 2, 4)
        .reshape(cache_nz.shape[0], cache_nz.shape[2], num_heads, head_size)
    )


@contextmanager
def _use_diffusion_paged_kv_write_plan(write_plan: Any):
    token = _diffusion_paged_kv_write_plan.set(write_plan)
    try:
        yield
    finally:
        _diffusion_paged_kv_write_plan.reset(token)


def _reshape_and_cache_without_cache_mode(cls, key, value, key_cache, value_cache, slot_mapping):
    """Run the legacy scatter schema, using a static diffusion plan when set."""

    del cls
    import torch_npu

    if (
        key_cache.dim() == 4
        and value_cache.dim() == 4
        and key.dim() == 3
        and key_cache.shape[2] == key.shape[1]
        and key_cache.shape[3] == key.shape[2]
        and value_cache.shape[2] == value.shape[1]
        and value_cache.shape[3] == value.shape[2]
    ):
        block_size = key_cache.shape[1]
        last_dim = 16
        if key.shape[-1] % last_dim or value.shape[-1] % last_dim:
            raise RuntimeError("Ascend cache-writer compatibility requires key/value head sizes divisible by 16")
        write_plan = _diffusion_paged_kv_write_plan.get()
        if write_plan is None:
            valid_mask = slot_mapping >= 0
            slot_blocks = slot_mapping // block_size
            block_ids = torch.unique(slot_blocks[valid_mask], sorted=True)
            local_slots = torch.full_like(slot_mapping, -1)
            local_slots[valid_mask] = (
                torch.searchsorted(block_ids, slot_blocks[valid_mask]) * block_size
                + slot_mapping[valid_mask] % block_size
            ).to(dtype=slot_mapping.dtype)
        else:
            block_ids = write_plan.block_ids
            local_slots = write_plan.local_slot_mapping
            if local_slots.numel() != slot_mapping.numel():
                raise RuntimeError(
                    "Static diffusion KV write plan does not match the current write: "
                    f"slots={local_slots.numel()}, expected={slot_mapping.numel()}"
                )
        if block_ids.numel() == 0:
            return
        key_cache_nz = _logical_cache_to_pa_nz(key_cache, block_ids, key.shape[-1], last_dim=last_dim)
        value_cache_nz = _logical_cache_to_pa_nz(value_cache, block_ids, value.shape[-1], last_dim=last_dim)
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache_nz,
            value_cache=value_cache_nz,
            slot_mapping=local_slots.contiguous(),
        )
        key_cache.index_copy_(
            0,
            block_ids,
            _pa_nz_to_logical_cache(key_cache_nz, key.shape[-1], last_dim=last_dim),
        )
        value_cache.index_copy_(
            0,
            block_ids,
            _pa_nz_to_logical_cache(value_cache_nz, value.shape[-1], last_dim=last_dim),
        )
        return
    torch_npu.npu_scatter_pa_kv_cache(
        key=key.contiguous(),
        value=value.contiguous(),
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping.contiguous(),
    )
