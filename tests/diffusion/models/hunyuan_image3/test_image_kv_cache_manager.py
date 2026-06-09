# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ImageKVCacheManager.

Covers: cache → reuse flow, AR KV injection, CFG (sequential & parallel), SP, cross-request isolation.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TRANSFORMER_MODULE = "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer"

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
IMAGE_TOKEN_LEN = 8
SUFFIX_TOKEN_LEN = 3
SCALING = 1.0 / math.sqrt(HEAD_DIM)
CAPTURED_ATTN_METADATA = []
CAPTURED_ATTN_CALLS = []


# ============================================================
# Mocks + helpers
# ============================================================


class MockAttention(nn.Module):
    def __init__(self, num_heads, head_size, causal=False, softmax_scale=None, num_kv_heads=None, **kwargs):
        super().__init__()

    def forward(self, query, key, value, attn_metadata=None, **kwargs):
        CAPTURED_ATTN_METADATA.append(attn_metadata)
        CAPTURED_ATTN_CALLS.append((query, key, value, attn_metadata))
        if attn_metadata is None or attn_metadata.attn_mask is None:
            return query
        return query


class FakePagedKVRunner:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def run(self, query, key, value, metadata, *, sm_scale, causal, attn_mask=None):
        del causal, attn_mask
        self.calls.append((query, key, value, metadata))
        outputs = []
        custom_mask_offset = 0
        for b in range(query.shape[0]):
            page_start = int(metadata.kv_indptr[b].item())
            page_end = int(metadata.kv_indptr[b + 1].item())
            page_indices = metadata.kv_indices[page_start:page_end]
            cached_len = int(metadata.cached_lens[b].item())
            prefix_key = metadata.key_cache[page_indices].reshape(-1, key.shape[2], key.shape[3])[:cached_len]
            prefix_value = metadata.value_cache[page_indices].reshape(-1, value.shape[2], value.shape[3])[:cached_len]
            dense_key = torch.cat([prefix_key, key[b]], dim=0).unsqueeze(0)
            dense_value = torch.cat([prefix_value, value[b]], dim=0).unsqueeze(0)
            custom_mask = None
            if metadata.custom_mask is not None:
                kv_len = int(metadata.seq_lens[b].item())
                item_size = query.shape[1] * kv_len
                custom_mask = metadata.custom_mask[
                    custom_mask_offset : custom_mask_offset + item_size
                ].reshape(1, query.shape[1], kv_len)
                custom_mask_offset += item_size
            outputs.append(_dense_attention(query[b : b + 1], dense_key, dense_value, sm_scale, attn_mask=custom_mask)[0])
        return torch.stack(outputs, dim=0)


@contextmanager
def patched_mgr_env(sp_size=1):
    target = _TRANSFORMER_MODULE
    patches = [
        patch(f"{target}.get_sequence_parallel_world_size", return_value=sp_size),
        patch(f"{target}.get_sequence_parallel_rank", return_value=0),
        patch(f"{target}.Attention", MockAttention),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _make_cache_mgr(image_token_len=IMAGE_TOKEN_LEN, sp_size=1):
    with patched_mgr_env(sp_size=sp_size):
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            ImageKVCacheManager,
        )

        mgr = ImageKVCacheManager(
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            scaling=SCALING,
            image_token_len=image_token_len,
        )
    return mgr


def _make_known_kv(num_tokens, base=0.0):
    """Create key/value with known values. Token i has all elements = base+i / base+i+0.5."""
    k = torch.full((num_tokens, NUM_KV_HEADS, HEAD_DIM), 0.0)
    v = torch.full((num_tokens, NUM_KV_HEADS, HEAD_DIM), 0.0)
    for i in range(num_tokens):
        k[i] = base + i
        v[i] = base + i + 0.5
    return k, v


def _repeat_kv_for_test(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(bsz, seq_len, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(bsz, seq_len, num_kv_heads * n_rep, head_dim)


def _dense_attention(query, key, value, scale, attn_mask=None):
    repeat_num = query.shape[2] // key.shape[2]
    key = _repeat_kv_for_test(key, repeat_num)
    value = _repeat_kv_for_test(value, repeat_num)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def _gen_timestep_index(bs: int, current_start: int) -> torch.Tensor:
    return torch.full((bs, 1), current_start, dtype=torch.long)


def _call_mgr(
    mgr,
    bs,
    q_len,
    seq_len,
    key_flat,
    value_flat,
    query_flat=None,
    attention_mask=None,
    first_step=False,
    uncond_cfg_prefill=False,
    num_image_tokens=IMAGE_TOKEN_LEN,
    shard_image_size=None,
    gen_timestep_scatter_index=None,
    position_ids=None,
    full_attn_spans=None,
):
    query = torch.randn(bs * q_len, NUM_HEADS, HEAD_DIM) if query_flat is None else query_flat
    attn_mask = torch.zeros(bs, 1, seq_len, seq_len) if attention_mask is None else attention_mask
    return mgr(
        query,
        key_flat,
        value_flat,
        attn_mask,
        query_lens=[q_len] * bs,
        seq_lens=[seq_len] * bs,
        first_step=first_step,
        uncond_cfg_prefill=uncond_cfg_prefill,
        num_image_tokens=num_image_tokens,
        shard_image_size=shard_image_size,
        gen_timestep_scatter_index=gen_timestep_scatter_index,
        position_ids=position_ids,
        full_attn_spans=full_attn_spans,
    )


# ============================================================
# Test 1: No AR KV — basic cache → reuse
# ============================================================


@pytest.mark.parametrize("bs", [1, 2])
def test_no_ar_kv(bs):
    """
    No AR KV injected. Tests the basic first_step cache → update reuse path.

    Sequence layout per batch on first_step (q_len=14, IMAGE_TOKEN_LEN=8):
        [prompt(3) | current timestep/image(8) | suffix(3)]
        gen_timestep_scatter_index = 3
        cached_prompt_len = 3

    After first_step:
        image_kv_cache_map stores prompt tokens [0:3] for each batch.

    Update step (q_len=IMAGE_TOKEN_LEN=8, seq_len = cached_prompt(3) + current(8) = 11):
        _reuse_prompt_kv produces [cached_prompt(3) | new_current(8)] per batch.
    """
    mgr = _make_cache_mgr()
    assert mgr.image_kv_cache_map is None

    # --- first_step ---
    prompt_len = 3
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    _call_mgr(
        mgr,
        bs,
        q_len=q_len,
        seq_len=q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    cached_key, cached_value = mgr.image_kv_cache_map
    # 3 prompt tokens cached per batch
    assert cached_key.shape == (bs, prompt_len, NUM_KV_HEADS, HEAD_DIM)
    assert torch.equal(mgr.image_kv_cache_lens, torch.full((bs,), prompt_len))
    for b in range(bs):
        flat_offset = b * q_len
        assert torch.allclose(cached_key[b, :prompt_len], k_flat[flat_offset : flat_offset + prompt_len])
        assert torch.allclose(cached_value[b, :prompt_len], v_flat[flat_offset : flat_offset + prompt_len])

    # --- update step ---
    img_q_len = IMAGE_TOKEN_LEN
    update_seq_len = prompt_len + img_q_len  # cached_prompt(3) + current(8) = 11
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)

    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    val_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    position_ids = torch.arange(prompt_len, prompt_len + img_q_len).repeat(bs, 1)
    result_k, result_v = mgr._reuse_prompt_kv(key_input, val_input, update_seq_len, bs=bs, position_ids=position_ids)

    assert result_k.shape == (bs, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
    for b in range(bs):
        img_offset = b * img_q_len
        # Cached prompt preserved
        assert torch.allclose(result_k[b, :prompt_len], cached_key[b, :prompt_len])
        # New image tokens
        assert torch.allclose(
            result_k[b, prompt_len : prompt_len + img_q_len],
            new_img_k[img_offset : img_offset + img_q_len],
        )


def test_profile_records_dense_later_reuse_and_attention(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_PROFILE", "1")
    mgr = _make_cache_mgr()
    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=prompt_len + IMAGE_TOKEN_LEN,
        key_flat=img_k,
        value_flat=img_v,
        first_step=False,
        position_ids=position_ids,
    )

    stats = mgr.get_paged_kv_cache_stats()
    assert stats["paged_kv_profile_enabled"] is True
    assert stats["profile_dense_reuse_calls"] == 1
    assert stats["profile_dense_later_attention_calls"] == 1
    assert stats["profile_dense_reuse_total_ms"] >= 0.0
    assert stats["profile_dense_later_attention_total_ms"] >= 0.0


# ============================================================
# Test 2: AR KV, no CFG
# ============================================================


@pytest.mark.parametrize("sp_size", [1, 2])
def test_ar_kv_no_cfg(sp_size):
    """
    AR KV injected, bs=1, no CFG. Tests AR prefix prepend + cache + reuse.

    sp_size=1:
        first_step input: q_len=14, ar_len=5, seq_len=19
        Sequence layout: [ar(5) | prompt(3) | current(8) | suffix(3)]
        gen_timestep_scatter_index = 3
        cached_prompt_len = ar(5) + prompt(3) = 8

        Update step: q_len=8, seq_len = cached_prompt(8) + current(8) = 16
        Result: [cached(8) | new_current(8)]

    sp_size=2:
        shard_image_size = 4 (passed externally, simulating SP sharding)
        gen_timestep_scatter_index = 8
        cached_prompt_len = seq_len - shard_image_size = 17 - 4 = 13 (= ar(5) + prompt(8))
        Note: with SP, more of the sequence is treated as "prompt" because
        image tokens are sharded across ranks.

        Update step: q_len=4 (shard_image_size), seq_len = cached_prompt(13) + shard_image(4) = 17
        SP path returns only cached prompt [bs, cached_prompt_len, ...] (no eoi concat).

    Both sp_size cases test _cache_prompt_kv and _reuse_prompt_kv directly to avoid
    needing full SP infrastructure mocks in __call__.
    """
    mgr = _make_cache_mgr(sp_size=sp_size)
    ar_len = 5
    ar_k, ar_v = _make_known_kv(ar_len, base=100.0)
    mgr._injected_ar_kv = [(ar_k.clone(), ar_v.clone())]

    # --- first_step: call _cache_prompt_kv directly ---
    bs = 1
    q_len = 12 if sp_size > 1 else 14
    seq_len = q_len + ar_len
    k_raw, v_raw = _make_known_kv(q_len, base=1.0)
    key_4d = k_raw.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)
    val_4d = v_raw.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)

    shard_image_size = 4 if sp_size > 1 else None
    current_start = 8 if sp_size > 1 else 3
    mgr.image_kv_cache_map = None
    mgr._cache_prompt_kv(
        key_4d,
        val_4d,
        seq_len,
        shard_image_size,
        gen_timestep_scatter_index=_gen_timestep_index(bs, current_start),
    )

    cached_key, cached_value = mgr.image_kv_cache_map
    if sp_size == 1:
        # cached = ar(5) + prompt(3) = 8
        expected_cached_len = 8
    else:
        # cached = seq_len - shard_image_size = 17 - 4 = 13
        expected_cached_len = seq_len - shard_image_size

    assert cached_key.shape == (bs, expected_cached_len, NUM_KV_HEADS, HEAD_DIM)
    # AR KV always at the front
    assert torch.allclose(cached_key[0, :ar_len], ar_k)
    assert torch.allclose(cached_value[0, :ar_len], ar_v)
    # Prompt tokens follow AR
    prompt_cached = expected_cached_len - ar_len
    assert torch.allclose(cached_key[0, ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])
    # AR KV consumed
    assert mgr._injected_ar_kv is None

    # --- update step: call _reuse_prompt_kv directly ---
    if sp_size == 1:
        img_q_len = IMAGE_TOKEN_LEN
        update_seq_len = expected_cached_len + img_q_len  # 8 + 8 = 16
        new_img_k, new_img_v = _make_known_kv(img_q_len, base=50.0)

        key_input = new_img_k.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        val_input = new_img_v.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        position_ids = torch.arange(expected_cached_len, expected_cached_len + img_q_len).reshape(1, img_q_len)
        result_k, result_v = mgr._reuse_prompt_kv(
            key_input,
            val_input,
            update_seq_len,
            bs=1,
            position_ids=position_ids,
        )

        assert result_k.shape == (1, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
        # AR + prompt preserved
        assert torch.allclose(result_k[0, :ar_len], ar_k)
        assert torch.allclose(result_k[0, ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])
        # New image tokens
        assert torch.allclose(result_k[0, expected_cached_len : expected_cached_len + img_q_len], new_img_k)
    else:
        # SP path: _reuse_prompt_kv returns only cached prompt (no image concat)
        img_q_len = shard_image_size  # 4
        update_seq_len = expected_cached_len + img_q_len  # 13 + 4 = 17
        new_img_k, new_img_v = _make_known_kv(img_q_len, base=50.0)

        key_input = new_img_k.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        val_input = new_img_v.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        result_k, result_v = mgr._reuse_prompt_kv(
            key_input,
            val_input,
            update_seq_len,
            bs=bs,
            shard_image_size=img_q_len,
        )

        # SP returns only the cached prompt portion
        assert result_k.shape == (1, expected_cached_len, NUM_KV_HEADS, HEAD_DIM)
        assert torch.allclose(result_k[0, :ar_len], ar_k)
        assert torch.allclose(result_k[0, ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])


# ============================================================
# Test 3: AR KV + CFG (sequential & parallel)
# ============================================================


@pytest.mark.parametrize("cfg_parallel,bs", [(False, 2), (True, 1)])
def test_ar_kv_with_cfg(cfg_parallel, bs):
    """
    AR KV + CFG. Tests uncond_cfg_prefill → first_step → update.

    Common setup:
        positive_reuse_len = 10, negative_reuse_len = 6, neg_uncond_cfg_q_len = 4
        AR KV: 10 tokens (base=100)

    Sequential CFG (cfg_parallel=False, bs=2):
        uncond_cfg_prefill (bs=1):
            Builds neg AR KV = [shared_prefix(6) from pos_ar | neg_prefill_tokens(4)]
            → _injected_ar_kv becomes [(pos_ar(10), pos_av(10)), (neg_k(10), neg_v(10))]

        first_step (bs=2, q_len=12, seq_len=22):
            Batch 0 (pos): [pos_ar(10) | prompt(3) | current(8) | suffix(3)]
            Batch 1 (neg): [neg_ar(10) | prompt(3) | current(8) | suffix(3)]
            cached_prompt_len per batch = 10 + 3 = 13

        Update (bs=2, seq_len = 13 + 8 = 21):
            Result per batch: [cached(13) | new_current(8)]

    CFG Parallel (cfg_parallel=True, bs=1):
        This rank handles only the negative branch.
        uncond_cfg_prefill (bs=1):
            Same as above: builds neg AR KV.
            Then we keep only the negative entry (simulating _keep_negative_kv_only).
            → _injected_ar_kv = [(neg_k(10), neg_v(10))]

        first_step (bs=1, q_len=12, seq_len=22):
            [neg_ar(10) | prompt(3) | current(8) | suffix(3)]
            cached_prompt_len = 13

        Update (bs=1, seq_len = 13 + 8 = 21):
            Result: [cached(13) | new_current(8)]
    """
    positive_reuse_len = 10
    negative_reuse_len = 6
    neg_uncond_cfg_q_len = positive_reuse_len - negative_reuse_len  # 4

    mgr = _make_cache_mgr()
    pos_ar_k, pos_ar_v = _make_known_kv(positive_reuse_len, base=100.0)
    mgr._injected_ar_kv = [(pos_ar_k.clone(), pos_ar_v.clone())]

    # --- uncond_cfg_prefill ---
    neg_k, neg_v = _make_known_kv(neg_uncond_cfg_q_len, base=200.0)
    prefill_seq_len = negative_reuse_len + neg_uncond_cfg_q_len  # 6 + 4 = 10

    _call_mgr(
        mgr,
        bs=1,
        q_len=neg_uncond_cfg_q_len,
        seq_len=prefill_seq_len,
        key_flat=neg_k,
        value_flat=neg_v,
        first_step=True,
        uncond_cfg_prefill=True,
        num_image_tokens=0,
    )

    # After prefill: _injected_ar_kv = [(pos), (neg)]
    assert len(mgr._injected_ar_kv) == 2
    neg_ar_k, neg_ar_v = mgr._injected_ar_kv[1]
    # neg AR KV = [shared_prefix(6) from pos | neg_prefill(4)], total 10
    assert neg_ar_k.shape[0] == positive_reuse_len
    assert torch.allclose(neg_ar_k[:negative_reuse_len], pos_ar_k[:negative_reuse_len])
    assert torch.allclose(neg_ar_k[negative_reuse_len:], neg_k)

    # --- simulate cfg_parallel: keep only negative ---
    if cfg_parallel:
        mgr._injected_ar_kv = [mgr._injected_ar_kv[1]]

    # --- first_step ---
    prompt_len = 3
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    seq_len = q_len + positive_reuse_len  # 22
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    _call_mgr(
        mgr,
        bs,
        q_len=q_len,
        seq_len=seq_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    cached_key, cached_value = mgr.image_kv_cache_map
    cached_prompt_len_per_batch = positive_reuse_len + prompt_len
    assert cached_key.shape == (bs, cached_prompt_len_per_batch, NUM_KV_HEADS, HEAD_DIM)
    assert mgr._injected_ar_kv is None

    if not cfg_parallel:
        # bs=2: batch 0 = pos, batch 1 = neg
        # Batch 0: pos_ar(10) + prompt(3)
        assert torch.allclose(cached_key[0, :positive_reuse_len], pos_ar_k)
        assert torch.allclose(cached_key[0, positive_reuse_len:13], k_flat[:3])
        # Batch 1: neg_ar(10) + prompt(3)
        assert torch.allclose(cached_key[1, :positive_reuse_len], neg_ar_k)
        assert torch.allclose(cached_key[1, positive_reuse_len:13], k_flat[q_len : q_len + 3])
    else:
        # bs=1: only neg branch
        assert torch.allclose(cached_key[0, :positive_reuse_len], neg_ar_k)
        assert torch.allclose(cached_key[0, positive_reuse_len:13], k_flat[:3])

    # --- update step ---
    img_q_len = IMAGE_TOKEN_LEN
    update_seq_len = cached_prompt_len_per_batch + img_q_len  # 13 + 8 = 21
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)

    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    val_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    position_ids = torch.arange(cached_prompt_len_per_batch, cached_prompt_len_per_batch + img_q_len).repeat(bs, 1)
    result_k, result_v = mgr._reuse_prompt_kv(
        key_input,
        val_input,
        update_seq_len,
        bs=bs,
        position_ids=position_ids,
    )

    assert result_k.shape == (bs, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
    for b in range(bs):
        # Cached prompt preserved
        assert torch.allclose(
            result_k[b, :cached_prompt_len_per_batch],
            cached_key[b, :cached_prompt_len_per_batch],
        )
        # New image tokens
        img_offset = b * img_q_len
        assert torch.allclose(
            result_k[b, cached_prompt_len_per_batch : cached_prompt_len_per_batch + img_q_len],
            new_img_k[img_offset : img_offset + img_q_len],
        )


# ============================================================
# Test 4: Cross-request isolation
# ============================================================


def test_cross_request_isolation():
    """
    Verify leftover image_kv_cache_map from a previous request is NOT treated as AR KV.

    Setup: mgr has stale cache from a prior request (9 tokens, base=999).
    New request: first_step with q_len=14, no AR KV.

    Expected: stale cache is overwritten. New cache = prompt tokens from current request.
    The stale values (999.x) must NOT appear in the new cache.
    """
    mgr = _make_cache_mgr()

    # Simulate leftover from previous request
    leftover_k, leftover_v = _make_known_kv(9, base=999.0)
    mgr.image_kv_cache_map = (
        leftover_k.reshape(1, 9, NUM_KV_HEADS, HEAD_DIM),
        leftover_v.reshape(1, 9, NUM_KV_HEADS, HEAD_DIM),
    )
    assert mgr._injected_ar_kv is None

    # New request first_step
    bs = 1
    prompt_len = 3
    seq_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(bs * seq_len, base=1.0)

    _call_mgr(
        mgr,
        bs,
        q_len=seq_len,
        seq_len=seq_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    cached_key, cached_value = mgr.image_kv_cache_map
    assert cached_key.shape == (bs, prompt_len, NUM_KV_HEADS, HEAD_DIM)
    # Must be from current request, not stale
    assert torch.allclose(cached_key[0, :prompt_len], k_flat[:prompt_len])
    assert torch.allclose(cached_value[0, :prompt_len], v_flat[:prompt_len])
    # Stale values must not be present
    assert not torch.any(cached_key >= 999.0)


# ============================================================
# Test 5: Paged KV opt-in path
# ============================================================


def test_paged_kv_disabled_by_default_uses_dense_update_path():
    CAPTURED_ATTN_CALLS.clear()
    mgr = _make_cache_mgr()
    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(bs * first_q_len, base=1.0)

    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    assert mgr._paged_prompt_kv_state is None

    update_seq_len = prompt_len + IMAGE_TOKEN_LEN
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    CAPTURED_ATTN_CALLS.clear()
    _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=update_seq_len,
        key_flat=img_k,
        value_flat=img_v,
        first_step=False,
        position_ids=position_ids,
    )

    assert CAPTURED_ATTN_CALLS
    _, dense_key, dense_value, _ = CAPTURED_ATTN_CALLS[-1]
    assert dense_key.shape == (bs, update_seq_len, NUM_HEADS, HEAD_DIM)
    assert dense_value.shape == (bs, update_seq_len, NUM_HEADS, HEAD_DIM)


def test_paged_prompt_state_built_on_first_step():
    mgr = _make_cache_mgr()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunyuanImage3PagedKVCacheManager,
    )

    assert isinstance(mgr._paged_kv_cache_manager, HunyuanImage3PagedKVCacheManager)

    bs = 2
    prompt_len = 5
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    _call_mgr(
        mgr,
        bs,
        q_len=q_len,
        seq_len=q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    state = mgr._paged_prompt_kv_state
    assert state is not None
    assert state.page_size == 4
    assert state.key_cache.shape == (4, 4, NUM_KV_HEADS, HEAD_DIM)
    assert state.prefix_page_indptr.tolist() == [0, 2, 4]
    assert torch.equal(state.cached_lens, torch.full((bs,), prompt_len, dtype=torch.int32))
    for b in range(bs):
        page_start = int(state.prefix_page_indptr[b].item())
        page_end = int(state.prefix_page_indptr[b + 1].item())
        prefix_from_pages = state.key_cache[page_start:page_end].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:prompt_len]
        flat_offset = b * q_len
        assert torch.allclose(prefix_from_pages, k_flat[flat_offset : flat_offset + prompt_len])


def test_paged_update_dispatches_runner_without_dense_reuse():
    CAPTURED_ATTN_CALLS.clear()
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    def _fail_dense_reuse(*args, **kwargs):
        raise AssertionError("_reuse_prompt_kv should not be called on the paged path")

    mgr._reuse_prompt_kv = _fail_dense_reuse
    CAPTURED_ATTN_CALLS.clear()
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    out = _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=prompt_len + IMAGE_TOKEN_LEN,
        key_flat=img_k,
        value_flat=img_v,
        first_step=False,
        position_ids=position_ids,
    )

    assert out.shape == (IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM)
    assert not CAPTURED_ATTN_CALLS
    assert len(runner.calls) == 1
    _, _, _, metadata = runner.calls[0]
    assert metadata.kv_indptr.tolist() == [0, 3]
    assert metadata.kv_last_page_len.tolist() == [3]
    assert metadata.append_batch_indices.tolist() == [0] * IMAGE_TOKEN_LEN
    assert metadata.append_positions.tolist() == list(range(prompt_len, prompt_len + IMAGE_TOKEN_LEN))
    stats = mgr.get_paged_kv_cache_stats()
    assert stats["paged_kv_num_pages"] == 3
    assert stats["paged_kv_batch_size"] == 1
    assert stats["paged_kv_cached_tokens"] == prompt_len
    assert stats["paged_kv_max_cached_tokens"] == prompt_len
    assert stats["paged_cache_expansions"] == 1
    assert stats["paged_attention_calls"] == 1
    assert stats["paged_kv_prefix_page_hits"] == 1
    assert stats["paged_kv_prefix_page_lookups"] == 1
    assert stats["paged_kv_prefix_page_hit_rate"] == 1.0
    assert stats["paged_kv_prefix_token_hits"] == prompt_len
    assert stats["paged_kv_prefix_token_lookups"] == prompt_len


def test_fake_paged_runner_matches_dense_attention_reference():
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    query = torch.randn(bs * IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM)
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    out = _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=prompt_len + IMAGE_TOKEN_LEN,
        key_flat=img_k,
        value_flat=img_v,
        query_flat=query,
        first_step=False,
        position_ids=position_ids,
    )

    cached_key, cached_value = mgr.image_kv_cache_map
    current_key = img_k.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    current_value = img_v.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    dense_key = torch.cat([cached_key, current_key], dim=1)
    dense_value = torch.cat([cached_value, current_value], dim=1)
    expected = _dense_attention(
        query.reshape(bs, IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM),
        dense_key,
        dense_value,
        SCALING,
    )
    assert torch.allclose(out.reshape_as(expected), expected)


def test_paged_update_accepts_boolean_attention_mask_as_custom_mask():
    CAPTURED_ATTN_CALLS.clear()
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    update_seq_len = prompt_len + IMAGE_TOKEN_LEN
    query = torch.randn(bs * IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM)
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    mask = torch.ones(bs, 1, IMAGE_TOKEN_LEN, update_seq_len, dtype=torch.bool)
    mask[..., 0] = False
    CAPTURED_ATTN_CALLS.clear()
    out = _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=update_seq_len,
        key_flat=img_k,
        value_flat=img_v,
        query_flat=query,
        attention_mask=mask,
        first_step=False,
        position_ids=position_ids,
    )

    assert not CAPTURED_ATTN_CALLS
    assert len(runner.calls) == 1
    _, _, _, metadata = runner.calls[0]
    assert metadata.custom_mask is not None
    assert metadata.custom_mask.numel() == IMAGE_TOKEN_LEN * update_seq_len
    assert mgr.get_paged_kv_cache_stats()["paged_attention_custom_mask_calls"] == 1

    cached_key, cached_value = mgr.image_kv_cache_map
    current_key = img_k.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    current_value = img_v.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    dense_key = torch.cat([cached_key, current_key], dim=1)
    dense_value = torch.cat([cached_value, current_value], dim=1)
    expected = _dense_attention(
        query.reshape(bs, IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM),
        dense_key,
        dense_value,
        SCALING,
        attn_mask=mask[:, 0],
    )
    assert torch.allclose(out.reshape_as(expected), expected)


def test_paged_update_matches_ragged_dense_reference_for_non_uniform_cfg_batch():
    CAPTURED_ATTN_CALLS.clear()
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 2
    cached_lens = torch.tensor([3, 5], dtype=torch.long)
    max_prompt_len = int(cached_lens.max().item())
    first_q_len = max_prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(bs * first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=cached_lens.reshape(bs, 1),
    )

    update_seq_len = max_prompt_len + IMAGE_TOKEN_LEN
    query = torch.randn(bs * IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM)
    img_k, img_v = _make_known_kv(bs * IMAGE_TOKEN_LEN, base=50.0)
    position_ids = cached_lens.unsqueeze(1) + torch.arange(IMAGE_TOKEN_LEN).unsqueeze(0)
    mask = torch.ones(bs, 1, IMAGE_TOKEN_LEN, update_seq_len, dtype=torch.bool)
    mask[0, 0, :, 3:5] = False  # Dense padding columns for sample 0.
    mask[0, 0, 0, 5] = False  # First current token after sample 0 prefix.
    mask[1, 0, 1, 4] = False  # Last prefix token for sample 1.
    mask[1, 0, 2, 12] = False  # Last current token for sample 1.

    CAPTURED_ATTN_CALLS.clear()
    out = _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=update_seq_len,
        key_flat=img_k,
        value_flat=img_v,
        query_flat=query,
        attention_mask=mask,
        first_step=False,
        position_ids=position_ids,
    )

    assert not CAPTURED_ATTN_CALLS
    assert len(runner.calls) == 1
    _, _, _, metadata = runner.calls[0]
    assert metadata.cached_lens.tolist() == cached_lens.tolist()
    assert metadata.seq_lens.tolist() == [int(cached_len.item()) + IMAGE_TOKEN_LEN for cached_len in cached_lens]
    assert metadata.custom_mask is not None

    cached_key, cached_value = mgr.image_kv_cache_map
    current_key = img_k.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    current_value = img_v.reshape(bs, IMAGE_TOKEN_LEN, NUM_KV_HEADS, HEAD_DIM)
    query_4d = query.reshape(bs, IMAGE_TOKEN_LEN, NUM_HEADS, HEAD_DIM)
    expected_parts = []
    for b in range(bs):
        cached_len = int(cached_lens[b].item())
        dense_key = torch.cat([cached_key[b, :cached_len], current_key[b]], dim=0).unsqueeze(0)
        dense_value = torch.cat([cached_value[b, :cached_len], current_value[b]], dim=0).unsqueeze(0)
        prefix_mask = mask[b, 0, :, :cached_len]
        current_mask = mask[b, 0, :, max_prompt_len : max_prompt_len + IMAGE_TOKEN_LEN]
        ragged_mask = torch.cat([prefix_mask, current_mask], dim=1).unsqueeze(0)
        expected_parts.append(
            _dense_attention(
                query_4d[b : b + 1],
                dense_key,
                dense_value,
                SCALING,
                attn_mask=ragged_mask,
            )[0]
        )
    expected = torch.stack(expected_parts, dim=0)
    assert torch.allclose(out.reshape_as(expected), expected)
    assert mgr.get_paged_kv_cache_stats()["paged_attention_custom_mask_calls"] == 1


def test_paged_update_falls_back_for_float_attention_mask():
    CAPTURED_ATTN_CALLS.clear()
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    update_seq_len = prompt_len + IMAGE_TOKEN_LEN
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    mask = torch.zeros(bs, 1, update_seq_len, update_seq_len)
    mask[..., 0] = float("-inf")
    CAPTURED_ATTN_CALLS.clear()
    _call_mgr(
        mgr,
        bs,
        q_len=IMAGE_TOKEN_LEN,
        seq_len=update_seq_len,
        key_flat=img_k,
        value_flat=img_v,
        attention_mask=mask,
        first_step=False,
        position_ids=position_ids,
    )

    assert runner.calls == []
    assert CAPTURED_ATTN_CALLS
    _, dense_key, _, _ = CAPTURED_ATTN_CALLS[-1]
    assert dense_key.shape[1] == update_seq_len
    assert mgr.get_paged_kv_cache_stats()["paged_attention_fallbacks"] == 1


def test_paged_update_required_mode_raises_for_float_attention_mask():
    mgr = _make_cache_mgr()
    runner = FakePagedKVRunner()
    mgr.set_paged_kv_cache_enabled(True, required=True)
    mgr.set_paged_kv_cache_page_size(4)
    mgr.set_paged_kv_runner(runner)

    bs = 1
    prompt_len = 3
    first_q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    k_flat, v_flat = _make_known_kv(first_q_len, base=1.0)
    _call_mgr(
        mgr,
        bs,
        q_len=first_q_len,
        seq_len=first_q_len,
        key_flat=k_flat,
        value_flat=v_flat,
        first_step=True,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    update_seq_len = prompt_len + IMAGE_TOKEN_LEN
    img_k, img_v = _make_known_kv(IMAGE_TOKEN_LEN, base=50.0)
    position_ids = torch.arange(prompt_len, prompt_len + IMAGE_TOKEN_LEN).reshape(1, IMAGE_TOKEN_LEN)
    mask = torch.zeros(bs, 1, update_seq_len, update_seq_len)
    mask[..., 0] = float("-inf")

    with pytest.raises(RuntimeError, match="metadata build failed"):
        _call_mgr(
            mgr,
            bs,
            q_len=IMAGE_TOKEN_LEN,
            seq_len=update_seq_len,
            key_flat=img_k,
            value_flat=img_v,
            attention_mask=mask,
            first_step=False,
            position_ids=position_ids,
        )

    assert runner.calls == []
    assert mgr.get_paged_kv_cache_stats()["paged_attention_fallbacks"] == 1
