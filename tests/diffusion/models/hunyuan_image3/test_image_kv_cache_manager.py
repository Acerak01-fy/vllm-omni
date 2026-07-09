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


# ============================================================
# Mocks + helpers
# ============================================================


class MockAttention(nn.Module):
    def __init__(self, num_heads, head_size, causal=False, softmax_scale=None, num_kv_heads=None, **kwargs):
        super().__init__()

    def forward(self, query, key, value, attn_metadata=None, **kwargs):
        return query


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


def _gen_timestep_index(bs: int, current_start: int) -> torch.Tensor:
    return torch.full((bs, 1), current_start, dtype=torch.long)


def _repeat_kv_local(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * SCALING
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def test_paged_kv_tensor_identity_key_handles_inference_tensor():
    from vllm_omni.diffusion.models.hunyuan_image3.paged_kv import _tensor_identity_key

    with torch.inference_mode():
        tensor = torch.zeros(2, 3)

    key = _tensor_identity_key(tensor)

    assert key is not None
    assert key[-1] == 0


def _call_mgr(
    mgr,
    bs,
    q_len,
    seq_len,
    key_flat,
    value_flat,
    first_step=False,
    uncond_cfg_prefill=False,
    num_image_tokens=IMAGE_TOKEN_LEN,
    shard_image_size=None,
    gen_timestep_scatter_index=None,
    position_ids=None,
):
    query = torch.randn(bs * q_len, NUM_HEADS, HEAD_DIM)
    attn_mask = torch.zeros(bs, 1, seq_len, seq_len)
    mgr(
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


def test_paged_prompt_kv_attention_matches_dense_concat(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", "2")
    mgr = _make_cache_mgr()
    assert mgr.paged_prompt_kv_enabled
    assert not mgr.paged_prompt_kv_required

    bs = 2
    prompt_len = 4
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    repeat_num = NUM_HEADS // NUM_KV_HEADS
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)
    key_4d = k_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM)
    value_4d = v_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM)

    mgr._cache_prompt_kv(
        key_4d,
        value_4d,
        q_len,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
        cache_repeat_num=repeat_num,
    )
    assert mgr.get_paged_prompt_kv_stats()["paged_cache_builds"] == 1
    assert mgr.image_kv_cache_map is None

    img_q_len = IMAGE_TOKEN_LEN
    query = torch.randn(bs, img_q_len, NUM_HEADS, HEAD_DIM)
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)
    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    value_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    seq_len = prompt_len + img_q_len

    output = mgr._run_paged_prompt_kv_attention(
        query,
        key_input,
        value_input,
        seq_len,
        bs,
        repeat_num,
        attention_mask=torch.zeros(bs, 1, img_q_len, seq_len),
        full_attn_spans=[[(prompt_len, seq_len)] for _ in range(bs)],
        position_ids=torch.arange(prompt_len, prompt_len + img_q_len).repeat(bs, 1),
    )

    page_batch = mgr._paged_prompt_kv.current_batch
    assert page_batch is not None
    cached_key, cached_value, _ = mgr._paged_prompt_kv.materialize_rows(page_batch.row_refs)
    assert cached_key.shape[2] == NUM_KV_HEADS
    expected_key = torch.cat(
        [_repeat_kv_local(cached_key, repeat_num), _repeat_kv_local(key_input, repeat_num)],
        dim=1,
    )
    expected_value = torch.cat(
        [_repeat_kv_local(cached_value, repeat_num), _repeat_kv_local(value_input, repeat_num)],
        dim=1,
    )
    expected = _dense_attention(query, expected_key, expected_value)

    assert output is not None
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    assert mgr.get_paged_prompt_kv_stats()["paged_attention_calls"] == 1


def test_paged_reuse_write_uses_host_block_rows(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", "2")
    mgr = _make_cache_mgr()

    bs = 2
    prompt_len = 4
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    repeat_num = NUM_HEADS // NUM_KV_HEADS
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)
    mgr._cache_prompt_kv(
        k_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM),
        v_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM),
        q_len,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
        cache_repeat_num=repeat_num,
    )

    img_q_len = IMAGE_TOKEN_LEN
    seq_len = prompt_len + img_q_len
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)
    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    value_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)

    pool = mgr._paged_prompt_kv
    inputs = pool._build_attention_inputs(key_input, seq_len, attention_mask=None)
    assert len(inputs.block_rows) == bs
    assert all(isinstance(row, tuple) for row in inputs.block_rows)

    from vllm_omni.diffusion.models.hunyuan_image3 import paged_kv as paged_kv_mod

    original_compute_slot_mapping = paged_kv_mod.compute_slot_mapping
    seen_block_rows = []

    def capture_compute_slot_mapping(block_ids, positions, block_size):
        seen_block_rows.append(block_ids)
        assert isinstance(block_ids, tuple)
        return original_compute_slot_mapping(block_ids, positions, block_size)

    monkeypatch.setattr(
        pool,
        "_build_attention_inputs",
        lambda key, seq_len, attention_mask=None: inputs,
    )
    monkeypatch.setattr(paged_kv_mod, "compute_slot_mapping", capture_compute_slot_mapping)
    monkeypatch.setattr(
        pool,
        "_run_attention_from_inputs",
        lambda query, inputs, *, softmax_scale: query,
    )

    query = torch.randn(bs, img_q_len, NUM_HEADS, HEAD_DIM)
    output = pool.run_paged_attention(
        query,
        key_input,
        value_input,
        seq_len=seq_len,
        softmax_scale=SCALING,
        attention_mask=None,
    )

    torch.testing.assert_close(output, query)
    assert seen_block_rows == list(inputs.block_rows)


def test_first_step_paged_cache_capture_uses_dense_attention_with_ar_prefix(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", "2")
    mgr = _make_cache_mgr()

    captured = {}

    def capture_attention(query, key, value, attn_metadata=None, **kwargs):
        captured["key"] = key.clone()
        captured["value"] = value.clone()
        return query

    mgr.attn.forward = capture_attention

    bs = 1
    ar_len = 3
    prompt_len = 2
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    seq_len = ar_len + q_len
    repeat_num = NUM_HEADS // NUM_KV_HEADS

    ar_k, ar_v = _make_known_kv(ar_len, base=100.0)
    mgr._injected_ar_kv = [(ar_k.clone(), ar_v.clone())]
    k_flat, v_flat = _make_known_kv(q_len, base=1.0)
    query = torch.randn(bs * q_len, NUM_HEADS, HEAD_DIM)
    attention_mask = torch.ones(bs, 1, q_len, seq_len, dtype=torch.bool)

    output = mgr(
        query,
        k_flat,
        v_flat,
        attention_mask,
        query_lens=[q_len],
        seq_lens=[seq_len],
        first_step=True,
        num_image_tokens=IMAGE_TOKEN_LEN,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
    )

    full_key = torch.cat(
        [ar_k.reshape(1, ar_len, NUM_KV_HEADS, HEAD_DIM), k_flat.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)],
        dim=1,
    )
    full_value = torch.cat(
        [ar_v.reshape(1, ar_len, NUM_KV_HEADS, HEAD_DIM), v_flat.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)],
        dim=1,
    )
    torch.testing.assert_close(
        output.reshape(bs, q_len, NUM_HEADS, HEAD_DIM),
        query.reshape(bs, q_len, NUM_HEADS, HEAD_DIM),
    )
    torch.testing.assert_close(captured["key"], _repeat_kv_local(full_key, repeat_num))
    torch.testing.assert_close(captured["value"], _repeat_kv_local(full_value, repeat_num))
    assert mgr.image_kv_cache_map is None
    assert torch.equal(mgr.image_kv_cache_lens, torch.tensor([ar_len + prompt_len]))
    page_batch = mgr._paged_prompt_kv.current_batch
    assert page_batch is not None
    cached_key, cached_value, lens = mgr._paged_prompt_kv.materialize_rows(page_batch.row_refs)
    assert torch.equal(lens, torch.tensor([ar_len + prompt_len]))
    torch.testing.assert_close(cached_key[0, :ar_len], ar_k)
    torch.testing.assert_close(cached_value[0, ar_len : ar_len + prompt_len], v_flat[:prompt_len])
    stats = mgr.get_paged_prompt_kv_stats()
    assert stats["paged_cache_builds"] == 1
    assert stats["paged_attention_calls"] == 0


def test_first_step_paged_cache_capture_skips_paged_runner_with_boolean_custom_mask(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", "2")
    mgr = _make_cache_mgr()

    class FakeRunner:
        def run(self, query, key_cache, value_cache, inputs, *, softmax_scale):
            raise AssertionError("first-step dense attention must not dispatch paged attention")

    mgr._paged_prompt_kv._flashinfer_runner = FakeRunner()

    bs = 1
    prompt_len = 3
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    seq_len = q_len
    k_flat, v_flat = _make_known_kv(q_len, base=1.0)
    query = torch.randn(bs * q_len, NUM_HEADS, HEAD_DIM)
    attention_mask = torch.ones(bs, 1, q_len, seq_len, dtype=torch.bool)
    attention_mask[0, 0, 0, 0] = False

    output = mgr(
        query,
        k_flat,
        v_flat,
        attention_mask,
        query_lens=[q_len],
        seq_lens=[seq_len],
        first_step=True,
        num_image_tokens=IMAGE_TOKEN_LEN,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
        full_attn_spans=[[(prompt_len, prompt_len + IMAGE_TOKEN_LEN)]],
    )

    torch.testing.assert_close(
        output.reshape(bs, q_len, NUM_HEADS, HEAD_DIM),
        query.reshape(bs, q_len, NUM_HEADS, HEAD_DIM),
    )
    assert mgr.image_kv_cache_map is None
    page_batch = mgr._paged_prompt_kv.current_batch
    assert page_batch is not None
    _, _, lens = mgr._paged_prompt_kv.materialize_rows(page_batch.row_refs)
    assert torch.equal(lens, torch.tensor([prompt_len]))
    stats = mgr.get_paged_prompt_kv_stats()
    assert stats["paged_cache_builds"] == 1
    assert stats["paged_attention_calls"] == 0
    assert stats["paged_attention_custom_mask_calls"] == 0


def test_paged_prompt_kv_unsupported_mask_raises_without_dense_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_HY3_PAGED_KV_CACHE_PAGE_SIZE", "2")
    mgr = _make_cache_mgr()

    bs = 1
    prompt_len = 4
    q_len = prompt_len + IMAGE_TOKEN_LEN + SUFFIX_TOKEN_LEN
    repeat_num = NUM_HEADS // NUM_KV_HEADS
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    mgr._cache_prompt_kv(
        k_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM),
        v_flat.reshape(bs, q_len, NUM_KV_HEADS, HEAD_DIM),
        q_len,
        gen_timestep_scatter_index=_gen_timestep_index(bs, prompt_len),
        cache_repeat_num=repeat_num,
    )
    assert mgr.image_kv_cache_map is None

    img_q_len = IMAGE_TOKEN_LEN
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)
    attention_mask = torch.zeros(bs, 1, img_q_len, prompt_len + img_q_len)
    attention_mask[:, :, :, 0] = float("-inf")

    with pytest.raises(RuntimeError, match="non-boolean custom attention masks"):
        mgr._run_paged_prompt_kv_attention(
            torch.randn(bs, img_q_len, NUM_HEADS, HEAD_DIM),
            new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM),
            new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM),
            prompt_len + img_q_len,
            bs,
            repeat_num,
            attention_mask=attention_mask,
            full_attn_spans=[[(prompt_len, prompt_len + img_q_len)]],
            position_ids=torch.arange(prompt_len, prompt_len + img_q_len).reshape(bs, img_q_len),
        )


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
