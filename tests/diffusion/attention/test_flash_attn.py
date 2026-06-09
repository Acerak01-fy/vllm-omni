# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test script for FlashAttention backend with padding handling.

This script tests two main scenarios:
1. Case 1: Comparing padded vs unpadded inputs for batch_size=1
2. Case 2: Comparing FlashAttention and SDPA backends for batch_size=2 with padding
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.backends.utils import fa  # noqa: E402
from vllm_omni.platforms import current_omni_platform

is_gpu = current_omni_platform.is_cuda_alike() or current_omni_platform.is_xpu()
HAS_FLASH_ATTN = fa.HAS_FLASH_ATTN
flash_attn_func = fa.flash_attn_func  # noqa: N813


def create_attention_mask(batch_size: int, seq_len: int, valid_len: int, device: torch.device) -> torch.Tensor:
    """
    Create attention mask where first valid_len tokens are valid (1) and rest are padding (0).

    Args:
        batch_size: Batch size
        seq_len: Total sequence length (including padding)
        valid_len: Number of valid (non-padded) tokens

    Returns:
        Attention mask of shape (batch_size, seq_len)
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[:, :valid_len] = True
    return mask


def pad_tensor(tensor: torch.Tensor, target_seq_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad tensor along sequence dimension (dim=1).

    Args:
        tensor: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        target_seq_len: Target sequence length after padding
        pad_value: Value to use for padding

    Returns:
        Padded tensor of shape (batch_size, target_seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, num_heads, head_dim = tensor.shape
    if target_seq_len <= seq_len:
        return tensor

    padding = torch.full(
        (batch_size, target_seq_len - seq_len, num_heads, head_dim), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=1)


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_padding_equivalence():
    """
    Case 1: Test that padded and unpadded inputs produce similar outputs.

    - Input A: batch_size=1, hidden_states (1, 48), encoder_hidden_states (1, 16)
      Concatenated length: 64, NO attention_mask
    - Input B: Same data but padded: hidden_states (1, 58), encoder_hidden_states (1, 26)
      Concatenated length: 84, WITH attention_mask

    Expected: Output A and Output B should be very close.
    """
    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    # Configuration
    batch_size = 1
    hidden_seq_len = 48
    encoder_seq_len = 16
    pad_length = 10
    num_heads = 8
    head_dim = 64

    # Initialize FlashAttention
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    # Create base tensors with random values (same for both A and B)
    torch.manual_seed(42)
    hidden_states_base = torch.randn(batch_size, hidden_seq_len, num_heads, head_dim, device=device, dtype=dtype)
    encoder_hidden_states_base = torch.randn(
        batch_size, encoder_seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # ========== Input A: Unpadded, no attention mask ==========
    query_a = torch.cat([hidden_states_base, encoder_hidden_states_base], dim=1)
    key_a = query_a.clone()
    value_a = query_a.clone()

    attn_metadata_a = AttentionMetadata(attn_mask=None)

    output_a = fa_impl.forward(query=query_a, key=key_a, value=value_a, attn_metadata=attn_metadata_a)

    # ========== Input B: Padded with attention mask ==========
    hidden_states_padded = pad_tensor(hidden_states_base, hidden_seq_len + pad_length)
    encoder_hidden_states_padded = pad_tensor(encoder_hidden_states_base, encoder_seq_len + pad_length)

    query_b = torch.cat([hidden_states_padded, encoder_hidden_states_padded], dim=1)
    key_b = query_b.clone()
    value_b = query_b.clone()

    # Create attention mask
    attn_mask_b = torch.cat(
        [
            create_attention_mask(batch_size, hidden_seq_len + pad_length, hidden_seq_len, device),
            create_attention_mask(batch_size, encoder_seq_len + pad_length, encoder_seq_len, device),
        ],
        dim=1,
    )

    attn_metadata_b = AttentionMetadata(attn_mask=attn_mask_b)

    output_b = fa_impl.forward(query=query_b, key=key_b, value=value_b, attn_metadata=attn_metadata_b)

    # Extract non-padded portion from output_b
    output_b_unpadded = torch.cat(
        [
            output_b[:, :hidden_seq_len, :, :],
            output_b[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )

    # Compare outputs
    max_diff = torch.max(torch.abs(output_a - output_b_unpadded)).item()
    mean_diff = torch.mean(torch.abs(output_a - output_b_unpadded)).item()

    print("\n=== Case 1: Padding Equivalence Test ===")
    print(f"Output A shape: {output_a.shape}")
    print(f"Output B shape: {output_b.shape}")
    print(f"Output B unpadded shape: {output_b_unpadded.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Assert that outputs are close
    # Using higher tolerance for bfloat16
    assert max_diff < 0.1, f"Max difference {max_diff} exceeds threshold 0.1"
    assert mean_diff < 0.01, f"Mean difference {mean_diff} exceeds threshold 0.01"

    print("✓ Case 1 PASSED: Padded and unpadded outputs are very close!")


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_fa_vs_sdpa():
    """
    Case 2: Compare FlashAttention and SDPA backends with padding.

    - batch_size=2
    - hidden_states: (2, 48) padded to (2, 58)
    - encoder_hidden_states: (2, 16) padded to (2, 26)
    - Concatenated length: 84
    - Compare FA and SDPA outputs

    Expected: FA and SDPA outputs should be very close.
    """
    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    # Configuration
    batch_size = 2
    hidden_seq_len = 48
    encoder_seq_len = 16
    pad_length = 10
    num_heads = 8
    head_dim = 64

    # Initialize both backends
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    sdpa_impl = SDPAImpl(num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False)

    # Create base tensors
    torch.manual_seed(123)
    hidden_states_base = torch.randn(batch_size, hidden_seq_len, num_heads, head_dim, device=device, dtype=dtype)
    encoder_hidden_states_base = torch.randn(
        batch_size, encoder_seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Pad tensors
    hidden_states_padded = pad_tensor(hidden_states_base, hidden_seq_len + pad_length)
    encoder_hidden_states_padded = pad_tensor(encoder_hidden_states_base, encoder_seq_len + pad_length)

    # Concatenate
    query = torch.cat([hidden_states_padded, encoder_hidden_states_padded], dim=1)
    key = query.clone()
    value = query.clone()

    # Create attention mask
    attn_mask = torch.cat(
        [
            create_attention_mask(batch_size, hidden_seq_len + pad_length, hidden_seq_len, device),
            create_attention_mask(batch_size, encoder_seq_len + pad_length, encoder_seq_len, device),
        ],
        dim=1,
    )

    attn_metadata = AttentionMetadata(attn_mask=attn_mask)

    # Run FlashAttention
    output_fa = fa_impl.forward(query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata)

    # Run SDPA
    # SDPA expects 4D attention mask: (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len)
    # For causal=False, we need to convert 2D mask to 4D
    if attn_mask is not None:
        # Expand mask for SDPA: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attn_mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        # Convert bool to float: True -> 0.0, False -> -inf
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=dtype)
        attn_mask_float.masked_fill_(~attn_mask_4d, float("-inf"))
        attn_metadata_sdpa = AttentionMetadata(attn_mask=attn_mask_float)
    else:
        attn_metadata_sdpa = AttentionMetadata(attn_mask=None)

    output_sdpa = sdpa_impl.forward(
        query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata_sdpa
    )

    # Compare outputs (only compare valid regions)
    output_fa_valid = torch.cat(
        [
            output_fa[:, :hidden_seq_len, :, :],
            output_fa[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )
    output_sdpa_valid = torch.cat(
        [
            output_sdpa[:, :hidden_seq_len, :, :],
            output_sdpa[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )

    max_diff = torch.max(torch.abs(output_fa_valid - output_sdpa_valid)).item()
    mean_diff = torch.mean(torch.abs(output_fa_valid - output_sdpa_valid)).item()

    print("\n=== Case 2: FA vs SDPA Comparison ===")
    print(f"Batch size: {batch_size}")
    print(f"FA output shape: {output_fa.shape}")
    print(f"SDPA output shape: {output_sdpa.shape}")
    print(f"Max absolute difference (valid region): {max_diff:.6f}")
    print(f"Mean absolute difference (valid region): {mean_diff:.6f}")

    # Assert that outputs are close
    # Using higher tolerance for bfloat16 and different implementations
    assert max_diff < 0.01, f"Max difference {max_diff} exceeds threshold 0.01"
    assert mean_diff < 0.001, f"Mean difference {mean_diff} exceeds threshold 0.001"

    print("✓ Case 2 PASSED: FA and SDPA outputs are very close!")


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_flash_attn_func_preferred_over_varlen():
    """Test flash_attn_func availability and basic forward call."""
    if not HAS_FLASH_ATTN:
        pytest.skip("No Flash Attention available")

    if flash_attn_func is None:
        pytest.skip("flash_attn_func not available, will use varlen fallback")

    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    num_heads, head_dim = 8, 64
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    torch.manual_seed(42)
    q = torch.randn(1, 32, num_heads, head_dim, device=device, dtype=dtype)
    k = q.clone()
    v = q.clone()

    attn_metadata = AttentionMetadata(attn_mask=None)
    output = fa_impl.forward(q, k, v, attn_metadata)

    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print("✓ flash_attn_func forward works correctly!")


def _sdpa_segment(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, softmax_scale: float):
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    attn_mask = None
    if causal:
        sq, sk = q_.shape[-2], k_.shape[-2]
        i = torch.arange(sq, device=q.device).unsqueeze(1)
        j = torch.arange(sk, device=q.device).unsqueeze(0)
        attn_mask = j <= (i + (sk - sq))
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def test_piecewise_uses_varlen_when_flash_attn_func_missing(monkeypatch):
    """The CUDA piecewise path must work when only flash_attn_varlen_func exists."""

    def fake_varlen_func(
        q,
        k,
        v,
        *,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        softmax_scale=None,
        **kwargs,
    ):
        del max_seqlen_q, max_seqlen_k, kwargs
        outputs = []
        for idx in range(cu_seqlens_q.numel() - 1):
            q_start = int(cu_seqlens_q[idx].item())
            q_end = int(cu_seqlens_q[idx + 1].item())
            k_start = int(cu_seqlens_k[idx].item())
            k_end = int(cu_seqlens_k[idx + 1].item())
            outputs.append(
                _sdpa_segment(
                    q[q_start:q_end].unsqueeze(0),
                    k[k_start:k_end].unsqueeze(0),
                    v[k_start:k_end].unsqueeze(0),
                    causal=causal,
                    softmax_scale=softmax_scale,
                ).squeeze(0)
            )
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "flash_attn_func", None)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", fake_varlen_func)

    torch.manual_seed(7)
    batch_size, num_heads, head_dim = 1, 2, 16
    key_len = 10
    query_start = 4
    query_len = key_len - query_start
    query = torch.randn(batch_size, query_len, num_heads, head_dim)
    key = torch.randn(batch_size, key_len, num_heads, head_dim)
    value = torch.randn(batch_size, key_len, num_heads, head_dim)
    softmax_scale = 1.0 / (head_dim**0.5)
    full_attn_spans = [[(5, 8)]]
    impl = FlashAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        softmax_scale=softmax_scale,
        causal=False,
    )

    got = impl.forward_cuda(
        query,
        key,
        value,
        AttentionMetadata(full_attn_spans=full_attn_spans),
    )

    causal_part = _sdpa_segment(
        query[:, :1],
        key[:, :5],
        value[:, :5],
        causal=True,
        softmax_scale=softmax_scale,
    )
    full_part = _sdpa_segment(
        query[:, 1:4],
        key[:, :8],
        value[:, :8],
        causal=False,
        softmax_scale=softmax_scale,
    )
    causal_tail = _sdpa_segment(
        query[:, 4:],
        key,
        value,
        causal=True,
        softmax_scale=softmax_scale,
    )
    expected = torch.cat([causal_part, full_part, causal_tail], dim=1)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    print("Running FlashAttention Padding Tests...")
    print("=" * 60)

    # Try to run tests
    if is_gpu:
        try:
            print("\n[Running Case 1: Padding Equivalence for FA]")
            test_padding_equivalence()
        except Exception as e:
            print(f"✗ Case 1 failed: {e}")
            import traceback

            traceback.print_exc()

        try:
            print("\n[Running Case 2: FA vs SDPA]")
            test_fa_vs_sdpa()
        except Exception as e:
            print(f"✗ Case 2 failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        raise RuntimeError("CUDA is not available")
    print("\n" + "=" * 60)
    print("Test suite completed!")
