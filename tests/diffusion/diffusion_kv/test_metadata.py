# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVContextMetadata,
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_diffusion_kv_metadata_uses_native_cache_group_block_ids() -> None:
    context = DiffusionKVContextMetadata(
        context_id="text",
        cache_role="cross_attention",
        num_tokens=3,
        block_ids=([7], [11]),
    )
    sequence = DiffusionKVSequenceMetadata(
        sequence_id=1,
        prefix_len=4,
        target_len=2,
        seq_len=8,
        block_ids=([1, 2], [5, 6]),
        contexts=(context,),
    )
    metadata = DiffusionKVMetadata(
        request_id="req-0",
        allocation_generation=3,
        sequences=(sequence,),
    )

    assert metadata.sequences[0].block_ids == ([1, 2], [5, 6])
    assert metadata.sequences[0].contexts[0].block_ids == ([7], [11])
