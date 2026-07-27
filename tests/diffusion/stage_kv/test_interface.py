# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from multiprocessing.reduction import ForkingPickler

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_omni.diffusion.stage_kv.interface import (
    BLOCK_KV_TOKEN_HEAD_DIM_V1,
    StageKVBranchMetadata,
    StageKVCacheMode,
    StageKVMetadata,
    StageKVPhysicalLayout,
    StageKVWorkerInitConfig,
    StageKVWorkerInitResult,
    validate_stage_kv_metadata,
    validate_stage_kv_worker_init_config,
    validate_stage_kv_worker_init_result,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def make_kv_cache_config(*, num_blocks: int = 8) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                shared_by=["layers.0.attn"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0.attn"],
                kv_cache_spec=spec,
            )
        ],
    )


def make_init_config(*, num_blocks: int = 8) -> StageKVWorkerInitConfig:
    return StageKVWorkerInitConfig(
        cache_mode=StageKVCacheMode.PAGED_SCHEDULER,
        cache_layout_fingerprint="layout-v1",
        kv_cache_config=make_kv_cache_config(num_blocks=num_blocks),
        physical_layout=BLOCK_KV_TOKEN_HEAD_DIM_V1,
    )


def make_metadata(*, request_id: str = "req-0") -> StageKVMetadata:
    return StageKVMetadata(
        request_id=request_id,
        allocation_id=1,
        cache_mode=StageKVCacheMode.PAGED_SCHEDULER,
        cache_layout_fingerprint="layout-v1",
        request_layout_digest="request-layout-v1",
        branches=(
            StageKVBranchMetadata(
                branch_id=0,
                block_ids=((1, 2),),
                stable_len=4,
                current_len=2,
            ),
        ),
    )


def test_native_worker_init_contract_and_result_match() -> None:
    config = make_init_config()
    validate_stage_kv_worker_init_config(config)

    result = StageKVWorkerInitResult(
        cache_mode=config.cache_mode,
        cache_layout_fingerprint=config.cache_layout_fingerprint,
        num_blocks=config.kv_cache_config.num_blocks,
        physical_layout=config.physical_layout,
    )
    validate_stage_kv_worker_init_result(config, result)


def test_worker_init_contract_survives_rpc_serialization() -> None:
    config = make_init_config()

    restored = ForkingPickler.loads(ForkingPickler.dumps(config))

    assert restored == config
    validate_stage_kv_worker_init_config(restored)


def test_worker_rejects_unknown_physical_layout() -> None:
    config = replace(
        make_init_config(),
        physical_layout=StageKVPhysicalLayout(version=2, name="future-layout"),
    )

    with pytest.raises(ValueError, match="Unsupported Stage KV physical layout"):
        validate_stage_kv_worker_init_config(config)


def test_worker_rejects_init_result_mismatch() -> None:
    config = make_init_config()
    result = StageKVWorkerInitResult(
        cache_mode=config.cache_mode,
        cache_layout_fingerprint=config.cache_layout_fingerprint,
        num_blocks=config.kv_cache_config.num_blocks + 1,
        physical_layout=config.physical_layout,
    )

    with pytest.raises(ValueError, match="num_blocks mismatch"):
        validate_stage_kv_worker_init_result(config, result)


def test_request_metadata_matches_native_group_geometry() -> None:
    validate_stage_kv_metadata(
        make_metadata(),
        config=make_init_config(),
        request_id="req-0",
    )


def test_request_metadata_rejects_request_and_block_range_mismatch() -> None:
    config = make_init_config()
    with pytest.raises(ValueError, match="request mismatch"):
        validate_stage_kv_metadata(
            make_metadata(request_id="other"),
            config=config,
            request_id="req-0",
        )

    metadata = replace(
        make_metadata(),
        branches=(
            StageKVBranchMetadata(
                branch_id=0,
                block_ids=((7, 8),),
                stable_len=4,
                current_len=2,
            ),
        ),
    )
    with pytest.raises(ValueError, match="out-of-range block id"):
        validate_stage_kv_metadata(
            metadata,
            config=config,
            request_id="req-0",
        )
