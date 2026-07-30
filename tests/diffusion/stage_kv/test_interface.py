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

REQUEST_LAYOUT_DIGEST = "request-layout-v1"


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


def make_metadata(
    *,
    request_id: str = "req-0",
    request_layout_digest: str = REQUEST_LAYOUT_DIGEST,
) -> StageKVMetadata:
    return StageKVMetadata(
        request_id=request_id,
        allocation_id=1,
        cache_mode=StageKVCacheMode.PAGED_SCHEDULER,
        cache_layout_fingerprint="layout-v1",
        request_layout_digest=request_layout_digest,
        branches=(
            StageKVBranchMetadata(
                branch_id=0,
                block_ids=((1, 2),),
                stable_len=4,
                current_len=2,
                seq_len=6,
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


@pytest.mark.parametrize(
    ("tensor_changes", "error_match"),
    [
        ({"size": 1}, "tensor size mismatch"),
        ({"offset": 123}, "requires zero tensor offset"),
        ({"block_stride": 456}, "requires unpacked tensors"),
    ],
)
def test_worker_rejects_incompatible_physical_tensor_geometry(
    tensor_changes: dict[str, int],
    error_match: str,
) -> None:
    config = make_init_config()
    tensor = replace(
        config.kv_cache_config.kv_cache_tensors[0],
        **tensor_changes,
    )
    config = replace(
        config,
        kv_cache_config=replace(
            config.kv_cache_config,
            kv_cache_tensors=[tensor],
        ),
    )

    with pytest.raises(ValueError, match=error_match):
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
        expected_request_layout_digest=REQUEST_LAYOUT_DIGEST,
    )


def test_request_metadata_uses_planner_seq_len_for_block_count() -> None:
    metadata = replace(
        make_metadata(),
        branches=(
            StageKVBranchMetadata(
                branch_id=0,
                block_ids=((1, 2),),
                stable_len=3,
                current_len=1,
                seq_len=5,
            ),
        ),
    )

    validate_stage_kv_metadata(
        metadata,
        config=make_init_config(),
        request_id="req-0",
        expected_request_layout_digest=REQUEST_LAYOUT_DIGEST,
    )


@pytest.mark.parametrize(
    ("branch_changes", "error_match"),
    [
        ({"seq_len": 0}, "seq_len must be positive"),
        ({"seq_len": 5}, r"stable_len \+ current_len must not exceed seq_len"),
    ],
)
def test_branch_metadata_rejects_invalid_seq_len(
    branch_changes: dict[str, int],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        replace(make_metadata().branches[0], **branch_changes)


def test_request_metadata_rejects_request_layout_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="expected request layout digest"):
        validate_stage_kv_metadata(
            make_metadata(),
            config=make_init_config(),
            request_id="req-0",
            expected_request_layout_digest="current-request-layout",
        )


def test_request_metadata_rejects_request_mismatch() -> None:
    config = make_init_config()
    with pytest.raises(ValueError, match="request mismatch"):
        validate_stage_kv_metadata(
            make_metadata(request_id="other"),
            config=config,
            request_id="req-0",
            expected_request_layout_digest=REQUEST_LAYOUT_DIGEST,
        )


@pytest.mark.parametrize("block_ids", [((0, 1),), ((7, 8),)])
def test_request_metadata_rejects_reserved_and_out_of_range_block_ids(
    block_ids: tuple[tuple[int, ...], ...],
) -> None:
    config = make_init_config()
    metadata = replace(
        make_metadata(),
        branches=(
            StageKVBranchMetadata(
                branch_id=0,
                block_ids=block_ids,
                stable_len=4,
                current_len=2,
                seq_len=6,
            ),
        ),
    )
    with pytest.raises(ValueError, match="reserved or out-of-range block id"):
        validate_stage_kv_metadata(
            metadata,
            config=config,
            request_id="req-0",
            expected_request_layout_digest=REQUEST_LAYOUT_DIGEST,
        )
