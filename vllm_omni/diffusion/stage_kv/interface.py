# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig


class StageKVCacheMode(str, Enum):
    """Migration mode for diffusion-stage KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    PAGED_WORKER_LOCAL = "paged_worker_local"
    PAGED_SCHEDULER = "paged_scheduler"


@dataclass(frozen=True)
class StageKVPhysicalLayout:
    """Versioned physical Tensor layout shared by Scheduler and Worker."""

    version: int
    name: str

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError(f"physical layout version must be positive, got {self.version}")
        if not self.name:
            raise ValueError("physical layout name must be non-empty")


BLOCK_KV_TOKEN_HEAD_DIM_V1 = StageKVPhysicalLayout(
    version=1,
    name="BLOCK_KV_TOKEN_HEAD_DIM_V1",
)
"""``[num_blocks, 2, block_size, num_kv_heads, head_size]``."""


@dataclass(frozen=True)
class StageKVBranchMetadata:
    """Scheduler allocation for one request-local execution branch.

    ``seq_len`` is copied from the Planner requirement and is the block-count
    allocation basis. It may exceed ``stable_len + current_len``.
    """

    branch_id: int
    block_ids: tuple[tuple[int, ...], ...]
    stable_len: int
    current_len: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError(f"branch_id must be non-negative, got {self.branch_id}")
        if not self.block_ids or any(not group for group in self.block_ids):
            raise ValueError("every KV cache group must contain at least one block")
        if self.stable_len < 0:
            raise ValueError(f"stable_len must be non-negative, got {self.stable_len}")
        if self.current_len <= 0:
            raise ValueError(f"current_len must be positive, got {self.current_len}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stable_len + self.current_len > self.seq_len:
            raise ValueError(
                "stable_len + current_len must not exceed seq_len: "
                f"stable_len={self.stable_len}, current_len={self.current_len}, seq_len={self.seq_len}"
            )


@dataclass(frozen=True)
class StageKVMetadata:
    """Serializable Scheduler-to-Worker allocation for a new request."""

    request_id: str
    allocation_id: int
    cache_mode: StageKVCacheMode
    cache_layout_fingerprint: str
    request_layout_digest: str
    branches: tuple[StageKVBranchMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cache_mode, StageKVCacheMode):
            raise TypeError("cache_mode must be StageKVCacheMode")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.allocation_id <= 0:
            raise ValueError(f"allocation_id must be positive, got {self.allocation_id}")
        if self.cache_mode is not StageKVCacheMode.PAGED_SCHEDULER:
            raise ValueError(f"StageKVMetadata requires paged_scheduler mode, got {self.cache_mode.value!r}")
        if not self.cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        if not self.request_layout_digest:
            raise ValueError("request_layout_digest must be non-empty")
        if not self.branches:
            raise ValueError("at least one branch metadata entry is required")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError(f"branch IDs must be unique, got {branch_ids}")


@dataclass(frozen=True)
class StageKVWorkerInitConfig:
    """Exact native cache config broadcast to every diffusion Worker."""

    cache_mode: StageKVCacheMode
    cache_layout_fingerprint: str
    kv_cache_config: KVCacheConfig
    physical_layout: StageKVPhysicalLayout

    def __post_init__(self) -> None:
        if not isinstance(self.cache_mode, StageKVCacheMode):
            raise TypeError("cache_mode must be StageKVCacheMode")
        if self.cache_mode is not StageKVCacheMode.PAGED_SCHEDULER:
            raise ValueError(f"StageKVWorkerInitConfig requires paged_scheduler mode, got {self.cache_mode.value!r}")
        if not self.cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        if not isinstance(self.kv_cache_config, KVCacheConfig):
            raise TypeError("kv_cache_config must be a native vLLM KVCacheConfig")
        if not isinstance(self.physical_layout, StageKVPhysicalLayout):
            raise TypeError("physical_layout must be StageKVPhysicalLayout")


@dataclass(frozen=True)
class StageKVWorkerInitResult:
    """Worker acknowledgement for Stage KV contract initialization."""

    cache_mode: StageKVCacheMode
    cache_layout_fingerprint: str
    num_blocks: int
    physical_layout: StageKVPhysicalLayout

    def __post_init__(self) -> None:
        if not isinstance(self.cache_mode, StageKVCacheMode):
            raise TypeError("cache_mode must be StageKVCacheMode")
        if self.cache_mode is not StageKVCacheMode.PAGED_SCHEDULER:
            raise ValueError(f"StageKVWorkerInitResult requires paged_scheduler mode, got {self.cache_mode.value!r}")
        if not self.cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if not isinstance(self.physical_layout, StageKVPhysicalLayout):
            raise TypeError("physical_layout must be StageKVPhysicalLayout")


def validate_stage_kv_worker_init_config(config: StageKVWorkerInitConfig) -> None:
    """Validate the Worker-supported subset of native vLLM KV cache configs."""

    if not isinstance(config, StageKVWorkerInitConfig):
        raise TypeError("config must be StageKVWorkerInitConfig")
    if config.physical_layout != BLOCK_KV_TOKEN_HEAD_DIM_V1:
        raise ValueError(f"Unsupported Stage KV physical layout: {config.physical_layout!r}")

    kv_cache_config = config.kv_cache_config
    if kv_cache_config.num_blocks <= 0:
        raise ValueError(f"KVCacheConfig.num_blocks must be positive, got {kv_cache_config.num_blocks}")
    if not kv_cache_config.kv_cache_groups:
        raise ValueError("KVCacheConfig must contain at least one KV cache group")
    if not kv_cache_config.kv_cache_tensors:
        raise ValueError("KVCacheConfig must contain at least one KV cache tensor")

    group_layers: list[str] = []
    specs_by_layer: dict[str, FullAttentionSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        if not group.layer_names:
            raise ValueError("every KV cache group must contain at least one layer")
        if not isinstance(group.kv_cache_spec, FullAttentionSpec):
            raise ValueError(
                "BLOCK_KV_TOKEN_HEAD_DIM_V1 supports only native vLLM "
                f"FullAttentionSpec, got {type(group.kv_cache_spec).__name__}"
            )
        spec = group.kv_cache_spec
        if spec.block_size <= 0 or spec.num_kv_heads <= 0 or spec.head_size <= 0:
            raise ValueError(f"invalid FullAttentionSpec geometry for layers {group.layer_names!r}")
        head_size_v = getattr(spec, "head_size_v", spec.head_size)
        if head_size_v != spec.head_size:
            raise ValueError(
                f"BLOCK_KV_TOKEN_HEAD_DIM_V1 requires equal K/V head sizes, got K={spec.head_size}, V={head_size_v}"
            )
        for layer_name in group.layer_names:
            if not layer_name:
                raise ValueError("KV cache layer names must be non-empty")
            if layer_name in specs_by_layer:
                raise ValueError(f"KV cache layer {layer_name!r} appears in multiple groups")
            specs_by_layer[layer_name] = spec
            group_layers.append(layer_name)

    tensor_layers: list[str] = []
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.size <= 0:
            raise ValueError(f"KV cache tensor size must be positive, got {tensor.size}")
        if not tensor.shared_by:
            raise ValueError("every KV cache tensor must be shared by at least one layer")
        page_sizes: set[int] = set()
        for layer_name in tensor.shared_by:
            if layer_name not in specs_by_layer:
                raise ValueError(f"KV cache tensor references unknown layer {layer_name!r}")
            page_sizes.add(specs_by_layer[layer_name].page_size_bytes)
            tensor_layers.append(layer_name)

        if len(page_sizes) != 1:
            raise ValueError(
                "BLOCK_KV_TOKEN_HEAD_DIM_V1 cannot share one tensor across "
                f"layers with different page sizes: {tensor.shared_by!r}"
            )
        expected_size = kv_cache_config.num_blocks * page_sizes.pop()
        if tensor.size != expected_size:
            raise ValueError(
                "BLOCK_KV_TOKEN_HEAD_DIM_V1 tensor size mismatch for "
                f"{tensor.shared_by!r}: expected {expected_size}, got {tensor.size}"
            )
        if tensor.offset != 0:
            raise ValueError(
                f"BLOCK_KV_TOKEN_HEAD_DIM_V1 requires zero tensor offset, got {tensor.offset} for {tensor.shared_by!r}"
            )
        if tensor.block_stride != 0:
            raise ValueError(
                "BLOCK_KV_TOKEN_HEAD_DIM_V1 requires unpacked tensors "
                f"(block_stride=0), got {tensor.block_stride} for {tensor.shared_by!r}"
            )

    if sorted(tensor_layers) != sorted(group_layers):
        raise ValueError("KV cache tensors must cover every grouped layer exactly once")


def validate_stage_kv_worker_init_result(
    config: StageKVWorkerInitConfig,
    result: StageKVWorkerInitResult,
) -> None:
    """Reject a Worker acknowledgement that differs from the sent config."""

    if not isinstance(config, StageKVWorkerInitConfig):
        raise TypeError("config must be StageKVWorkerInitConfig")
    if not isinstance(result, StageKVWorkerInitResult):
        raise TypeError("result must be StageKVWorkerInitResult")
    if result.cache_mode is not config.cache_mode:
        raise ValueError(
            f"Worker Stage KV mode mismatch: expected={config.cache_mode.value!r}, got={result.cache_mode.value!r}"
        )
    if result.cache_layout_fingerprint != config.cache_layout_fingerprint:
        raise ValueError("Worker Stage KV layout fingerprint does not match the init config")
    if result.num_blocks != config.kv_cache_config.num_blocks:
        raise ValueError(
            "Worker Stage KV num_blocks mismatch: "
            f"expected={config.kv_cache_config.num_blocks}, got={result.num_blocks}"
        )
    if result.physical_layout != config.physical_layout:
        raise ValueError("Worker Stage KV physical layout does not match the init config")


def validate_stage_kv_metadata(
    metadata: StageKVMetadata,
    *,
    config: StageKVWorkerInitConfig,
    request_id: str,
    expected_request_layout_digest: str,
) -> None:
    """Validate one allocation against the current request before forward."""

    if not isinstance(config, StageKVWorkerInitConfig):
        raise TypeError("config must be StageKVWorkerInitConfig")
    if not isinstance(metadata, StageKVMetadata):
        raise TypeError("metadata must be StageKVMetadata")
    if not expected_request_layout_digest:
        raise ValueError("expected_request_layout_digest must be non-empty")
    if metadata.request_id != request_id:
        raise ValueError(f"Stage KV metadata request mismatch: expected={request_id!r}, got={metadata.request_id!r}")
    if metadata.request_layout_digest != expected_request_layout_digest:
        raise ValueError(
            f"Stage KV metadata for request {request_id!r} does not match the expected request layout digest"
        )
    if metadata.cache_layout_fingerprint != config.cache_layout_fingerprint:
        raise ValueError(f"Stage KV metadata for request {request_id!r} has an incompatible layout fingerprint")

    groups = config.kv_cache_config.kv_cache_groups
    for branch in metadata.branches:
        if len(branch.block_ids) != len(groups):
            raise ValueError(
                f"Stage KV branch {branch.branch_id} has {len(branch.block_ids)} block groups; expected {len(groups)}"
            )
        for group_idx, (block_ids, group) in enumerate(zip(branch.block_ids, groups, strict=True)):
            block_size = group.kv_cache_spec.block_size
            expected_num_blocks = (branch.seq_len + block_size - 1) // block_size
            if len(block_ids) != expected_num_blocks:
                raise ValueError(
                    f"Stage KV branch {branch.branch_id} group {group_idx} has "
                    f"{len(block_ids)} blocks; expected {expected_num_blocks}"
                )
            if any(not 0 < block_id < config.kv_cache_config.num_blocks for block_id in block_ids):
                raise ValueError(
                    f"Stage KV branch {branch.branch_id} group {group_idx} contains a reserved or out-of-range "
                    f"block id; expected 0 < block_id < {config.kv_cache_config.num_blocks}"
                )
