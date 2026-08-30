# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum
from vllm_omni.platforms.npu.utils import (
    _reshape_and_cache_without_cache_mode,
    _use_diffusion_paged_kv_write_plan,
)

logger = init_logger(__name__)

_DIFFUSION_PACKED_MODULES_MAPPING = {
    "HunyuanImage3Pipeline": {
        "experts": ["experts.0.gate_up_proj", "experts.0.down_proj"],
    },
}


class NPUOmniPlatform(OmniPlatform, NPUPlatform):
    """NPU/Ascend implementation of OmniPlatform.

    Inherits all NPU-specific implementations from vllm-ascend's NPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.NPU
    dist_backend: str = "hccl"

    # conv2d convolution operator in the code2wav module of Qwen3-TTS not being able to run on Aclnn
    def __init__(self) -> None:
        from vllm_ascend.utils import adapt_patch

        from vllm_omni.platforms.npu._310p import apply_patches as apply_310p_patches
        from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import (
            apply_minicpmo_4_5_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import (
            apply_qwen3_tts_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_tokenizer_v2 import (
            apply_qwen3_tts_tokenizer_v2_patch,
        )

        adapt_patch(is_global_patch=True)
        apply_minicpmo_4_5_code2wav_patch()
        apply_qwen3_tts_code2wav_patch()
        apply_qwen3_tts_tokenizer_v2_patch()
        apply_310p_patches()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        super().set_device(device)

        # Register vllm_ascend custom ops (torch.ops._C_ascend.*).
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()

        # Ascend quantized weights are converted from ND to FRACTAL_NZ
        # after loading. Enable internal format so the NZ storage layout
        # is preserved for fused NPU kernels.
        torch.npu.config.allow_internal_format = True

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker"

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.utils import adapt_patch

        # Omni's custom DiffusionWorker does not pass through vLLM-Ascend's
        # NPUWorker constructor, where worker-local patches are normally
        # installed.  In particular, AscendBlockTables needs the patched
        # non-UVA buffer implementation on NPU.
        adapt_patch()
        init_ascend_config(vllm_config)
        cls._patch_scatter_cache_mode_compat()

    @staticmethod
    def _patch_scatter_cache_mode_compat() -> None:
        """Bridge the 0.27 writer to torch_npu schemas without ``cache_mode``.

        vLLM-Ascend 0.27 passes ``cache_mode=\"Norm\"`` to
        ``npu_scatter_pa_kv_cache``.  Some torch_npu 2.10 builds expose the
        same normal-layout operator without that keyword; the normal layout
        is the operator default, so only the unsupported argument is removed.
        """

        from vllm_ascend.device import device_op

        op = getattr(getattr(torch.ops, "npu", None), "npu_scatter_pa_kv_cache", None)
        schema = getattr(getattr(op, "default", None), "_schema", None)
        if schema is None or "cache_mode" in str(schema):
            return
        device_operator = device_op.DeviceOperator
        if getattr(device_operator, "_omni_cache_mode_compat", False):
            return

        device_operator.reshape_and_cache = classmethod(_reshape_and_cache_without_cache_mode)
        device_operator._omni_cache_mode_compat = True
        logger.warning_once(
            "Applied narrow vLLM-Ascend cache-writer compatibility: "
            "torch_npu.npu_scatter_pa_kv_cache does not accept cache_mode."
        )

    @classmethod
    def configure_diffusion_vllm_config(cls, vllm_config: Any, od_config: Any) -> None:
        """Use the block geometry required by Ascend's native paged kernel."""
        if getattr(od_config, "diffusion_kv_mode", None) is None:
            return
        from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode

        if od_config.diffusion_kv_mode is not DiffusionKVCacheMode.PAGED_SCHEDULER:
            return
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        supported_sizes = [
            size for size in AscendAttentionBackend.get_supported_kernel_block_sizes() if type(size) is int and size > 0
        ]
        if not supported_sizes:
            raise RuntimeError("Ascend paged attention did not expose an integer kernel block size")
        # vLLM's generic default is 16, while the 0.27 Ascend FIA backend
        # stores cache pages as 128-token blocks.  Set the Manager geometry
        # before KV specs are collected so Scheduler and Worker agree.
        vllm_config.cache_config.block_size = supported_sizes[0]

    @classmethod
    def supports_diffusion_paged_kv_write_plan(cls) -> bool:
        return True

    @classmethod
    def build_diffusion_paged_kv_write_plans(
        cls,
        *,
        rows,
        row_bindings,
        kv_cache_config,
        block_tables,
        device,
    ) -> dict[str, Any]:
        """Build compact PA_NZ mappings from Scheduler-owned physical blocks."""

        del cls
        from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import DiffusionPagedKVWritePlan

        if len(rows) != len(row_bindings):
            raise ValueError("Diffusion KV write-plan rows and bindings must have equal length")
        plans: dict[str, Any] = {}
        for group_index, cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kernel_block_size = int(block_tables.kernel_block_sizes[group_index])
            blocks_per_kv_block = int(block_tables.blocks_per_kv_block[group_index])
            global_slots: list[int] = []
            for row, binding in zip(rows, row_bindings, strict=True):
                if len(binding.block_ids) != len(kv_cache_config.kv_cache_groups):
                    raise RuntimeError(
                        f"Paged KV row {row.identity!r} has {len(binding.block_ids)} block groups; "
                        f"expected {len(kv_cache_config.kv_cache_groups)}"
                    )
                logical_blocks = binding.block_ids[group_index]
                kernel_blocks = tuple(
                    block_id * blocks_per_kv_block + offset
                    for block_id in logical_blocks
                    for offset in range(blocks_per_kv_block)
                )
                for position in range(row.kv_start_pos, row.kv_start_pos + row.query_len):
                    block_index = position // (kernel_block_size * block_tables.cp_size)
                    block_offset = position % (kernel_block_size * block_tables.cp_size)
                    if block_index >= len(kernel_blocks):
                        raise RuntimeError(
                            f"Paged KV row {row.identity!r} has no physical block for write position {position}"
                        )
                    if block_tables.cp_size == 1:
                        local_offset = block_offset
                    else:
                        is_local = (
                            block_offset // block_tables.cp_interleave % block_tables.cp_size == block_tables.cp_rank
                        )
                        if not is_local:
                            global_slots.append(-1)
                            continue
                        rounds = block_offset // (block_tables.cp_interleave * block_tables.cp_size)
                        local_offset = rounds * block_tables.cp_interleave + block_offset % block_tables.cp_interleave
                    global_slots.append(kernel_blocks[block_index] * kernel_block_size + local_offset)

            touched_blocks = sorted({slot // kernel_block_size for slot in global_slots if slot >= 0})
            compact_block_index = {block_id: index for index, block_id in enumerate(touched_blocks)}
            local_slots = [
                -1
                if slot < 0
                else compact_block_index[slot // kernel_block_size] * kernel_block_size + slot % kernel_block_size
                for slot in global_slots
            ]
            plan = DiffusionPagedKVWritePlan(
                block_ids=torch.tensor(touched_blocks, dtype=torch.int32, device=device),
                local_slot_mapping=torch.tensor(local_slots, dtype=torch.int32, device=device),
            )
            for layer_name in cache_group.layer_names:
                plans[layer_name] = plan
        return plans

    @classmethod
    def use_diffusion_paged_kv_write_plan(cls, write_plan: Any):
        del cls
        return _use_diffusion_paged_kv_write_plan(write_plan)

    @classmethod
    def get_diffusion_kv_block_tables_cls(cls) -> type:
        from vllm_ascend.worker.v2.block_table import AscendBlockTables

        return AscendBlockTables

    @classmethod
    def build_diffusion_kv_attn_metadata(cls, **kwargs: Any) -> dict[str, Any]:
        """Build the Ascend metadata required by the native NPU backend."""
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.worker.v2.attn_utils import build_attn_metadata

        kwargs = dict(kwargs)
        seq_lens_cpu = kwargs.pop("seq_lens_cpu")
        kwargs["seq_lens_np"] = seq_lens_cpu.detach().cpu().numpy()
        # The diffusion adapter always supplies a paged cache and the current
        # K/V write span. ChunkedPrefill is Ascend's cache-backed FIA state for
        # both multi-token updates and single-token updates in this path.
        kwargs["attn_state"] = AscendAttentionState.ChunkedPrefill
        return build_attn_metadata(**kwargs)

    @classmethod
    def init_diffusion_model_runner_runtime(cls, vllm_config: Any, od_config: Any, device: torch.device) -> None:
        from vllm_ascend.ascend_forward_context import set_mc2_mask, set_mc2_tokens_capacity

        from vllm_omni.platforms.npu.models.minimax_h3 import (
            apply_minimax_h3_qwen3vl_patch,
            apply_minimax_h3_qwen3vl_sdpa_patch,
            apply_minimax_h3_qwen3vl_swiglu_patch,
        )

        # These patches import the MiniMax encoder package, whose __init__ loads
        # pipeline_minimax_h3 → diffusion.data. Doing that during platform
        # construction races vllm_omni/__init__.py (patch before config) and
        # closes a cycle through pipeline_registry → PI0_PIPELINE →
        # DiffusionOutput. Apply them only after the platform exists, before
        # the diffusion pipeline is loaded.
        apply_minimax_h3_qwen3vl_patch()
        apply_minimax_h3_qwen3vl_sdpa_patch()
        apply_minimax_h3_qwen3vl_swiglu_patch()
        set_mc2_tokens_capacity(vllm_config, od_config.max_num_seqs, 1)
        set_mc2_mask(vllm_config, device)

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def prepare_diffusion_op_runtime(cls, op_name: str, **kwargs: Any) -> None:
        if op_name != "fused_moe":
            return

        from vllm_omni.platforms.npu.layers.fused_moe import prepare_fused_moe_runtime

        prepare_fused_moe_runtime()

    @classmethod
    def register_additional_diffusion_fused_moe_hooks(cls, moe_runner: Any) -> None:
        from vllm_omni.platforms.npu.layers.fused_moe import fused_moe_forward_context_pre_hook

        moe_runner.register_forward_pre_hook(
            fused_moe_forward_context_pre_hook,
            with_kwargs=True,
        )

    @classmethod
    def reset_diffusion_fused_moe_forward_context(cls) -> None:
        from vllm_omni.platforms.npu.layers.fused_moe import reset_fused_moe_forward_context

        reset_fused_moe_forward_context()

    @classmethod
    def get_diffusion_packed_modules_mapping(
        cls,
        model_class: type[nn.Module],
    ) -> dict[str, list[str]] | None:
        return _DIFFUSION_PACKED_MODULES_MAPPING.get(model_class.__name__, None)

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = True,
    ) -> str:
        # NPU has no TRTLLM backend; accepted for signature parity, ignored.
        from importlib.util import find_spec

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            cls.validate_diffusion_attn_backend(backend_upper)
            if backend_upper in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB"):
                logger.warning(
                    "HuggingFace kernels-backed FlashAttention is "
                    "not supported on NPU. Falling back to local "
                    "FLASH_ATTN."
                )
                backend_upper = "FLASH_ATTN"

            if backend_upper in ("FLASH_ATTN", "RAINFUSION_ATTN") and find_spec("mindiesd"):
                # Eager-import mindiesd only for backends that actually reach
                # mindiesd kernels: FLASH_ATTN directly, and RAINFUSION_ATTN
                # via its dense FlashAttention fallback (used before
                # start_step and on any layer without a sparsifiable video
                # segment). Other backends (e.g. TORCH_SDPA) never touch
                # mindiesd, so a broken optional install must not block them.
                # CANN snapshots the custom-op registry at the first
                # custom-op regInfo lookup in the process (e.g. a
                # vllm-ascend custom op during model load/warmup). Import
                # mindiesd here so its env.py prepends the mindiesd vendor
                # dirs (aie_ascendc etc.) to ASCEND_CUSTOM_OPP_PATH before
                # that snapshot; otherwise aclnnLaserAttention /
                # FusedAttentionScore fail with EZ1001 "does not support
                # opType" for the rest of the process.
                import mindiesd  # noqa: F401

            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        # Try FLASH_ATTN if mindiesd is available, otherwise fall back to SDPA
        if find_spec("mindiesd"):
            # Configure ASCEND_CUSTOM_OPP_PATH for mindiesd custom ops upon import
            import mindiesd  # noqa: F401

            logger.debug("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.debug("Falling back to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        """Return whether MindIE-SD is installed for dense NPU FlashAttention."""

        from importlib.util import find_spec

        return find_spec("mindiesd") is not None

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("npu")
        return torch.device("npu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()

    @classmethod
    def record_device_event(cls) -> torch.Event | None:
        """Record a NPU event on the default stream to mark tensor readiness.

        On NPU/Ascend with HCCL, distributed communication may use internal
        streams not visible to the default stream. Synchronize the default
        stream first so that HCCL results are written back before we record
        the event, ensuring d2h_stream.wait_event() captures the complete
        output data.
        """
        try:
            torch.npu.current_stream().synchronize()
            # The async output worker uses the public ``torch.Stream`` API.
            # With torch_npu 2.10, a native ``torch.npu.Event`` cannot be
            # consumed by that wrapper stream (the reverse direction is
            # supported), while ``torch.Event`` is dispatched to the active
            # NPU backend and remains cross-stream compatible.
            event = torch.Event()
            event.record()
            return event
        except Exception:
            logger.warning("Failed to record NPU event for cross-stream sync")
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.npu.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        free, total = torch.npu.mem_get_info(device)
        return free, total

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def create_autocast_context(cls, *, device_type, dtype, enabled=True):
        if device_type != "npu":
            return super().create_autocast_context(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            )
        if not enabled:
            return nullcontext()

        # NPU-specific fallback
        try:
            return torch.npu.amp.autocast(dtype=dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("autocast unavailable for device_type=%s dtype=%s: %s", device_type, dtype, exc)
        return nullcontext()

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_omni.platforms.npu.profiler.NPUTorchProfilerWrapper"

    @classmethod
    def get_graph_wrapper_cls(cls) -> type:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        return ACLGraphWrapper

    @classmethod
    def set_forward_context(
        cls,
        attn_metadata,
        vllm_config,
        *,
        cudagraph_runtime_mode,
        batch_descriptor,
    ):
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            attn_metadata,
            vllm_config,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        )
