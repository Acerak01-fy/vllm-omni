# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Mapping
from functools import lru_cache

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

logger = init_logger(__name__)

DIFFUSION_MODEL_INDEX_FILES = (
    "model_index.json",
    "modular_model_index.json",
)


def get_diffusion_model_index(
    model_name: str,
    *,
    revision: str | None = None,
) -> dict | None:
    """Read the first standard Diffusers pipeline index available."""
    for filename in DIFFUSION_MODEL_INDEX_FILES:
        config = get_hf_file_to_dict(filename, model_name, revision=revision)
        if isinstance(config, Mapping):
            return dict(config)
    return None


def load_diffusers_config(model_name, *, revision: str | None = None) -> dict:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    load_kwargs = {"revision": revision} if revision is not None else {}
    config = DiffusionPipeline.load_config(model_name, **load_kwargs)
    return config


def _looks_like_bagel(model_name: str, *, revision: str | None = None) -> bool:
    """Best-effort detection for Bagel (non-diffusers) diffusion models."""
    try:
        file_kwargs = {"revision": revision} if revision is not None else {}
        cfg = get_hf_file_to_dict("config.json", model_name, **file_kwargs)
        model_type = cfg.get("model_type")
        if model_type == "bagel":
            return True
        architectures = cfg.get("architectures") or []
        return "BagelForConditionalGeneration" in architectures
    except Exception:
        return False


def _looks_like_dreamzero(model_name: str, *, revision: str | None = None) -> bool:
    """Best-effort detection for DreamZero-style VLA diffusion checkpoints."""
    try:
        file_kwargs = {"revision": revision} if revision is not None else {}
        cfg = get_hf_file_to_dict("config.json", model_name, **file_kwargs)
        if cfg.get("model_type") != "vla":
            return False
        action_head_cfg = cfg.get("action_head_cfg") or {}
        if not isinstance(action_head_cfg, Mapping):
            return False
        action_head_config = action_head_cfg.get("config") or {}
        if not isinstance(action_head_config, Mapping):
            return False
        diffusion_model_cfg = action_head_config.get("diffusion_model_cfg") or {}
        if not isinstance(diffusion_model_cfg, Mapping):
            return False
        return (
            action_head_cfg.get("_target_")
            == "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
            and diffusion_model_cfg.get("_target_")
            == ("groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel")
        )
    except Exception:
        return False


HIDREAM_O1_SIGNATURE_WEIGHTS = (
    "model.final_layer2.linear.weight",
    "final_layer2.linear.weight",
)


def _looks_like_hidream_o1(
    model_name: str,
    config: Mapping | None = None,
    *,
    revision: str | None = None,
) -> bool:
    """Detect HiDream-O1 without matching regular Qwen3-VL checkpoints."""
    try:
        file_kwargs = {"revision": revision} if revision is not None else {}
        cfg = config if config is not None else get_hf_file_to_dict("config.json", model_name, **file_kwargs)
        if not isinstance(cfg, Mapping) or cfg.get("model_type") != "qwen3_vl":
            return False

        index = get_hf_file_to_dict("model.safetensors.index.json", model_name, **file_kwargs)
        if not isinstance(index, Mapping):
            return False
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            return False
        return any(key in weight_map for key in HIDREAM_O1_SIGNATURE_WEIGHTS)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


@lru_cache
def is_diffusion_model(model_name: str, revision: str | None = None) -> bool:
    """Check if a model is a diffusion model.

    Uses multiple fallback strategies to detect diffusion models:
    1. Check the local file system for a standard Diffusers index
    2. Check using vllm's get_hf_file_to_dict utility
    3. Try the standard diffusers approach (may fail due to import issues)
    """
    # Strategy 1: Check local file system first (fastest, avoids import issues)
    if os.path.isdir(model_name):
        for filename in DIFFUSION_MODEL_INDEX_FILES:
            model_index_path = os.path.join(model_name, filename)
            if not os.path.exists(model_index_path):
                continue
            try:
                import json

                with open(model_index_path) as f:
                    config_dict = json.load(f)
                if config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
                    logger.debug("Detected diffusion model via local %s", filename)
                    return True
            except Exception as e:
                logger.debug("Failed to read local %s: %s", filename, e)

    # Strategy 2: Check using vllm's utility (works for both local and remote models)
    if revision is None:
        config_dict = get_diffusion_model_index(model_name)
    else:
        config_dict = get_diffusion_model_index(model_name, revision=revision)
    if config_dict is not None and config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
        logger.debug("Detected diffusion model via a standard Diffusers index")
        return True

    # Strategy 3: Try the standard diffusers approach (may fail due to import issues)
    # This is last because it requires importing diffusers/xformers/flash_attn
    # which may have compatibility issues
    try:
        if revision is None:
            load_diffusers_config(model_name)
        else:
            load_diffusers_config(model_name, revision=revision)
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("Failed to import diffusers dependencies: %s", e)
        logger.debug("This may be due to flash_attn/PyTorch version mismatch")
    except Exception as e:
        logger.debug("Failed to load diffusers config via DiffusionPipeline: %s", e)

    if revision is None:
        looks_like_bagel = _looks_like_bagel(model_name)
        looks_like_dreamzero = _looks_like_dreamzero(model_name)
        looks_like_hidream = _looks_like_hidream_o1(model_name)
    else:
        looks_like_bagel = _looks_like_bagel(model_name, revision=revision)
        looks_like_dreamzero = _looks_like_dreamzero(model_name, revision=revision)
        looks_like_hidream = _looks_like_hidream_o1(model_name, revision=revision)
    return looks_like_bagel or looks_like_dreamzero or looks_like_hidream
