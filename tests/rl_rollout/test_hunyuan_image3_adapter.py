# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the CPU-only HunyuanImage3 rollout adapter mock."""

from __future__ import annotations

import pytest
import torch

from tests.rl_rollout.mock_hunyuan_image3 import MockARRolloutEngine, MockDiffusionRolloutEngine
from tests.rl_rollout.mock_hunyuan_image3_adapter import (
    HunyuanImage3ParameterPolicy,
    MockHunyuanImage3RLAdapter,
)
from vllm_omni.rl_rollout import ARRolloutSample, DiffusionRolloutSample

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_hunyuan_image3_ar_mock_publishes_backbone_and_lm_head() -> None:
    engine = MockARRolloutEngine()
    adapter = MockHunyuanImage3RLAdapter(engine, mode="ar", chunk_size=1)

    result = await adapter.run_round("a red kite", seed=5)

    assert isinstance(result.rollout, ARRolloutSample)
    assert result.mode == "ar"
    assert engine.current_version == 1
    assert adapter.current_version == 1
    assert all(name.startswith(("model.", "lm_head.")) for name in result.published_parameter_names)
    assert "vision_model.encoder.weight" not in result.published_parameter_names


@pytest.mark.asyncio
async def test_hunyuan_image3_diffusion_mock_keeps_vae_frozen() -> None:
    engine = MockDiffusionRolloutEngine(num_inference_steps=4)
    adapter = MockHunyuanImage3RLAdapter(engine, mode="diffusion")
    vae_before = engine.parameter_snapshot()["vae.decoder.weight"]

    result = await adapter.run_round("a red kite", seed=5)

    assert isinstance(result.rollout, DiffusionRolloutSample)
    assert result.mode == "diffusion"
    assert result.rollout.sde_step_indices == (0, 2)
    assert torch.equal(engine.parameter_snapshot()["vae.decoder.weight"], vae_before)
    assert "vae.decoder.weight" not in result.published_parameter_names
    assert engine.current_version == 1


def test_hunyuan_image3_stage_policy_manifests_are_explicit() -> None:
    parameters = {
        "model.layers.0.weight": 1,
        "lm_head.weight": 2,
        "patch_embed.proj.weight": 3,
        "time_embed.weight": 4,
        "final_layer.linear.weight": 5,
        "vae.decoder.weight": 6,
    }

    ar_names = tuple(name for name, _ in HunyuanImage3ParameterPolicy("ar").select(parameters))
    diffusion_names = tuple(name for name, _ in HunyuanImage3ParameterPolicy("diffusion").select(parameters))
    assert ar_names == ("model.layers.0.weight", "lm_head.weight")
    assert diffusion_names == (
        "model.layers.0.weight",
        "patch_embed.proj.weight",
        "time_embed.weight",
        "final_layer.linear.weight",
    )


def test_hunyuan_image3_adapter_requires_matching_stage_engine() -> None:
    with pytest.raises(ValueError, match="does not match"):
        MockHunyuanImage3RLAdapter(MockARRolloutEngine(), mode="diffusion")
