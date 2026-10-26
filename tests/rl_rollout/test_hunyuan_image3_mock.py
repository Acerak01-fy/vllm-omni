# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the mock HunyuanImage3 RL rollout boundary."""

from __future__ import annotations

import pytest
import torch

from tests.rl_rollout.mock_hunyuan_image3 import (
    MockARRolloutEngine,
    MockDiffusionRolloutEngine,
    MockHunyuanImage3Trainer,
    MockWeightTransferError,
)
from tests.rl_rollout.mock_types import ARRolloutSample, DiffusionRolloutSample

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_ar_rollout_exposes_token_logprob_and_response_boundaries() -> None:
    engine = MockARRolloutEngine()
    trainer = MockHunyuanImage3Trainer(engine)

    result = await trainer.run_round("a red kite", seed=7)
    sample = result.rollout
    assert isinstance(sample, ARRolloutSample)

    assert result.mode == "ar"
    assert sample.response_start == len(sample.prompt_token_ids)
    assert sample.response_end == len(sample.token_ids)
    assert len(sample.response_token_ids) == len(sample.response_token_logprobs)
    assert result.published_parameter_names == engine.trainable_parameter_names
    assert engine.current_version == 1
    assert result.published_version == 1
    assert engine.weight_transfer.published_versions == [0, 1]


@pytest.mark.asyncio
async def test_diffusion_rollout_exposes_image_trajectory_and_keeps_vae_frozen() -> None:
    engine = MockDiffusionRolloutEngine(num_inference_steps=4)
    vae_before = engine.parameter_snapshot()["vae.decoder.weight"]
    trainer = MockHunyuanImage3Trainer(engine)

    result = await trainer.run_round("a red kite", seed=7)
    sample = result.rollout
    assert isinstance(sample, DiffusionRolloutSample)

    assert result.mode == "diffusion"
    assert sample.image.shape == (1, 3, 2, 2)
    assert len(sample.latent_trajectory) == 4
    assert len(sample.sigmas) == len(sample.timesteps) == 4
    assert sample.sde_step_indices == (0, 2)
    assert sample.conditioning["prompt"] == "a red kite"
    assert engine.current_version == 1
    assert torch.equal(engine.parameter_snapshot()["vae.decoder.weight"], vae_before)
    assert set(result.published_parameter_names) == set(engine.trainable_parameter_names)


@pytest.mark.asyncio
async def test_generation_is_paused_until_the_full_update_finishes() -> None:
    engine = MockARRolloutEngine()
    await engine.init_weight_transfer_engine({"backend": "mock", "packed": False})
    before = await engine.generate("a red kite", seed=3)
    updated = {name: value + 1.0 for name, value in engine.export_trainable_weights().items()}

    await engine.start_weight_update()
    await engine.update_weights(updated)
    assert engine.current_version == 0
    for name, value in updated.items():
        torch.testing.assert_close(engine.parameter_snapshot()[name], value)
    with pytest.raises(MockWeightTransferError, match="paused"):
        await engine.generate("a red kite", seed=3)

    await engine.finish_weight_update("1")
    after = await engine.generate("a red kite", seed=3)
    assert after.weight_version == 1
    assert before.response_token_logprobs != after.response_token_logprobs


@pytest.mark.asyncio
async def test_native_lifecycle_rejects_invalid_order_and_incomplete_snapshot() -> None:
    engine = MockARRolloutEngine()
    weights = engine.export_trainable_weights()
    with pytest.raises(MockWeightTransferError, match="init_weight_transfer_engine"):
        await engine.start_weight_update()
    await engine.init_weight_transfer_engine({"backend": "mock"})
    with pytest.raises(MockWeightTransferError, match="active weight update"):
        await engine.update_weights(weights)
    with pytest.raises(MockWeightTransferError, match="active weight update"):
        await engine.finish_weight_update("1")
    await engine.start_weight_update()
    first_name = next(iter(weights))
    await engine.update_weights({first_name: weights[first_name]})
    with pytest.raises(MockWeightTransferError, match="incomplete"):
        await engine.finish_weight_update("1")
    with pytest.raises(MockWeightTransferError, match="already"):
        await engine.start_weight_update()
    with pytest.raises(MockWeightTransferError, match="paused"):
        await engine.generate("a red kite")


@pytest.mark.asyncio
async def test_mock_trainer_publishes_consecutive_versions_through_native_calls() -> None:
    engine = MockDiffusionRolloutEngine()
    trainer = MockHunyuanImage3Trainer(engine)
    results = [await trainer.run_round("a red kite") for _ in range(3)]
    assert [result.rollout.weight_version for result in results] == [0, 1, 2]
    assert [result.published_version for result in results] == [1, 2, 3]
    assert engine.weight_transfer.published_versions == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_duplicate_publication_version_is_rejected_by_mock_bookkeeping() -> None:
    engine = MockDiffusionRolloutEngine()
    await MockHunyuanImage3Trainer(engine).run_round("a red kite")
    await engine.start_weight_update()
    await engine.update_weights(engine.export_trainable_weights())
    with pytest.raises(MockWeightTransferError, match="must advance"):
        await engine.finish_weight_update("1")
    assert engine.current_version == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["shape", "dtype", "duplicate", "frozen"])
async def test_mock_session_rejects_invalid_cpu_payload(invalid: str) -> None:
    engine = MockDiffusionRolloutEngine()
    await engine.init_weight_transfer_engine({"backend": "mock"})
    await engine.start_weight_update()
    name, value = next(iter(engine.export_trainable_weights().items()))
    if invalid == "shape":
        value = torch.zeros(3)
    elif invalid == "dtype":
        value = value.to(torch.float64)
    elif invalid == "duplicate":
        await engine.update_weights({name: value})
    else:
        name = "vae.decoder.weight"
        value = engine.parameter_snapshot()[name]
    with pytest.raises(MockWeightTransferError):
        await engine.update_weights({name: value})
    assert engine.current_version == 0
