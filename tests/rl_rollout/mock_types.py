# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only samples and results for the CPU RL simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

RolloutMode = Literal["ar", "diffusion"]


@dataclass(frozen=True)
class ARRolloutSample:
    """One autoregressive sample returned to a trainer.

    ``response_token_logprobs`` is aligned with ``response_token_ids``.  The
    two offsets locate that response inside the complete ``token_ids`` stream,
    so a trainer does not need to infer the prompt/response boundary again.
    """

    prompt: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_token_logprobs: tuple[float, ...]
    response_start: int
    response_end: int
    weight_version: int

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return prompt and response tokens in model-input order."""
        return self.prompt_token_ids + self.response_token_ids

    def __post_init__(self) -> None:
        if len(self.response_token_ids) != len(self.response_token_logprobs):
            raise ValueError("response token IDs and logprobs must have the same length")
        if self.response_start != len(self.prompt_token_ids):
            raise ValueError("response_start must equal the prompt token count")
        if self.response_end != self.response_start + len(self.response_token_ids):
            raise ValueError("response_end must point just past the response tokens")
        if self.weight_version < 0:
            raise ValueError("weight_version must be non-negative")


@dataclass(frozen=True)
class DiffusionRolloutSample:
    """One diffusion sample and the denoising data used by a trainer."""

    prompt: str
    image: Any
    latent_trajectory: tuple[Any, ...]
    sigmas: tuple[float, ...]
    timesteps: tuple[int, ...]
    sde_step_indices: tuple[int, ...]
    conditioning: Mapping[str, Any]
    trajectory_logprob: float
    weight_version: int

    def __post_init__(self) -> None:
        step_count = len(self.latent_trajectory)
        if len(self.sigmas) != step_count or len(self.timesteps) != step_count:
            raise ValueError("diffusion trajectory metadata must have one entry per latent")
        if any(index < 0 or index >= step_count for index in self.sde_step_indices):
            raise ValueError("SDE step indices must reference the latent trajectory")
        if self.weight_version < 0:
            raise ValueError("weight_version must be non-negative")


@dataclass(frozen=True)
class RLUpdateResult:
    """Trainer result after a rollout, mock/real reward, and publication."""

    mode: RolloutMode
    rollout: ARRolloutSample | DiffusionRolloutSample
    reward: float
    advantage: float
    loss: float
    published_version: int
    published_parameter_names: tuple[str, ...]
