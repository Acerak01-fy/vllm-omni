# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stable rollout-side contracts for RL integrations.

This package contains the model-neutral boundary shared by RL trainers and
vLLM-Omni.  Transport payloads and model-specific output extraction remain in
adapters; the package does not depend on a trainer implementation.
"""

from vllm_omni.rl_rollout.adapter import RLPublicationError, RLWeightTransferAdapter
from vllm_omni.rl_rollout.adapters import QwenImageParameterPolicy, QwenImageRLAdapter
from vllm_omni.rl_rollout.contracts import (
    ARRolloutSample,
    DiffusionRolloutSample,
    RLUpdateResult,
    RolloutMode,
    VLLMWeightTransferClient,
    WeightPublication,
    WeightUpdateChunk,
)

__all__ = [
    "ARRolloutSample",
    "DiffusionRolloutSample",
    "RLPublicationError",
    "RLUpdateResult",
    "RLWeightTransferAdapter",
    "QwenImageParameterPolicy",
    "QwenImageRLAdapter",
    "RolloutMode",
    "VLLMWeightTransferClient",
    "WeightPublication",
    "WeightUpdateChunk",
]
