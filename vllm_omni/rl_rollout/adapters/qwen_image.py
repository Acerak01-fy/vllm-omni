# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""QwenImage full-parameter rollout adapter.

QwenImage RL normally updates the DiT transformer while keeping the VAE and
conditioning components fixed.  This module makes that model boundary
explicit and validates names under ``transformer.`` in the
model-neutral weight publication contract.  The publication is then handed to
the existing vLLM/Omni weight-transfer lifecycle; this module does not know
whether control uses RPC or HTTP, or data uses IPC or another upstream backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vllm_omni.rl_rollout.adapter import RLWeightTransferAdapter
from vllm_omni.rl_rollout.contracts import VLLMWeightTransferClient

ParameterItems = Mapping[str, Any] | Iterable[tuple[str, Any]]


def _items(parameters: ParameterItems) -> tuple[tuple[str, Any], ...]:
    """Materialize a parameter source and reject ambiguous manifests."""
    materialized = tuple(parameters.items()) if isinstance(parameters, Mapping) else tuple(parameters)
    names = [name for name, _ in materialized]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("parameter names must be unique")
    return materialized


class QwenImageParameterPolicy:
    """Select the full QwenImage RL parameter set.

    ``full parameter`` refers to all parameters in the trainable DiT
    transformer.  The VAE, text encoders, and other conditioning modules are
    intentionally excluded by default because they are inference-time
    components in the QwenImage RL recipe.  A caller can provide a different
    prefix policy when a future recipe deliberately trains an auxiliary
    component.
    """

    DEFAULT_TRAINABLE_PREFIXES = ("transformer.",)
    DEFAULT_FROZEN_PREFIXES = (
        "vae.",
        "text_encoder.",
        "text_encoder_2.",
        "vision_model.",
    )

    def __init__(
        self,
        *,
        trainable_prefixes: Sequence[str] = DEFAULT_TRAINABLE_PREFIXES,
        frozen_prefixes: Sequence[str] = DEFAULT_FROZEN_PREFIXES,
    ) -> None:
        self.trainable_prefixes = tuple(trainable_prefixes)
        self.frozen_prefixes = tuple(frozen_prefixes)
        if not self.trainable_prefixes:
            raise ValueError("at least one trainable prefix is required")
        if any(not prefix for prefix in (*self.trainable_prefixes, *self.frozen_prefixes)):
            raise ValueError("parameter prefixes must be non-empty")

    def select_names(self, parameters: ParameterItems) -> tuple[str, ...]:
        """Return the ordered trainable manifest for a pipeline snapshot."""
        selected = tuple(name for name, _ in _items(parameters) if name.startswith(self.trainable_prefixes))
        if not selected:
            prefixes = ", ".join(repr(prefix) for prefix in self.trainable_prefixes)
            raise ValueError(f"no QwenImage trainable parameters matched prefixes {prefixes}")
        frozen = tuple(name for name in selected if name.startswith(self.frozen_prefixes))
        if frozen:
            raise ValueError(f"trainable manifest includes frozen QwenImage parameters: {frozen}")
        return selected

    def select(self, parameters: ParameterItems) -> tuple[tuple[str, Any], ...]:
        """Return parameter values for the selected trainable manifest."""
        materialized = _items(parameters)
        selected_names = set(self.select_names(materialized))
        return tuple((name, value) for name, value in materialized if name in selected_names)


class QwenImageRLAdapter(RLWeightTransferAdapter):
    """Validate the QwenImage manifest and forward trainer-encoded WTE payloads.

    The trainer owns IPC handles, packed buffers, and their lifetime. This
    adapter selects no transport and never turns tensors into a wire payload.
    Use ``start_publication/update_chunk/finish_publication`` with a streaming
    sender, or ``publish`` with independently owned, pre-encoded chunks.
    """

    def __init__(
        self,
        client: VLLMWeightTransferClient,
        *,
        policy: QwenImageParameterPolicy | None = None,
        current_version: int = 0,
    ) -> None:
        super().__init__(client, current_version=current_version)
        self.policy = policy or QwenImageParameterPolicy()

    def trainable_parameter_names(self, parameters: ParameterItems) -> tuple[str, ...]:
        """Return the trainable manifest using canonical pipeline parameter names."""
        return self.policy.select_names(parameters)

    async def start_publication(self, *, version: int, parameter_names: tuple[str, ...]) -> None:
        """Reject parameters outside the recipe before the worker enters reload."""
        selected = self.policy.select_names((name, None) for name in parameter_names)
        if selected != parameter_names:
            raise ValueError("QwenImage publication contains parameters outside the trainable policy")
        await super().start_publication(version=version, parameter_names=parameter_names)
