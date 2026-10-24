# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Trainer-facing adapter over vLLM-Omni's existing WTE lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.rl_rollout.contracts import VLLMWeightTransferClient, WeightPublication, WeightUpdateChunk


class RLPublicationError(ValueError):
    """Raised when a trainer publication cannot be sent safely."""


class RLWeightTransferAdapter:
    """Validate RL publication metadata and call the existing Omni WTE API.

    The adapter owns trainer-side version and manifest bookkeeping.  It does
    not implement IPC, HTTP, NCCL, layerwise reload, or model parameter
    application; those remain in the upstream WTE and Omni workers.
    """

    def __init__(self, client: VLLMWeightTransferClient, *, current_version: int = 0) -> None:
        if current_version < 0:
            raise ValueError("current_version must be non-negative")
        self._client = client
        self._current_version = current_version
        self._initialized = False
        self._active_version: int | None = None
        self._expected_names: set[str] = set()
        self._received_names: set[str] = set()
        self._failed = False

    @property
    def current_version(self) -> int:
        """Highest version successfully handed to the rollout client."""
        return self._current_version

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self, init_info: Mapping[str, Any]) -> None:
        """Initialize the configured upstream transport exactly once."""
        if self._initialized:
            raise RLPublicationError("RL weight transfer is already initialized")
        await self._client.init_weight_transfer_engine(init_info)
        self._initialized = True

    def _require_ready(self) -> None:
        if not self._initialized:
            raise RLPublicationError("initialize must run before publish")
        if self._failed:
            raise RLPublicationError("publication failed; recreate the rollout client before retrying")

    async def start_publication(self, *, version: int, parameter_names: tuple[str, ...]) -> None:
        """Declare a complete manifest before a streaming sender reuses its buffer."""
        self._require_ready()
        if self._active_version is not None:
            raise RLPublicationError("a publication is already active")
        if version != self._current_version + 1:
            raise RLPublicationError(
                f"publication version must advance from {self._current_version} to {self._current_version + 1}, "
                f"got {version}"
            )
        if not parameter_names or any(not isinstance(name, str) or not name for name in parameter_names):
            raise RLPublicationError("publication requires non-empty parameter names")
        if len(set(parameter_names)) != len(parameter_names):
            raise RLPublicationError("publication parameter names must be unique")
        self._active_version = version
        self._expected_names = set(parameter_names)
        self._received_names = set()
        try:
            await self._client.start_weight_update()
        except BaseException:
            self._failed = True
            raise

    async def update_chunk(self, chunk: WeightUpdateChunk) -> None:
        """Consume an opaque chunk before its trainer-owned storage may be reused."""
        self._require_ready()
        if self._active_version is None:
            raise RLPublicationError("start_publication must run before update_chunk")
        try:
            if chunk.version != self._active_version:
                raise RLPublicationError("chunk version does not match publication version")
            names = set(chunk.parameter_names)
            if names - self._expected_names:
                raise RLPublicationError("chunk contains a parameter outside the publication manifest")
            if names & self._received_names:
                raise RLPublicationError("a parameter appears in more than one publication chunk")
            await self._client.update_weights(chunk.payload)
            self._received_names.update(names)
        except BaseException:
            self._failed = True
            raise

    async def finish_publication(self) -> None:
        """Expose the version only after all declared parameters and finish succeed."""
        self._require_ready()
        if self._active_version is None:
            raise RLPublicationError("start_publication must run before finish_publication")
        try:
            if self._received_names != self._expected_names:
                missing = sorted(self._expected_names - self._received_names)
                raise RLPublicationError(f"publication manifest mismatch; missing={missing}")
            await self._client.finish_weight_update(str(self._active_version))
        except BaseException:
            self._failed = True
            raise
        self._current_version = self._active_version
        self._active_version = None
        self._expected_names.clear()
        self._received_names.clear()

    async def publish(self, publication: WeightPublication) -> None:
        """Publish pre-encoded chunks through the same streaming lifecycle."""
        expected_names = set(publication.parameter_names)
        received_names: set[str] = set()
        for chunk in publication.chunks:
            chunk_names = set(chunk.parameter_names)
            if chunk_names - expected_names:
                raise RLPublicationError("chunk contains a parameter outside the publication manifest")
            if received_names.intersection(chunk_names):
                raise RLPublicationError("a parameter appears in more than one publication chunk")
            received_names.update(chunk_names)
        if received_names != expected_names:
            raise RLPublicationError(
                f"publication manifest mismatch; missing={sorted(expected_names - received_names)}"
            )

        await self.start_publication(version=publication.version, parameter_names=publication.parameter_names)
        for chunk in publication.chunks:
            await self.update_chunk(chunk)
        await self.finish_publication()
