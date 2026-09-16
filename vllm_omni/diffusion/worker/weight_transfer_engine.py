# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni-specific weight transfer engine for diffusers-style diffusion pipelines.

Upstream's IPC/NCCL engines wrap every update round in the layerwise-reload
protocol: parameters are moved to meta device at ``start_weight_update`` and
restored through the per-parameter ``weight_loader`` wiring as chunks stream
in, so quantized/fused kernel-format storages can be rebuilt in place. That
protocol assumes the model was built through vLLM's native reload-aware
construction, which omni's diffusion pipelines do not fully satisfy.

Omni's diffusion pipelines serve unquantized vLLM layers, where a received
checkpoint tensor can be applied by the ordinary ``load_weights`` path onto
the live parameters — the same "apply in place, no layerwise reload" shape
upstream's ``sparse_nccl`` engine uses. ``OmniIPCWeightTransferEngine``
therefore inherits the whole IPC payload contract (handshake, handles,
packed mode) and only replaces the lifecycle hooks:

- ``start_weight_update`` / ``finish_weight_update``: no layerwise
  meta-ization; only the packed-importer bookkeeping is kept.
- ``receive_weights``: inherited — rebuilds CUDA IPC tensors and calls
  ``model.load_weights`` on the live pipeline.

The engine is registered as backend ``"omni_ipc"`` so deployments opt in per
stage (diffusion stages only; AR stages use the native upstream backends with
their layerwise semantics).

The trainer side needs no changes: payloads are the standard IPC ones, so the
upstream ``IPCTrainerWeightTransferEngine`` / ``HTTPVLLMWeightSyncClient``
drive this engine unmodified.
"""

from __future__ import annotations

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

_BACKEND_NAME = "omni_ipc"


class OmniIPCWeightTransferEngine(IPCWeightTransferEngine):
    """IPC engine that applies weights in place, without layerwise reload."""

    def start_weight_update(self) -> None:
        """No-op: weights stream onto the live (unmeta-ized) parameters."""
        # The layerwise protocol's meta-ization is intentionally skipped: it
        # deletes parameters and restores them through weight_loader wiring
        # that omni's diffusers-style pipelines do not implement.

    def finish_weight_update(self) -> None:
        """Close packed-importer bookkeeping; no layerwise finalization."""
        self._packed_importer.close()


def register_omni_weight_transfer_engines() -> str:
    """Register omni's engine variants; returns the backend name to use.

    Idempotent: re-registration (e.g. repeated worker imports) is ignored.
    """
    try:
        WeightTransferEngineFactory.register_engine(
            _BACKEND_NAME,
            OmniIPCWeightTransferEngine,
        )
    except ValueError:
        # Already registered by an earlier import.
        pass
    return _BACKEND_NAME


OMNI_IPC_BACKEND = register_omni_weight_transfer_engines()
