# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion application policies over the upstream IPC transport.

``direct-load`` passes received weights to the live pipeline's ``load_weights``
without entering the layerwise meta/restore lifecycle. It supports unquantized
models whose loaders can update existing parameters. The upstream IPC payload,
receive path, and update synchronization are inherited unchanged.

``layerwise`` delegates to the upstream factory and requires the model to
implement vLLM's reload lifecycle; selecting it does not add that capability.
"""

from __future__ import annotations

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine


class IPCDirectLoadWeightTransferEngine(IPCWeightTransferEngine):
    """Load standard IPC weights through the live pipeline without layerwise reload."""

    def start_weight_update(self) -> None:
        """No-op: weights stream onto the live (unmeta-ized) parameters."""
        # The layerwise protocol's meta-ization is intentionally skipped: it
        # deletes parameters and restores them through weight_loader wiring
        # that omni's diffusers-style pipelines do not implement.

    def finish_weight_update(self) -> None:
        """Close packed-importer bookkeeping; no layerwise finalization."""
        self._packed_importer.close()


def create_diffusion_weight_transfer_engine(
    config,
    vllm_config,
    device,
    model,
    *,
    apply_strategy: str | None,
):
    """Create an IPC engine with the requested diffusion apply strategy.

    ``direct-load`` is valid only for dense, unquantized diffusion modules.  The
    layerwise strategy is delegated to the upstream factory, preserving its
    quantized reload behavior and its future backend additions.
    """
    if apply_strategy in (None, "layerwise"):
        return WeightTransferEngineFactory.create_engine(config, vllm_config, device, model)
    if apply_strategy == "direct-load":
        if getattr(vllm_config, "quant_config", None) is not None:
            raise ValueError(
                "weight_transfer_apply_strategy='direct-load' does not support "
                "quantized diffusion weights; a reload-aware loader is required"
            )
        if config.backend != "ipc":
            raise ValueError("weight_transfer_apply_strategy='direct-load' currently requires backend='ipc'")
        return IPCDirectLoadWeightTransferEngine(config, vllm_config, device, model)
    raise ValueError(
        f"Unsupported weight_transfer_apply_strategy={apply_strategy!r}; expected 'direct-load' or 'layerwise'"
    )
