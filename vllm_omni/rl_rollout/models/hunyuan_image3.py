# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 recipe parameter scopes exercised by the CPU mock.

Names follow the canonical model.* backbone namespace, not its transformer.*
alias. These selectors do not enable reload, route to workers, synchronize
shared backbone replicas, or invalidate caches. Real HI3 WTE integration still
requires model-loader and multi-stage validation.
"""

from collections.abc import Iterable, Iterator
from typing import Literal, TypeVar

T = TypeVar("T")
HunyuanImage3Stage = Literal["ar", "diffusion"]

_STAGE_PREFIXES: dict[HunyuanImage3Stage, tuple[str, ...]] = {
    "ar": ("model.", "lm_head."),
    "diffusion": (
        "model.",
        "patch_embed.",
        "time_embed.",
        "time_embed_2.",
        "timestep_emb.",
        "final_layer.",
        "guidance_emb.",
        "timestep_r_emb.",
    ),
}


def select_stage_parameters(
    parameters: Iterable[tuple[str, T]], *, stage: HunyuanImage3Stage
) -> Iterator[tuple[str, T]]:
    """Select the shared backbone and recipe heads; leave VAE/vision frozen."""
    if stage not in _STAGE_PREFIXES:
        raise ValueError(f"Unknown HunyuanImage3 stage: {stage!r}")
    prefixes = _STAGE_PREFIXES[stage]
    for name, value in parameters:
        if name.startswith(prefixes):
            yield name, value
