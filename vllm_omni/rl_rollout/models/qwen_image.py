# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""QwenImage parameter selection for the full-transformer RL recipe.

Inputs use pipeline parameter names. The iterator preserves values and their
storage; the trainer owns materialization, IPC encoding, and buffer lifetime.
This optional selector does not restrict which components a different recipe
can train or publish through the pipeline's existing load_weights method.
"""

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def select_transformer_parameters(parameters: Iterable[tuple[str, T]]) -> Iterator[tuple[str, T]]:
    """Select transformer.* without materializing the parameter stream."""
    for name, value in parameters:
        if name.startswith("transformer."):
            yield name, value
