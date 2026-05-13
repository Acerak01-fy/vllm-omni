# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Temporary global switches for experimental diffusion features."""

import os

EXPERIMENT_CACHEPOOL = False
EXPERIMENT_PAGED_CACHE = os.environ.get("EXPERIMENT_PAGED_CACHE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
