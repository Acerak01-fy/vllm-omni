# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end A/B tests: baseline (``execute_model``) vs CachePool
(``execute_stepwise``).

Two execution paths coexist in the diffusion engine on this branch:

* **Baseline / ``execute_model``** – default.  ``DiffusionEngine`` uses
  ``RequestScheduler`` + ``executor.execute_request``.  One call runs the full
  ``pipeline.forward(req)`` (all denoise steps) for a single request.  Cache
  acceleration, if any, lives in per-pipeline hooks and is torn down after the
  request.  ``CacheManager`` / ``CacheBackendSlot`` are **not** used.

* **CachePool / ``execute_stepwise``** – opt-in via ``step_execution=True``
  when constructing ``Omni`` / ``AsyncOmni``.  ``DiffusionEngine`` uses
  ``StepScheduler`` + ``executor.execute_step``.  Each engine tick runs **one**
  denoise step through ``pipeline.denoise_step(input_batch)`` for a batch of
  scheduled ``DiffusionRequestState``s.  Before each step,
  ``CacheManager.activate(scheduled_states)`` installs a per-request
  ``CacheBackendSlot``; after the request finishes (or is preempted),
  ``CacheManager.free(state)`` deactivates / releases it.  Requires a pipeline
  that implements the step contract (``supports_step_execution=True``; today
  only ``QwenImagePipeline``).

The model ``Qwen/Qwen-Image`` is used throughout because it is the only
pipeline that currently supports the stepwise contract.

Usage::

    # All tests
    pytest tests/e2e/offline_inference/test_cachepool.py -s -v

    # Only the A/B parity check
    pytest tests/e2e/offline_inference/test_cachepool.py::test_baseline_vs_cachepool_parity -s
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

OUTPUT_DIR = Path(__file__).parent / "cachepool_outputs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

models = ["/home/wangfuyin/.cache/modelscope/hub/models/Qwen/Qwen-Image"]

HEIGHT = 256
WIDTH = 256
NUM_INFERENCE_STEPS = 4

PROMPTS = [
    "a cup of coffee on a wooden table",
    "a toy dinosaur on a sandy beach",
    "a futuristic city skyline at sunset",
    "a bowl of fresh strawberries on a plate",
]
                                
CACHE_DIT_CONFIG = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 2,
    "residual_diff_threshold": 0.24,
    "max_continuous_cached_steps": 3,
}

TEACACHE_CONFIG = {
    "rel_l1_thresh": 0.2,
}

# Parametrize table for backend coverage: (cache_backend, cache_config, id).
CACHE_BACKENDS = [
    pytest.param("cache_dit", CACHE_DIT_CONFIG, id="cache_dit"),
    pytest.param("tea_cache", TEACACHE_CONFIG, id="tea_cache"),
]

# ---------------------------------------------------------------------------
# Engine construction helpers
# ---------------------------------------------------------------------------


def _baseline_kwargs(model: str) -> dict[str, Any]:
    """Kwargs for the baseline path (execute_model).

    - No ``step_execution``: engine uses RequestScheduler + execute_request.
    - No ``cache_backend``: pipeline runs vanilla forward, no CacheManager.
    """
    return dict(model=model)


def _cachepool_kwargs(
    model: str,
    cache_backend: str,
    cache_config: dict[str, Any],
) -> dict[str, Any]:
    """Kwargs for the CachePool path (execute_stepwise).

    - ``step_execution=True``: engine uses StepScheduler + execute_step →
      ``DiffusionModelRunner.execute_stepwise``.
    - ``cache_backend`` + ``cache_config``: runner builds a ``CacheManager``
      from the backend's ``cache_state_driver`` and drives
      ``activate/free`` per scheduled state.
    """
    return dict(
        model=model,
        step_execution=True,
        cache_backend=cache_backend,
        cache_config=cache_config,
    )


def _sync_sampling_params(**overrides) -> OmniDiffusionSamplingParams:
    defaults = dict(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=1,
    )
    defaults.update(overrides)
    return OmniDiffusionSamplingParams(**defaults)


def _async_sampling_params(**overrides) -> OmniDiffusionSamplingParams:
    defaults = dict(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=0.0,
        num_outputs_per_prompt=1,
    )
    defaults.update(overrides)
    return OmniDiffusionSamplingParams(**defaults)


def _extract_images(output: OmniRequestOutput) -> list:
    if output.images:
        return output.images
    inner = getattr(output, "request_output", None)
    if inner is not None and hasattr(inner, "images") and inner.images:
        return inner.images
    return []


async def _collect_generate(
    omni: AsyncOmni,
    prompt,
    request_id: str,
    sampling_params_list: list[OmniDiffusionSamplingParams],
) -> OmniRequestOutput:
    last_output: OmniRequestOutput | None = None
    async for output in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params_list,
    ):
        last_output = output
    if last_output is None:
        raise RuntimeError(f"No output received for request {request_id}")
    return last_output


def _save_images(images: list, test_name: str, prefix: str = "") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        tag = f"{prefix}_" if prefix else ""
        path = OUTPUT_DIR / f"{test_name}_{tag}{i}.png"
        img.save(path)
        print(f"   Saved: {path}")


def _assert_valid_image_output(
    output: OmniRequestOutput,
    expected_count: int = 1,
    expected_width: int = WIDTH,
    expected_height: int = HEIGHT,
    test_name: str = "",
    save_prefix: str = "",
) -> list:
    assert output.final_output_type == "image", (
        f"Expected final_output_type='image', got '{output.final_output_type}'"
    )
    images = _extract_images(output)
    assert images is not None and len(images) == expected_count, (
        f"Expected {expected_count} image(s), got {len(images) if images else 0}"
    )
    for i, img in enumerate(images):
        assert img.width == expected_width, f"Image {i} width: expected {expected_width}, got {img.width}"
        assert img.height == expected_height, f"Image {i} height: expected {expected_height}, got {img.height}"
    if test_name:
        _save_images(images, test_name, save_prefix)
    return images


# ===========================================================================
# Baseline: execute_model path (RequestScheduler + pipeline.forward)
# ===========================================================================


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_baseline_execute_model_single(model_name: str):
    """Baseline: one request through ``DiffusionModelRunner.execute_model``.

    No ``step_execution``, no cache backend.  Pipeline runs vanilla
    ``forward(req)`` and returns a single image.
    """
    m = None
    try:
        m = Omni(**_baseline_kwargs(model_name))
        outputs = m.generate(PROMPTS[0], _sync_sampling_params())
        _assert_valid_image_output(outputs[0], test_name="baseline_single")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_baseline_execute_model_sequential(model_name: str):
    """Baseline: engine reused across sequential requests on ``execute_model``."""
    m = None
    try:
        m = Omni(**_baseline_kwargs(model_name))
        sp = _sync_sampling_params()

        for i, prompt in enumerate(PROMPTS[:3]):
            outputs = m.generate(prompt, sp)
            images = _assert_valid_image_output(outputs[0], test_name="baseline_seq", save_prefix=f"p{i}")
            print(f"   baseline prompt {i}: OK ({len(images)} image)")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_baseline_execute_model_multi_prompt(model_name: str):
    """Baseline: a single generate() call with a list of prompts.

    Each prompt is submitted as its own request; the engine serves them
    through ``execute_model`` back-to-back (request-level scheduling).
    """
    m = None
    try:
        m = Omni(**_baseline_kwargs(model_name))
        sp = _sync_sampling_params()
        prompts = PROMPTS[:4]

        outputs = m.generate(prompts, sp)
        assert len(outputs) == len(prompts)

        request_ids = set()
        for i, output in enumerate(outputs):
            _assert_valid_image_output(output, test_name="baseline_multi", save_prefix=f"p{i}")
            assert output.request_id not in request_ids, f"Duplicate request_id: {output.request_id}"
            request_ids.add(output.request_id)
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


# ===========================================================================
# CachePool: execute_stepwise path (StepScheduler + CacheManager)
# ===========================================================================


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("cache_backend,cache_config", CACHE_BACKENDS)
def test_cachepool_execute_stepwise_single(
    model_name: str,
    cache_backend: str,
    cache_config: dict[str, Any],
):
    """CachePool: single request through ``execute_stepwise``.

    Exercises one full slot lifecycle:
      create_empty_slot → install_slot → initialize_fresh_slot →
      (per-step denoise with slot active) → deactivate_slot → clear_slot.
    """
    m = None
    try:
        m = Omni(**_cachepool_kwargs(model_name, cache_backend, cache_config))
        outputs = m.generate(PROMPTS[0], _sync_sampling_params())
        _assert_valid_image_output(outputs[0], test_name=f"cachepool_single_{cache_backend}")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("cache_backend,cache_config", CACHE_BACKENDS)
def test_cachepool_execute_stepwise_sequential(
    model_name: str,
    cache_backend: str,
    cache_config: dict[str, Any],
):
    """CachePool: sequential requests; each must get a fresh slot with no
    state leakage from the previous request."""
    m = None
    try:
        m = Omni(**_cachepool_kwargs(model_name, cache_backend, cache_config))
        sp = _sync_sampling_params()

        for i, prompt in enumerate(PROMPTS[:3]):
            outputs = m.generate(prompt, sp)
            images = _assert_valid_image_output(
                outputs[0],
                test_name=f"cachepool_seq_{cache_backend}",
                save_prefix=f"p{i}",
            )
            print(f"   cachepool[{cache_backend}] prompt {i}: OK ({len(images)} image)")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("cache_backend,cache_config", CACHE_BACKENDS)
def test_cachepool_execute_stepwise_multi_prompt(
    model_name: str,
    cache_backend: str,
    cache_config: dict[str, Any],
):
    """CachePool: list-of-prompts generate() submits N separate requests.

    The StepScheduler services them (``max_num_seqs=1`` today, so serialized
    at the step level), and the CacheManager must allocate / deactivate a
    distinct slot per request.
    """
    m = None
    try:
        m = Omni(**_cachepool_kwargs(model_name, cache_backend, cache_config))
        sp = _sync_sampling_params()
        prompts = PROMPTS[:4]

        outputs = m.generate(prompts, sp)
        assert len(outputs) == len(prompts)

        request_ids = set()
        for i, output in enumerate(outputs):
            _assert_valid_image_output(
                output,
                test_name=f"cachepool_multi_{cache_backend}",
                save_prefix=f"p{i}",
            )
            assert output.request_id not in request_ids
            request_ids.add(output.request_id)
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("cache_backend,cache_config", CACHE_BACKENDS)
def test_cachepool_execute_stepwise_async_concurrent(
    model_name: str,
    cache_backend: str,
    cache_config: dict[str, Any],
):
    """CachePool core test: concurrent async requests under ``execute_stepwise``.

    This is the scenario CachePool is designed for.  Multiple coroutines
    submit requests via ``asyncio.gather``; the StepScheduler interleaves
    them step-by-step and the CacheManager must install / deactivate the
    correct slot around every step.  Any cross-request state leak shows up
    as a wrong / corrupted image or a wrong ``request_id``.
    """

    async def _inner():
        omni = AsyncOmni(**_cachepool_kwargs(model_name, cache_backend, cache_config))
        try:
            sp = _async_sampling_params()
            prompts = PROMPTS[:4]
            request_ids = [f"cachepool-{cache_backend}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]

            tasks = [
                _collect_generate(omni, prompt=p, request_id=rid, sampling_params_list=[sp])
                for p, rid in zip(prompts, request_ids)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == len(prompts)
            returned_ids = {r.request_id for r in results}
            for rid in request_ids:
                assert rid in returned_ids, f"Missing request_id {rid} in results"

            for i, result in enumerate(results):
                _assert_valid_image_output(
                    result,
                    test_name=f"cachepool_async_{cache_backend}",
                    save_prefix=f"p{i}",
                )
                print(f"   cachepool[{cache_backend}] prompt {i}: OK (request_id={result.request_id})")
        finally:
            omni.shutdown()

    asyncio.run(_inner())


# ===========================================================================
# CachePool: slot-compatibility & determinism scenarios
# ===========================================================================


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_cachepool_execute_stepwise_different_step_counts(model_name: str):
    """CachePool: sequential requests with different ``num_inference_steps``.

    Exercises ``CacheStateDriver.is_slot_compatible`` — when the step count
    changes, the old slot must be cleared and a fresh one allocated.
    """
    m = None
    try:
        m = Omni(**_cachepool_kwargs(model_name, "cache_dit", CACHE_DIT_CONFIG))

        for steps in (3, 6, 10):
            sp = _sync_sampling_params(num_inference_steps=steps)
            outputs = m.generate(PROMPTS[0], sp)
            images = _assert_valid_image_output(
                outputs[0], test_name="cachepool_diff_steps", save_prefix=f"s{steps}"
            )
            print(f"   steps={steps}: OK ({len(images)} image)")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_cachepool_execute_stepwise_async_mixed_step_counts(model_name: str):
    """CachePool: concurrent async requests each with a different
    ``num_inference_steps``.

    The CacheManager must maintain a per-request slot (different lengths)
    concurrently across scheduler ticks.
    """

    async def _inner():
        omni = AsyncOmni(**_cachepool_kwargs(model_name, "cache_dit", CACHE_DIT_CONFIG))
        try:
            step_counts = [2, 4, 3]
            tasks = []
            request_ids = []
            for i, steps in enumerate(step_counts):
                sp = _async_sampling_params(num_inference_steps=steps)
                rid = f"mixed-steps-{i}-{uuid.uuid4().hex[:8]}"
                request_ids.append(rid)
                tasks.append(
                    _collect_generate(omni, prompt=PROMPTS[i], request_id=rid, sampling_params_list=[sp])
                )

            results = await asyncio.gather(*tasks)
            assert len(results) == len(step_counts)
            returned_ids = {r.request_id for r in results}
            for rid in request_ids:
                assert rid in returned_ids

            for i, result in enumerate(results):
                _assert_valid_image_output(
                    result,
                    test_name="cachepool_mixed_steps",
                    save_prefix=f"s{step_counts[i]}",
                )
                print(f"   steps={step_counts[i]}: OK (request_id={result.request_id})")
        finally:
            omni.shutdown()

    asyncio.run(_inner())


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_cachepool_execute_stepwise_deterministic_with_seed(model_name: str):
    """CachePool: seeded generation is reproducible under ``execute_stepwise``.

    Two runs with the same seed and prompt must produce identical images;
    any non-determinism introduced by slot install/deactivate would surface
    as a pixel diff.
    """
    m = None
    try:
        m = Omni(**_cachepool_kwargs(model_name, "cache_dit", CACHE_DIT_CONFIG))

        def _generate_with_seed(seed: int):
            sp = _sync_sampling_params(
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            )
            return m.generate(PROMPTS[0], sp)

        outputs_run1 = _generate_with_seed(42)
        outputs_run2 = _generate_with_seed(42)

        img1 = _extract_images(outputs_run1[0])[0]
        img2 = _extract_images(outputs_run2[0])[0]

        assert img1.size == img2.size, f"Image sizes differ: {img1.size} vs {img2.size}"

        import numpy as np

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), "Seeded images are not identical across two runs"

        _save_images([img1], "cachepool_deterministic", prefix="run1")
        _save_images([img2], "cachepool_deterministic", prefix="run2")
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


# ===========================================================================
# A/B parity: same prompt + seed, baseline vs CachePool
# ===========================================================================


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_baseline_vs_cachepool_parity(model_name: str):
    """Run the same prompt+seed through both paths and sanity-check outputs.

    This is NOT a pixel-equality check (CachePool's cache_dit actually skips
    denoise steps, so the numerics diverge by design).  It verifies both
    paths:

    - finish without error,
    - emit a single image of the requested shape,
    - and, most importantly, exercise the two *different* code paths end to
      end in a single test file so a bug on either side (wrong scheduler,
      missing slot install, etc.) shows up here.
    """

    def _one_run(kwargs: dict[str, Any], tag: str):
        m = Omni(**kwargs)
        try:
            sp = _sync_sampling_params()
            outputs = m.generate(PROMPTS[0], sp)
            imgs = _assert_valid_image_output(outputs[0], test_name="ab_parity", save_prefix=tag)
            return imgs[0]
        finally:
            if hasattr(m, "close"):
                m.close()

    baseline_img = _one_run(_baseline_kwargs(model_name), tag="baseline")
    cachepool_img = _one_run(
        _cachepool_kwargs(model_name, "cache_dit", CACHE_DIT_CONFIG), tag="cachepool"
    )

    assert baseline_img.size == cachepool_img.size, (
        f"Image sizes differ between paths: baseline={baseline_img.size} "
        f"cachepool={cachepool_img.size}"
    )
    print(f"   A/B parity OK: baseline={baseline_img.size}, cachepool={cachepool_img.size}")
