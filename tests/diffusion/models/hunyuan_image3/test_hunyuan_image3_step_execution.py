# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as hy3_module
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.models.hunyuan_image3.paged_kv import (
    HunyuanFlashInferPagedKVRunner,
    HunyuanPagedAttentionInputs,
    HunyuanPromptKVPagePool,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_CFG_FACTOR,
    _STEP_GENERATOR,
    _STEP_GUIDANCE_SCALE,
    _STEP_INPUT_IDS,
    _STEP_MODEL_KWARGS,
    _STEP_PROMPT_KV,
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=0)
    pipeline.od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1),
        cache_backend=None,
        diffusion_kv_cache_skip_step_indices=None,
    )
    pipeline._pipeline = SimpleNamespace()
    return pipeline


def _state(request_id: str, step_index: int) -> DiffusionRequestState:
    state = DiffusionRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
        prompt="prompt",
    )
    state.step_index = step_index
    state.timesteps = torch.tensor([1.0, 0.5, 0.25, 0.0])
    state.latents = torch.zeros(1, 4, 8, 8)
    state.extra = {
        _STEP_CFG_FACTOR: 1,
        _STEP_AR_KV: None,
        _STEP_INPUT_IDS: None,
        _STEP_GUIDANCE_SCALE: 1.0,
        _STEP_MODEL_KWARGS: {
            "num_image_tokens": 17,
            "ar_kv_reuse_len": 0,
        },
    }
    return state


def _sampling_params(**extra_args):
    return SimpleNamespace(
        timesteps=None,
        sigmas=None,
        num_outputs_per_prompt=None,
        extra_args=extra_args,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        guidance_rescale=0.0,
        generator=None,
    )


def test_hunyuan_step_group_key_ignores_step_index_for_later_steps():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]

    groups = pipeline._split_step_groups(states)

    assert len(groups) == 1
    assert [state.request_id for state in groups[0]] == ["req-0", "req-1"]


@pytest.mark.parametrize(
    ("sampling", "prompt_item", "expected_model_bot_task", "expected_system_bot_task"),
    [
        pytest.param(
            _sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "think",
            "think",
            id="extra-args-precedence",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "image",
            "image",
            id="prompt-dict-fallback",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt"},
            "auto",
            "image",
            id="default-auto-system-prompt",
        ),
    ],
)
def test_prepare_encode_preserves_normal_hunyuan_bot_task_semantics(
    monkeypatch,
    sampling,
    prompt_item,
    expected_model_bot_task,
    expected_system_bot_task,
):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    state = DiffusionRequestState(
        request_id="req-bot-task",
        sampling=sampling,
        prompt=prompt_item,
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.prepare_encode(state)

    assert captured["bot_task"] == expected_model_bot_task
    assert captured["system_prompt_bot_task"] == expected_system_bot_task


def test_forward_uses_same_hunyuan_bot_task_semantics(monkeypatch):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    req = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="req-forward-bot-task",
                sampling_params=_sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
                prompt={"prompt": "prompt", "bot_task": "vanilla"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.forward(req)

    assert captured["bot_task"] == "think"
    assert captured["system_prompt_bot_task"] == "think"


def test_prepare_model_inputs_broadcasts_string_context_for_prompt_batch(monkeypatch):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after chat template")

    pipeline._tkwrapper = FakeTokenizer()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    pipeline.config = SimpleNamespace(image_base_size=1024)
    pipeline.generation_config = SimpleNamespace(drop_think=False, sequence_template="instruct")
    pipeline.image_processor = SimpleNamespace(
        build_image_info=lambda image_size: SimpleNamespace(image_size=image_size),
    )

    generators = [torch.Generator(device="cpu").manual_seed(i) for i in range(4)]
    with pytest.raises(RuntimeError, match="stop after chat template"):
        pipeline.prepare_model_inputs(
            prompt=[f"prompt-{idx}" for idx in range(4)],
            cot_text="shared cot",
            system_prompt="shared system",
            mode="gen_image",
            image_size=(512, 512),
            guidance_scale=5.0,
            generator=generators,
        )

    assert captured["batch_system_prompt"] == ["shared system"] * 4
    assert captured["batch_cot_text"] == ["shared cot"] * 4
    assert len(captured["batch_prompt"]) == 4
    assert len(captured["batch_gen_image_info"]) == 4
    assert captured["cfg_factor"] == 2


def test_grouped_denoise_rejects_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    with pytest.raises(ValueError, match="only supports TORCH_SDPA"):
        pipeline._ensure_grouped_attention_backend_supported(2)


def test_single_denoise_allows_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    pipeline._ensure_grouped_attention_backend_supported(1)


def test_grouped_denoise_allows_sdpa_attention_backend():
    pipeline = _pipeline()

    pipeline._ensure_grouped_attention_backend_supported(2)


def test_step_scheduler_preserves_latent_dtype_for_mixed_progress_batches():
    pipeline = _pipeline()
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda step, kwargs: {})

    class FakeScheduler:
        def step(self, noise_pred, timestep, latents, **kwargs):
            del timestep, kwargs
            return (latents.float() + noise_pred.float(),)

    state = _state("req", 0)
    state.timesteps = torch.tensor([1.0])
    state.scheduler = FakeScheduler()
    state.latents = torch.zeros(1, 4, 8, 8, dtype=torch.bfloat16)
    state.extra[_STEP_GENERATOR] = None

    pipeline.step_scheduler(state, torch.ones_like(state.latents, dtype=torch.float32))

    assert state.latents.dtype == torch.bfloat16
    assert state.step_index == 1


def test_later_step_merge_shifts_spans_without_polluting_request_state():
    pipeline = _pipeline()
    states = [_state("short", 2), _state("long", 4)]
    states[0].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 5, dtype=torch.bool),
            "full_attn_spans": [[(2, 5)]],
        }
    )
    states[1].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 7, dtype=torch.bool),
            "full_attn_spans": [[(4, 7)]],
        }
    )
    states[0].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]
    states[1].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([4])}]

    row_state_indexes = [0, 1]
    row_branches = [0, 0]
    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes,
        row_branches,
        first_step=False,
    )

    assert merged["attention_mask"].shape == (2, 1, 3, 7)
    assert merged["full_attn_spans"] == [[(4, 7)], [(4, 7)]]

    pipeline._split_merged_kwargs_to_states(states, merged, row_state_indexes, row_branches)

    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 5)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 7)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 5)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 7)]]


def test_later_step_merge_allows_request_local_step_counts_and_guidance_values():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
                "full_attn_spans": [[(2, 4)]],
                "guidance_scale": 3.0 + idx,
                "num_inference_steps": 20 + idx,
            }
        )
        state.extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=False,
    )

    assert "guidance_scale" not in merged
    assert "num_inference_steps" not in merged


@pytest.mark.parametrize(
    ("request_id", "mutate_state", "error_match"),
    [
        pytest.param(
            "broken-req",
            lambda state: state.extra.pop(_STEP_MODEL_KWARGS),
            "broken-req",
            id="missing-model-kwargs",
        ),
        pytest.param(
            "bad-cfg",
            lambda state: state.extra.__setitem__(_STEP_CFG_FACTOR, 3),
            "bad-cfg",
            id="unsupported-cfg-factor",
        ),
    ],
)
def test_denoise_step_reports_invalid_group_state_with_request_id(request_id, mutate_state, error_match):
    pipeline = _pipeline()
    state = _state(request_id, 0)
    mutate_state(state)

    with pytest.raises(ValueError, match=error_match):
        pipeline.denoise_step(InputBatch.make_batch([state]))


def test_denoise_step_uses_input_batch_group_order_and_splits_back(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = [
            {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        ]

    captured = {}

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        captured["row_state_indexes"] = list(row_state_indexes)
        captured["row_branches"] = list(row_branches)

    def fake_prepare_inputs_for_generation(input_ids, images, timestep, **model_kwargs):
        captured["input_ids"] = input_ids
        captured["images"] = images.clone()
        captured["timestep"] = timestep.clone()
        captured["merged_attention_mask_shape"] = tuple(model_kwargs["attention_mask"].shape)
        captured["merged_full_attn_spans"] = model_kwargs["full_attn_spans"]
        return {"model_kwargs": model_kwargs}

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = fake_prepare_inputs_for_generation
    pipeline.forward_call = lambda **kwargs: {"diffusion_prediction": torch.tensor([[10.0], [20.0], [1.0], [2.0]])}
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    batch = InputBatch.make_batch(states)
    out = pipeline.denoise_step(batch)

    assert captured["row_state_indexes"] == [0, 1, 0, 1]
    assert captured["row_branches"] == [0, 0, 1, 1]
    assert captured["input_ids"] is None
    assert tuple(captured["images"].shape) == (4, 1)
    assert captured["timestep"].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert captured["merged_attention_mask_shape"] == (4, 1, 2, 6)
    assert captured["merged_full_attn_spans"] == [[(4, 6)], [(4, 6)], [(4, 6)], [(4, 6)]]
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 4)], [(2, 4)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 6)], [(4, 6)]]


def test_prompt_kv_capture_restore_preserves_paged_branch_handles():
    pipeline = _pipeline()
    states = [_state("req-0", 0), _state("req-1", 0)]
    key = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5, 1, 1)
    value = key + 100.0
    lens = torch.tensor([3, 5, 2, 4], dtype=torch.long)

    class DummyManager:
        def __init__(self):
            self.image_kv_cache_map = (key, value)
            self.image_kv_cache_lens = lens
            self.page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
            self.page_batch = self.page_pool.capture_prefix(key, value, lens)
            self.restored_lens = None

        @property
        def paged_prompt_kv_enabled(self):
            return True

        @property
        def paged_prompt_kv_required(self):
            return True

        def capture_paged_prompt_kv_rows(self, row_indices, branches):
            return self.page_batch.view_rows(row_indices, branches)

        def restore_paged_prompt_kv_rows(self, row_refs):
            self.restored_lens = [row.lens for row in row_refs]
            self.page_pool.restore_batch(row_refs)

        def clear_paged_prompt_kv_current_batch(self):
            self.page_pool.clear_current()

    mgr = DummyManager()
    pipeline.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(image_attn=mgr))])
    row_state_indexes = [0, 1, 0, 1]
    row_branches = [0, 0, 1, 1]

    pipeline._capture_prompt_kv_cache(states, row_state_indexes, row_branches)

    req0_cache = states[0].extra[_STEP_PROMPT_KV][0]
    req1_cache = states[1].extra[_STEP_PROMPT_KV][0]
    assert req0_cache["lens"].tolist() == [3, 2]
    assert req1_cache["lens"].tolist() == [5, 4]
    assert "key" not in req0_cache
    assert "value" not in req0_cache
    assert mgr.image_kv_cache_map is None
    assert req0_cache["paged"].select_branch(0).lens == 3
    assert req0_cache["paged"].select_branch(1).lens == 2
    assert req1_cache["paged"].select_branch(0).lens == 5
    assert req1_cache["paged"].select_branch(1).lens == 4

    states[0].step_index = 1
    states[1].step_index = 1
    pipeline._restore_prompt_kv_cache(states, row_state_indexes=[1, 0], row_branches=[1, 1])

    assert mgr.restored_lens == [4, 2]
    assert mgr.image_kv_cache_map is None
    assert mgr.image_kv_cache_lens.tolist() == [4, 2]


def test_paged_prompt_kv_custom_mask_drops_dense_prefix_padding():
    page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
    key = torch.arange(2 * 5 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 3)
    value = key + 0.5
    lens = torch.tensor([3, 5], dtype=torch.long)
    batch = page_pool.capture_prefix(key, value, lens, reserve_current_tokens=2)
    reserved_num_blocks = page_pool.get_stats()["paged_kv_num_blocks"]
    assert reserved_num_blocks == 7

    q_len = 2
    dense_prefix_len = int(lens.max().item())
    seq_len = dense_prefix_len + q_len
    attention_mask = torch.ones(2, 1, q_len, seq_len, dtype=torch.bool)
    attention_mask[0, :, :, 3:5] = False  # dense padding columns for row 0; must be dropped.
    attention_mask[0, :, 1, 6] = False  # current-token mask; must be preserved.
    attention_mask[1, :, :, 1] = False  # valid prefix-token mask; must be preserved.

    packed = HunyuanPromptKVPagePool.build_custom_attention_mask(
        attention_mask,
        row_refs=batch.row_refs,
        q_len=q_len,
        seq_len=seq_len,
    )

    expected_row0 = torch.tensor(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
        dtype=torch.bool,
    )
    expected_row1_one_query = torch.tensor([True, False, True, True, True, True, True], dtype=torch.bool)
    expected = torch.cat([expected_row0, expected_row1_one_query, expected_row1_one_query])
    assert torch.equal(packed, expected)

    current_key = torch.zeros(2, q_len, 2, 3)
    inputs = page_pool._build_attention_inputs(current_key, seq_len, attention_mask)
    assert inputs.custom_mask is not None
    assert torch.equal(inputs.custom_mask, expected)
    assert inputs.kv_indptr.tolist() == [0, 3, 7]
    assert inputs.kv_last_page_len.tolist() == [1, 1]
    assert page_pool.get_stats()["paged_kv_num_blocks"] == reserved_num_blocks


def test_paged_prompt_kv_effective_all_keep_mask_uses_fast_attention_inputs():
    HunyuanPromptKVPagePool._packed_custom_mask_cache.clear()
    page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
    key = torch.arange(2 * 5 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 3)
    value = key + 0.5
    lens = torch.tensor([3, 5], dtype=torch.long)
    page_pool.capture_prefix(key, value, lens, reserve_current_tokens=2)

    q_len = 2
    dense_prefix_len = int(lens.max().item())
    seq_len = dense_prefix_len + q_len
    attention_mask = torch.ones(2, 1, q_len, seq_len, dtype=torch.bool)
    attention_mask[0, :, :, 3:5] = False

    current_key = torch.zeros(2, q_len, 2, 3)
    inputs = page_pool._build_attention_inputs(current_key, seq_len, attention_mask)

    assert inputs.custom_mask is None
    assert inputs.packed_custom_mask is None
    stats = page_pool.get_stats()
    assert stats["paged_mask_effective_all_keep_skips"] == 1
    assert stats["paged_mask_custom_builds"] == 0
    assert stats["paged_mask_packed_cache_misses"] == 1
    HunyuanPromptKVPagePool._packed_custom_mask_cache.clear()


def test_paged_prompt_kv_packed_mask_cache_stats_are_reported():
    HunyuanPromptKVPagePool._packed_custom_mask_cache.clear()
    page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
    key = torch.arange(1 * 3 * 1 * 1, dtype=torch.float32).reshape(1, 3, 1, 1)
    page_pool.capture_prefix(key, key, torch.tensor([3], dtype=torch.long), reserve_current_tokens=2)

    current_key = torch.zeros(1, 2, 1, 1)
    attention_mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    attention_mask[0, 0, 1, 0] = False

    inputs = page_pool._build_attention_inputs(current_key, seq_len=5, attention_mask=attention_mask)
    assert inputs.custom_mask is not None
    assert inputs.packed_custom_mask is None
    assert inputs.plan_cache_key is not None
    mask_cache_key = inputs.plan_cache_key[-1]
    fake_packed = torch.tensor([0x7F], dtype=torch.uint8)
    HunyuanPromptKVPagePool._store_packed_custom_mask(mask_cache_key, fake_packed)

    cached_inputs = page_pool._build_attention_inputs(current_key, seq_len=5, attention_mask=attention_mask)

    assert cached_inputs.custom_mask is None
    assert cached_inputs.packed_custom_mask is fake_packed
    stats = page_pool.get_stats()
    assert stats["paged_mask_custom_builds"] == 1
    assert stats["paged_mask_packed_cache_misses"] == 1
    assert stats["paged_mask_packed_cache_hits"] == 1
    HunyuanPromptKVPagePool._packed_custom_mask_cache.clear()


def test_paged_prompt_kv_custom_mask_dispatches_flashinfer_runner():
    page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
    key = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    value = key + 10.0
    page_pool.capture_prefix(key, value, torch.tensor([3], dtype=torch.long))

    current_key = torch.tensor([[[[30.0]], [[31.0]]]])
    current_value = current_key + 10.0
    query = torch.ones(1, 2, 1, 1)
    attention_mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    attention_mask[0, 0, 1, 0] = False

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def run(self, query, key_cache, value_cache, inputs, *, softmax_scale):
            self.calls.append((query, key_cache, value_cache, inputs, softmax_scale))
            assert inputs.custom_mask is not None
            return query + 7.0

    fake_runner = FakeRunner()
    page_pool._flashinfer_runner = fake_runner

    output = page_pool.run_paged_attention(
        query,
        current_key,
        current_value,
        seq_len=5,
        softmax_scale=1.0,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(output, query + 7.0)
    assert len(fake_runner.calls) == 1
    _, key_cache, value_cache, inputs, softmax_scale = fake_runner.calls[0]
    assert softmax_scale == 1.0
    assert inputs.kv_indptr.tolist() == [0, 3]
    assert inputs.kv_indices.tolist() == [0, 1, 2]
    assert inputs.kv_last_page_len.tolist() == [1]
    assert inputs.custom_mask.tolist() == [True, True, True, True, True, False, True, True, True, True]
    torch.testing.assert_close(key_cache.reshape(-1, 1, 1)[3:5], current_key[0])
    torch.testing.assert_close(value_cache.reshape(-1, 1, 1)[3:5], current_value[0])
    stats = page_pool.get_stats()
    assert stats["paged_attention_calls"] == 1
    assert stats["paged_attention_custom_mask_calls"] == 1


def test_flashinfer_runner_reuses_identical_plan_and_passes_packed_mask():
    runner = HunyuanFlashInferPagedKVRunner()
    runner._wrapper_cls = object

    class FakeWrapper:
        def __init__(self):
            self.plan_calls = []

        def plan(self, *args, custom_mask=None, packed_custom_mask=None, **kwargs):
            self.plan_calls.append((args, custom_mask, packed_custom_mask, kwargs))

        def run(self, query, kv_cache, *, return_lse=False):
            return query + 3.0

    fake_wrapper = FakeWrapper()
    runner._wrapper_by_device[("cpu", None)] = fake_wrapper

    query = torch.ones(1, 2, 1, 1)
    key_cache = torch.zeros(1, 2, 1, 1)
    value_cache = torch.zeros_like(key_cache)
    custom_mask = torch.ones(4, dtype=torch.bool)
    packed_custom_mask = torch.tensor([0xFF], dtype=torch.uint8)
    inputs = HunyuanPagedAttentionInputs(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_last_page_len=torch.tensor([2], dtype=torch.int32),
        custom_mask=custom_mask,
        packed_custom_mask=packed_custom_mask,
        plan_cache_key=("unit", 1),
        max_query_len=2,
        max_seq_len=2,
        prefix_blocks=1,
        current_blocks=0,
    )
    stats = {"paged_profile_flashinfer_plan_cache_hits": 0}

    torch.testing.assert_close(
        runner.run(query, key_cache, value_cache, inputs, softmax_scale=1.0, profile_stats=stats),
        query + 3.0,
    )
    torch.testing.assert_close(
        runner.run(query, key_cache, value_cache, inputs, softmax_scale=1.0, profile_stats=stats),
        query + 3.0,
    )

    assert len(fake_wrapper.plan_calls) == 1
    _, planned_custom_mask, planned_packed_mask, _ = fake_wrapper.plan_calls[0]
    assert planned_custom_mask is None
    assert planned_packed_mask is packed_custom_mask
    assert stats["paged_profile_flashinfer_plan_cache_hits"] == 1


def test_paged_prompt_kv_reset_stats_can_clear_prefix_cursor_without_freeing_capacity():
    page_pool = HunyuanPromptKVPagePool(page_size=2, enabled=True, required=True)
    key = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
    page_pool.capture_prefix(key, key, torch.tensor([5], dtype=torch.long), reserve_current_tokens=2)
    before = page_pool.get_stats()
    assert before["paged_cache_builds"] == 1
    assert before["paged_kv_persistent_blocks"] == 3
    assert before["paged_kv_num_blocks"] == 4

    page_pool.reset_stats(clear_cache=True)

    after = page_pool.get_stats()
    assert after["paged_cache_builds"] == 0
    assert after["paged_kv_cache_active"] is False
    assert after["paged_kv_persistent_blocks"] == 0
    assert after["paged_kv_num_blocks"] == before["paged_kv_num_blocks"]
