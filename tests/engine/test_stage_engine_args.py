from __future__ import annotations

import copy
import sys
import types
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict,
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"
_LLM_BACKEND_FIELDS = frozenset(field.name for field in fields(OmniEngineArgs))
_DIFFUSION_BACKEND_FIELDS = frozenset(field.name for field in fields(OmniDiffusionConfig))


@pytest.fixture(autouse=True)
def _stable_engine_arg_environment(monkeypatch):
    from vllm_omni import platforms

    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP16", raising=False)
    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)

    def _resolve_test_worker(engine_args):
        worker_type = engine_args.get("worker_type")
        if worker_type is not None and engine_args.get("worker_cls") in (None, "auto"):
            engine_args["worker_cls"] = f"test.{worker_type}.Worker"

    monkeypatch.setattr(stage_init_utils, "resolve_worker_cls", _resolve_test_worker)


def _engine_arg_inputs(tmp_path: Path) -> tuple[PipelineConfig, DeployConfig, str]:
    parent_model = tmp_path / "parent-model"
    stage_model = tmp_path / "stage-model"
    diffusion_model = tmp_path / "diffusion-model"
    for root in (parent_model, stage_model, diffusion_model):
        root.mkdir()
    (stage_model / "ar-model").mkdir()
    (stage_model / "ar-tokenizer").mkdir()

    pipeline = PipelineConfig(
        model_type="typed-engine-args",
        model_arch="PipelineModel",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                requires_multimodal_data=True,
                hf_config_name="thinker_config",
                engine_output_type="hidden_states",
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_subdir="ar-model",
                tokenizer_subdir="ar-tokenizer",
                async_chunk_process_next_stage_input_func="operator.add",
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="talker",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(0,),
                engine_output_type="audio_tokens",
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(1,),
                final_output=True,
                final_output_type="image",
                model_arch="TypedDiffusionPipeline",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=True,
        trust_remote_code=True,
        dtype="float16",
        distributed_executor_backend="mp",
        stages=[
            StageDeployConfig(
                stage_id=0,
                tensor_parallel_size=2,
                max_num_seqs=8,
                default_sampling_params={"extra_args": {"voice": "test"}},
                engine_extras={
                    "model": str(stage_model),
                    "attention_backend": "FLASHINFER",
                    "hf_overrides": {"rope_scaling": {"factor": 2}},
                    "limit_mm_per_prompt": {"audio": 1},
                    "kv_cache_memory_bytes": 1024,
                    "omni_kv_config": {"need_send_cache": True},
                    "max_cudagraph_capture_size": 0,
                    "worker_cls": "test.custom.Worker",
                },
            ),
            StageDeployConfig(
                stage_id=1,
                max_num_seqs=4,
            ),
            StageDeployConfig(
                stage_id=2,
                tensor_parallel_size=2,
                vae_parallel_mode="spatial_shard_height",
                diffusion_attention_config={
                    "default": {"backend": "FLASH_ATTN"},
                    "per_role": {"cross": {"backend": "TORCH_SDPA"}},
                },
                engine_extras={
                    "model": str(diffusion_model),
                    "engine_backend": "test.diffusion.Engine",
                },
            ),
        ],
    )
    return pipeline, deploy, str(parent_model)


def _legacy_and_typed_stages(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    model: str,
):
    legacy_stages = [
        stage.to_omegaconf()
        for stage in merge_pipeline_deploy(
            pipeline,
            copy.deepcopy(deploy),
        )
    ]
    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
        cli_overrides={"model": model},
    )
    return legacy_stages, omni_config


def test_typed_llm_engine_args_preserve_legacy_adapter_behavior(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)
    connector_spec = {"name": "SharedMemoryConnector", "extra": {"mode": "test"}}
    cli_tokenizer = "/external/tokenizer"

    typed_args_by_stage = {}
    for stage_id in (0, 1):
        legacy_args = build_legacy_engine_args_dict(
            legacy_stages[stage_id],
            model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=cli_tokenizer,
        )
        typed_args = build_engine_args_dict_from_omni_stage_config(
            omni_config.stage_by_id(stage_id),
            model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=cli_tokenizer,
        )
        typed_args_by_stage[stage_id] = typed_args

        assert {name: typed_args[name] for name in legacy_args} == legacy_args

    thinker_args = typed_args_by_stage[0]
    assert thinker_args["model"] == str(tmp_path / "stage-model" / "ar-model")
    assert thinker_args["tokenizer"] == str(tmp_path / "stage-model" / "ar-tokenizer")
    assert thinker_args["stage_id"] == 0
    assert thinker_args["stage_connector_spec"] == connector_spec
    assert thinker_args["has_sampling_extra_args"] is True
    assert thinker_args["omni_kv_config"] == {"need_send_cache": True}
    assert thinker_args["worker_cls"] == "test.custom.Worker"
    assert thinker_args["attention_backend"] == "FLASHINFER"
    assert thinker_args["trust_remote_code"] is True
    assert thinker_args["dtype"] == "float16"
    assert thinker_args["distributed_executor_backend"] == "mp"
    assert thinker_args["tensor_parallel_size"] == 2
    assert thinker_args["hf_overrides"] == {"rope_scaling": {"factor": 2}}
    assert thinker_args["limit_mm_per_prompt"] == {"audio": 1}
    assert thinker_args["kv_cache_memory_bytes"] == 1024
    assert thinker_args["max_cudagraph_capture_size"] == 0

    talker_args = typed_args_by_stage[1]
    assert talker_args["model"] == model
    assert talker_args["tokenizer"] == cli_tokenizer
    assert talker_args["worker_cls"] == "test.generation.Worker"
    assert talker_args["disable_hybrid_kv_cache_manager"] is True
    assert talker_args["enable_prefix_caching"] is False


def test_typed_diffusion_engine_args_use_structured_diffusion_config(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[2], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(2),
        model,
    )

    assert typed_args["stage_id"] == legacy_args["stage_id"] == 2
    assert typed_args["model"] == legacy_args["model"] == str(tmp_path / "diffusion-model")
    assert omni_config.stage_by_id(2).diffusion_config.model == str(tmp_path / "diffusion-model")
    assert typed_args["model_stage"] == legacy_args["model_stage"] == "dit"
    assert typed_args["model_arch"] == legacy_args["model_arch"] == "TypedDiffusionPipeline"
    assert typed_args["model_class_name"] == legacy_args["model_class_name"] == "TypedDiffusionPipeline"
    assert typed_args["engine_backend"] == legacy_args["engine_backend"] == "test.diffusion.Engine"
    assert typed_args["parallel_config"]["tensor_parallel_size"] == 2
    assert typed_args["parallel_config"]["vae_parallel_mode"] == "spatial_shard_height"
    assert isinstance(typed_args["diffusion_attention_config"], AttentionConfig)
    assert typed_args["diffusion_attention_config"].default.backend == "FLASH_ATTN"
    assert typed_args["diffusion_attention_config"].per_role["cross"].backend == "TORCH_SDPA"


def test_build_engine_args_dict_preserves_legacy_api(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, _ = _legacy_and_typed_stages(pipeline, deploy, model)
    connector_spec = {"name": "SharedMemoryConnector", "extra": {}}

    compatibility_args = build_engine_args_dict(
        copy.deepcopy(legacy_stages[0]),
        model,
        stage_connector_spec=connector_spec,
        cli_tokenizer="/external/tokenizer",
    )
    explicitly_legacy_args = build_legacy_engine_args_dict(
        copy.deepcopy(legacy_stages[0]),
        model,
        stage_connector_spec=connector_spec,
        cli_tokenizer="/external/tokenizer",
    )

    assert compatibility_args == explicitly_legacy_args


def test_legacy_engine_args_drop_none_tp_without_mutating_stage_config():
    stage_config = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"tensor_parallel_size": None},
        default_sampling_params={},
    )

    engine_args = build_legacy_engine_args_dict(
        stage_config,
        model="test-model",
    )

    assert "tensor_parallel_size" not in engine_args
    assert stage_config.engine_args == {"tensor_parallel_size": None}


def test_typed_engine_args_own_rocm_attention_default(monkeypatch, tmp_path):
    from vllm_omni import platforms

    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)
    monkeypatch.setattr(platforms.current_omni_platform, "is_rocm", lambda: True)
    aiter_module = types.ModuleType("vllm._aiter_ops")
    setattr(
        aiter_module,
        "rocm_aiter_ops",
        types.SimpleNamespace(is_enabled=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_module)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[1], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(1),
        model,
    )

    assert "attention_backend" not in legacy_args
    assert typed_args["attention_backend"] == "TRITON_ATTN"


@pytest.mark.parametrize("model_type", sorted(OMNI_PIPELINES))
def test_typed_engine_args_cover_current_registry_backend_fields(model_type):
    pipeline = resolve_pipeline_config(model_type)
    if pipeline is None:
        pytest.skip(f"Pipeline {model_type!r} requires an HF config to resolve")

    deploy = (
        load_deploy_config(_DEPLOY_DIR / pipeline.default_deploy_config_name)
        if pipeline.default_deploy_config_name is not None
        else DeployConfig()
    )
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model="/tmp",
    )

    for legacy_stage in legacy_stages:
        stage_id = legacy_stage.stage_id
        legacy_args = build_legacy_engine_args_dict(legacy_stage, model="/tmp")
        typed_args = build_engine_args_dict_from_omni_stage_config(
            omni_config.stage_by_id(stage_id),
            model="/tmp",
        )
        backend_fields = (
            _DIFFUSION_BACKEND_FIELDS if legacy_stage.stage_type == StageType.DIFFUSION else _LLM_BACKEND_FIELDS
        )

        missing_fields = (legacy_args.keys() & backend_fields) - typed_args.keys()
        assert not missing_fields, f"{model_type} stage {stage_id} lost backend fields: {sorted(missing_fields)}"

        comparable_fields = (legacy_args.keys() & backend_fields) - {
            "cache_config",
            "parallel_config",
        }
        assert {name: typed_args[name] for name in comparable_fields} == {
            name: legacy_args[name] for name in comparable_fields
        }

        legacy_parallel = legacy_args.get("parallel_config")
        if isinstance(legacy_parallel, Mapping):
            typed_parallel = typed_args["parallel_config"]
            assert {name: typed_parallel[name] for name in legacy_parallel} == dict(legacy_parallel)

        legacy_cache = legacy_args.get("cache_config")
        typed_cache = typed_args.get("cache_config")
        if isinstance(legacy_cache, Mapping) and typed_cache is not None:
            assert {name: getattr(typed_cache, name) for name in legacy_cache} == dict(legacy_cache)
