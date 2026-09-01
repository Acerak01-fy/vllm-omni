# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Config factories for vllm-omni, e.g., StageConfigFactory."""

from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageType,
    _apply_platform_overrides,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
    normalize_pipeline_cli_overrides,
)
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.io_support import get_diffusion_output_type
from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

logger = init_logger(__name__)


# Default degree for any parallel axis / replica count that isn't set anywhere
# (CLI, deploy YAML, or pipeline default): a single, un-parallelized rank.
# TODO(composable_parallel): this "1" default and the per-axis device-layout
# fallbacks are currently re-derived in the merge, apply, and reconcile layers.
# Centralize them in one schema so the pre-spawn device guard and the engine-args
# defaults can't drift apart. This is the light slice; the full device-layout
# centralization is tracked as a follow-up.
_DEFAULT_PARALLEL_DEGREE = 1


def _optional_revision_kwargs(revision: str | None) -> dict[str, str]:
    """Return a revision keyword only when the caller explicitly pinned one.

    A few downstream integrations still monkeypatch the discovery helpers with
    their historical two-argument signatures.  Keeping the no-revision call
    shape unchanged preserves that compatibility while a pinned revision is
    propagated end to end.
    """
    return {"revision": revision} if revision is not None else {}


@dataclasses.dataclass(frozen=True)
class StageConfigResolution:
    """Aligned structured and compatibility views of one pipeline resolution."""

    omni_config: VllmOmniConfig | None
    legacy_stage_configs: list[StageConfig] | None
    deploy_config: DeployConfig | None
    deploy_config_path: str | None
    omni_lb_policy: str | None


@functools.cache
def _materialize_object_storage_configs(model: str) -> str:
    """Materialize an object-storage model URI's config files locally.

    vLLM's Run:AI streamer keeps ``s3://``/``gs://``/``az://`` URIs opaque until
    each stage builds its ``ModelConfig``; parent-process resolution (HF config
    lookup, pipeline/pipeline-key matching) would instead hand the URI to
    ``huggingface_hub`` helpers, which reject it with ``HFValidationError``.
    Pull the lightweight files once into vLLM's deterministic
    ``model_streamer/<hash>`` directory so config reads work here, and so the
    stage processes' own pull lands in that same directory.

    Returns the input unchanged for non object-storage paths.
    """
    if not is_runai_obj_uri(model):
        return model
    object_storage_model = ObjectStorageModel(url=model)
    object_storage_model.pull_files(model, allow_pattern=["*.model", "*.py", "*.json"])
    logger.info("Materialized object-storage configs for %s at %s", model, object_storage_model.dir)
    return object_storage_model.dir


def _name_match_candidate(model: str) -> str:
    """Last path component of a model reference, used for name-based matching.

    Object-storage URIs and HF repo ids carry non-model segments (bucket name,
    organization) that must not participate in substring matching; e.g. a
    bucket named ``qwen3-tts-models`` holding a ``Qwen3-Omni`` checkpoint must
    not resolve to the ``qwen3_tts`` pipeline.
    """
    return model.rstrip("/").rsplit("/", 1)[-1]


def with_trust_remote_code_override(
    overrides: Mapping[str, Any],
    trust_remote_code: bool | None,
) -> dict[str, Any]:
    """Merge the tri-state ``trust_remote_code`` into an override mapping.

    Single home for the precedence rule (explicit caller value > deploy
    yaml per-stage value > vLLM default False): a non-None value becomes an
    explicit override; ``None`` means "not specified" and leaves the deploy
    yaml's per-stage setting in effect. The serve ``--trust-remote-code``
    flag is store_true — its absent-False must be mapped to ``None`` at the
    CLI boundary before reaching here, since it cannot express an explicit
    False.
    """
    merged = dict(overrides)
    if trust_remote_code is not None:
        merged["trust_remote_code"] = trust_remote_code
    return merged


class StageConfigFactory:
    """Factory that loads pipeline YAML and merges CLI overrides.

    Handles both single-stage and multi-stage models.

    Pipelines are declared in ``vllm_omni/config/pipeline_registry.py`` and
    where keys in OMNI_PIPELINES map to either a PipelineConfig, or a callable
    which accepts a Transformers config as an arg & resolves to a PipelineConfig.

    NOTE: Models with generic HF ``model_type`` collisions (e.g. MiMo Audio
    reports ``qwen2``) should declare ``hf_architectures=(...)`` on their
    ``PipelineConfig`` so the factory can disambiguate via ``hf_config.architectures``.
    """

    @classmethod
    def get_pipeline_endpoint_restrictions(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None,
        revision: str | None = None,
    ) -> tuple[EndpointRestriction, ...]:
        """Given a model string, determine the corresponding endpoint restrictions.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.
            deploy_config_path: Optional path to the deploy config for the pipeline.

        Returns:
            A tuple of model specific endpoint restrictions.
        """
        pipeline_cfg = StageConfigFactory.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
            **_optional_revision_kwargs(revision),
        )
        return pipeline_cfg.endpoint_restrictions if pipeline_cfg else ()

    @classmethod
    @functools.cache
    def get_hf_config(
        cls,
        model: str,
        trust_remote_code: bool,
        revision: str | None = None,
    ) -> PretrainedConfig | None:
        """Fetch the HF config (if it exists) from the model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            the model's config or None.
        """
        hf_config = None
        try:
            config_kwargs = {"trust_remote_code": trust_remote_code}
            config_kwargs.update(_optional_revision_kwargs(revision))
            return get_config(_materialize_object_storage_configs(model), **config_kwargs)
        except Exception as e:
            logger.debug(f"`get_config` failed with exception {e}; inferred HF config is None")
        return hf_config

    @classmethod
    @functools.cache
    def try_infer_model_type(
        cls,
        model: str,
        trust_remote_code: bool,
        revision: str | None = None,
    ) -> str | None:
        """Auto-detect model_type from model directory and apply any model
        specific patches to get the correct model_type str. If we are unable
        to infer it from the model directory, we fall back to the PipelineConfig.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            model_type as a string; may be None on failure.
        """
        model_type = cls._try_infer_model_type(
            model=model,
            trust_remote_code=trust_remote_code,
            **_optional_revision_kwargs(revision),
        )
        if model_type == "vla":
            if revision is None:
                looks_like_dreamzero = _looks_like_dreamzero(model)
            else:
                looks_like_dreamzero = _looks_like_dreamzero(model, revision=revision)
            if looks_like_dreamzero:
                model_type = "dreamzero"
        return model_type

    @classmethod
    def _try_infer_model_type(
        cls,
        model: str,
        trust_remote_code: bool,
        revision: str | None = None,
    ) -> str | None:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            model_type as a string; may be None on failure.
        """
        hf_config = cls.get_hf_config(
            model=model,
            trust_remote_code=trust_remote_code,
            **_optional_revision_kwargs(revision),
        )
        if hf_config is not None:
            return hf_config.model_type

        config_source = _materialize_object_storage_configs(model)

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            config_dict = get_hf_file_to_dict("config.json", config_source, revision=revision)
            if config_dict:
                if "model_type" in config_dict:
                    return config_dict["model_type"]
                # VoxCPM2-style configs use singular ``architecture`` rather
                # than HF's standard ``model_type`` / ``architectures``. Accept
                # it as a fallback so the pipeline registry can still match.
                if "architecture" in config_dict and isinstance(config_dict["architecture"], str):
                    return config_dict["architecture"]
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")

        # Fallback for diffusers-style models: check model_index.json.
        # Some models (e.g. GLM-Image) have no root config.json but ship a
        # model_index.json with _class_name that maps to a pipeline key via
        # PipelineConfig.diffusers_class_name.
        try:
            model_index = get_hf_file_to_dict("model_index.json", config_source, revision=revision)
            if model_index and "_class_name" in model_index:
                class_name = model_index["_class_name"]
                for obj in OMNI_PIPELINES.values():
                    # If we have a resolver, call it with the optional hf_config
                    # to get the default pipeline config for this key
                    pipeline_cfg = obj(hf_config) if callable(obj) else obj
                    if pipeline_cfg is not None and class_name in (
                        pipeline_cfg.diffusers_class_name,
                        *pipeline_cfg.diffusers_class_aliases,
                    ):
                        logger.info(
                            "Detected pipeline %r from model_index.json (_class_name=%r)",
                            pipeline_cfg.model_type,
                            class_name,
                        )
                        return pipeline_cfg.model_type
        except Exception as e:
            logger.debug(f"Failed to detect model type for diffusers-style models: {e}")

        # Final fallback: some models (e.g. CosyVoice3) ship an empty
        # config.json and rely on naming conventions. Match the model path
        # basename against registered pipeline keys — longest match wins
        # so "cosyvoice3" (length 10) beats "cosyvoice" (length 9). Only
        # the basename is scanned so URI segments such as the bucket name
        # cannot select an unrelated pipeline.
        model_lower = _name_match_candidate(model).lower().replace("-", "").replace("_", "")
        best: str | None = None
        best_len = 0
        for registered_key in OMNI_PIPELINES.keys():
            candidate = registered_key.lower().replace("-", "").replace("_", "")
            if candidate and candidate in model_lower and len(candidate) > best_len:
                best = registered_key
                best_len = len(candidate)
        if best is not None:
            return best

        return None

    @classmethod
    def get_pipeline_config(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None = None,
        user_deploy_config: DeployConfig | None = None,
        revision: str | None = None,
    ) -> PipelineConfig | None:
        """Resolve the PipelineConfig for a model path/name."""
        model_type = cls.try_infer_model_type(
            model=model,
            trust_remote_code=trust_remote_code,
            **_optional_revision_kwargs(revision),
        )
        hf_config = cls.get_hf_config(
            model=model,
            trust_remote_code=trust_remote_code,
            **_optional_revision_kwargs(revision),
        )

        # Resolve the deploy config & check if the user set the pipeline;
        # If the pipeline is explicitly set, it takes highest priority
        if user_deploy_config is None:
            user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        deploy_config_pipe = cls._get_deploy_override_pipe_config(hf_config, user_deploy_config)
        if deploy_config_pipe is not None:
            return deploy_config_pipe

        # Pipeline isn't set in the yaml spec, so we need infer it ourselves.
        if model_type and model_type in OMNI_PIPELINES:
            pipeline_cfg = resolve_pipeline_config(model_type, hf_config)
            if pipeline_cfg is not None:
                return pipeline_cfg

        if hf_config is not None:
            if model_type is not None:
                logger.warning("Inferred model type %s is not registered to an Omni pipeline", model_type)
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered in OMNI_PIPELINES.values():
                    pipeline_cfg = registered if isinstance(registered, PipelineConfig) else registered(hf_config)
                    if pipeline_cfg is None:
                        continue
                    predicate = pipeline_cfg.hf_config_predicate
                    if predicate is not None:
                        try:
                            if not predicate(hf_config):
                                logger.debug(
                                    "Pipeline %r matched on architectures %s but its "
                                    "hf_config_predicate rejected the loaded config; "
                                    "continuing fallback search.",
                                    pipeline_cfg.model_type,
                                    sorted(hf_archs.intersection(pipeline_cfg.hf_architectures)),
                                )
                                continue
                        except Exception:
                            logger.exception(
                                "Pipeline %r hf_config_predicate raised; skipping.",
                                pipeline_cfg.model_type,
                            )
                            continue
                    if isinstance(pipeline_cfg, PipelineConfig) and hf_archs.intersection(
                        pipeline_cfg.hf_architectures
                    ):
                        return pipeline_cfg
        return None

    @classmethod
    def _get_deploy_override_pipe_config(
        cls,
        hf_config: PretrainedConfig | None,
        deploy_config: DeployConfig | None,
    ) -> PipelineConfig | None:
        """Resolve an explicit pipeline override from a loaded deploy config."""
        if deploy_config is None or deploy_config.pipeline is None:
            return None

        pipeline_cfg = resolve_pipeline_config(deploy_config.pipeline, hf_config)
        if pipeline_cfg is None:
            raise KeyError(
                f"Pipeline {deploy_config.pipeline!r} from deploy config is not registered "
                f"to OMNI_PIPELINES. Available: {sorted(OMNI_PIPELINES)}"
            )
        return pipeline_cfg

    @staticmethod
    def _resolve_user_deploy_path(deploy_config_path: str) -> Path:
        """Resolve an explicit deploy path using the legacy lookup rules."""
        deploy_path = Path(deploy_config_path)
        if not deploy_path.exists() and deploy_path.parent == Path("."):
            bare_name = deploy_path.name
            if not bare_name.endswith(".yaml"):
                bare_name = f"{bare_name}.yaml"
            candidate = _DEPLOY_DIR / bare_name
            if candidate.exists():
                deploy_path = candidate
        return deploy_path

    @classmethod
    def _load_user_deploy_config(cls, deploy_config_path: str | None) -> DeployConfig | None:
        """Load an explicit deploy YAML once for resolution and construction."""
        if deploy_config_path is None:
            return None
        deploy_path = cls._resolve_user_deploy_path(deploy_config_path)
        if not deploy_path.exists():
            raise FileNotFoundError(f"Deploy config not found: {deploy_path}")
        return load_deploy_config(deploy_path)

    @classmethod
    def _select_deploy_config(
        cls,
        pipeline_cfg: PipelineConfig,
        deploy_config_path: str | None,
        user_deploy_config: DeployConfig | None,
    ) -> tuple[DeployConfig, str | None]:
        """Select and load the deploy input once for both config views."""
        if user_deploy_config is not None:
            resolved_path = (
                str(cls._resolve_user_deploy_path(deploy_config_path)) if deploy_config_path is not None else None
            )
            return copy.deepcopy(user_deploy_config), resolved_path
        if deploy_config_path is not None:
            deploy_path = cls._resolve_user_deploy_path(deploy_config_path)
            if not deploy_path.exists():
                raise FileNotFoundError(f"Deploy config not found: {deploy_path}")
            return load_deploy_config(deploy_path), str(deploy_path)
        if pipeline_cfg.default_deploy_config_name is not None:
            deploy_path = _DEPLOY_DIR / pipeline_cfg.default_deploy_config_name
            return load_deploy_config(deploy_path), str(deploy_path)
        return DeployConfig(), None

    @staticmethod
    def _prepare_registry_inputs(
        pipeline_cfg: PipelineConfig,
        deploy_cfg: DeployConfig,
        cli_overrides: dict[str, Any],
    ) -> tuple[PipelineConfig, DeployConfig]:
        """Apply topology/deploy transforms shared by both resolved views."""
        # Normalize aliases (for example ``--num-gpus`` and diffusion-specific
        # spellings) once before constructing either compatibility view.  This
        # keeps the typed consumer and the legacy wire payload on the same
        # effective override set.
        normalized_cli_overrides = normalize_pipeline_cli_overrides(pipeline_cfg, cli_overrides)
        # Replace the input mapping rather than updating it in place.  Alias
        # normalization intentionally removes source spellings (for example
        # ``diffusion_streaming_output``); retaining them would route a
        # diffusion-only value to every AR/generation stage in a mixed
        # pipeline before the typed owner check runs.
        cli_overrides.clear()
        cli_overrides.update(normalized_cli_overrides)
        deploy_cfg = copy.deepcopy(deploy_cfg)
        # Apply platform overlays once before either representation is built;
        # otherwise the legacy merge and typed projection can observe
        # different devices/env/engine extras on platform-specific deploys.
        deploy_cfg = _apply_platform_overrides(deploy_cfg)
        cli_async_chunk = cli_overrides.get("async_chunk")
        if cli_async_chunk is not None:
            deploy_cfg.async_chunk = bool(cli_async_chunk)

        from vllm_omni.utils.forced_aligner import inject_forced_aligner_stage

        return inject_forced_aligner_stage(pipeline_cfg, deploy_cfg, cli_overrides)

    @staticmethod
    def _deploy_for_materialized_builder(deploy_cfg: DeployConfig) -> DeployConfig:
        """Copy an effective deploy without re-applying its platform block."""
        builder_deploy = copy.deepcopy(deploy_cfg)
        # ``merge_pipeline_deploy`` and ``VllmOmniConfig.from_pipeline_config``
        # retain their historical overlay call for standalone callers. The
        # shared resolver already applied it, so hide only the declarative
        # block while preserving it on StageConfigResolution.deploy_config.
        builder_deploy.platforms = None
        return builder_deploy

    @staticmethod
    def _typed_strategy_overrides(
        cli_overrides: dict[str, Any],
        applied: Any,
    ) -> dict[str, Any]:
        """Translate strategy-owned values into typed per-stage inputs.

        Strategy values fill only axes that the CLI did not set. This preserves
        the legacy order: deploy, then strategy, then global/per-stage CLI.
        """
        overrides = dict(cli_overrides)
        if applied is None:
            return overrides

        axis_fields = {
            "tp": ("tensor_parallel_size", "tensor_parallel_size"),
            "dp": ("data_parallel_size", "data_parallel_size"),
            "pp": ("pipeline_parallel_size", "pipeline_parallel_size"),
        }
        for stage_id, strategy_cfg in applied.per_stage_config.items():
            declared = set(strategy_cfg.l1_owners)
            for kind, (field_name, attr_name) in axis_fields.items():
                if kind not in declared:
                    continue
                stage_key = f"stage_{stage_id}_{field_name}"
                if overrides.get(stage_key) is None and overrides.get(field_name) is None:
                    overrides[stage_key] = getattr(strategy_cfg, attr_name)

            if "ep" in declared:
                stage_key = f"stage_{stage_id}_enable_expert_parallel"
                if overrides.get(stage_key) is None and overrides.get("enable_expert_parallel") is None:
                    overrides[stage_key] = strategy_cfg.enable_expert_parallel

            if "stage_replica" in declared:
                stage_key = f"stage_{stage_id}_num_replicas"
                if overrides.get(stage_key) is None and overrides.get("num_replicas") is None:
                    overrides[stage_key] = strategy_cfg.stage_replica_size
        return overrides

    @staticmethod
    def _validate_config_view_alignment(
        omni_config: VllmOmniConfig,
        legacy_stage_configs: list[StageConfig],
    ) -> None:
        typed_stages = omni_config.stage_configs
        if len(typed_stages) != len(legacy_stage_configs):
            raise ValueError(
                "Structured and legacy stage views have different lengths: "
                f"{len(typed_stages)} != {len(legacy_stage_configs)}"
            )
        for typed_stage, legacy_stage in zip(typed_stages, legacy_stage_configs):
            typed_identity = (
                typed_stage.stage_id,
                StageType(typed_stage.stage_type),
                typed_stage.model_stage,
            )
            legacy_identity = (
                legacy_stage.stage_id,
                StageType(legacy_stage.stage_type),
                legacy_stage.model_stage,
            )
            if typed_identity != legacy_identity:
                raise ValueError(
                    "Structured and legacy stage views are not aligned: "
                    f"typed={typed_identity!r}, legacy={legacy_identity!r}"
                )

            typed_runtime = typed_stage.runtime_config
            # ``legacy_stage_configs`` are the dataclass compatibility view at
            # this point (conversion to OmegaConf happens in entrypoints).
            # Materialize once here so the alignment check observes the same
            # runtime projection that downstream consumers receive.
            legacy_runtime = legacy_stage.to_omegaconf().runtime
            typed_layout = (
                typed_runtime.num_replicas,
                typed_runtime.devices,
                typed_runtime.env,
            )
            legacy_layout = (
                int(legacy_runtime.get("num_replicas", 1)),
                legacy_runtime.get("devices"),
                legacy_runtime.get("env"),
            )
            if typed_layout != legacy_layout:
                raise ValueError(
                    "Structured and legacy stage runtime layouts are not aligned: "
                    f"stage={typed_stage.stage_id}, typed={typed_layout!r}, "
                    f"legacy={legacy_layout!r}"
                )

    @classmethod
    def create_config_views_from_model(
        cls,
        model: str,
        *,
        trust_remote_code: bool | None,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
        strategy_specs: Mapping[Any, Any] | None = None,
    ) -> StageConfigResolution:
        """Resolve one model into aligned typed and legacy stage views."""
        cli_overrides = with_trust_remote_code_override(
            cli_overrides,
            trust_remote_code,
        )
        user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            trust_remote_code=bool(trust_remote_code),
            deploy_config_path=deploy_config_path,
            user_deploy_config=user_deploy_config,
            **_optional_revision_kwargs(cli_overrides.get("revision")),
        )
        if pipeline_cfg is None:
            return StageConfigResolution(None, None, None, deploy_config_path, None)

        deploy_cfg, resolved_deploy_path = cls._select_deploy_config(
            pipeline_cfg,
            deploy_config_path,
            user_deploy_config,
        )
        prepared_pipeline, prepared_deploy = cls._prepare_registry_inputs(
            pipeline_cfg,
            deploy_cfg,
            cli_overrides,
        )
        legacy_stages, applied = cls._create_legacy_from_prepared_registry(
            prepared_pipeline,
            prepared_deploy,
            cli_overrides,
            strategy_specs,
        )
        typed_overrides = cls._typed_strategy_overrides(cli_overrides, applied)
        typed_overrides["model"] = model
        omni_config = VllmOmniConfig.from_pipeline_config(
            prepared_pipeline,
            user_deploy_config=cls._deploy_for_materialized_builder(prepared_deploy),
            deploy_config_path=resolved_deploy_path,
            cli_overrides=typed_overrides,
        )
        cls._validate_config_view_alignment(omni_config, legacy_stages)
        return StageConfigResolution(
            omni_config=omni_config,
            legacy_stage_configs=legacy_stages,
            deploy_config=prepared_deploy,
            deploy_config_path=resolved_deploy_path,
            omni_lb_policy=applied.omni_lb_policy if applied is not None else None,
        )

    @classmethod
    def create_from_model(
        cls,
        model: str,
        *,
        trust_remote_code: bool | None,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
    ) -> VllmOmniConfig | None:
        """Build the structured Omni config for a model/deploy pair."""
        user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            # HF config resolution needs a real bool: transformers treats
            # None as "prompt for consent", which blocks non-interactively.
            trust_remote_code=bool(trust_remote_code),
            deploy_config_path=deploy_config_path,
            user_deploy_config=user_deploy_config,
            **_optional_revision_kwargs(cli_overrides.get("revision")),
        )
        if pipeline_cfg is None:
            return None

        registry_cli_overrides = with_trust_remote_code_override(
            {**cli_overrides, "model": model},
            trust_remote_code,
        )
        return VllmOmniConfig.from_pipeline_config(
            pipeline_cfg,
            user_deploy_config=user_deploy_config,
            deploy_config_path=deploy_config_path,
            cli_overrides=registry_cli_overrides,
        )

    @classmethod
    def create_legacy_stage_configs_from_model(
        cls,
        model: str,
        *,
        trust_remote_code: bool | None,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
        strategy_specs: Mapping[Any, Any] | None = None,
    ) -> tuple[list[StageConfig] | None, str | None]:
        """Build current runtime stage configs from the shared resolution.

        The engine still consumes the legacy StageConfig/OmegaConf shape.
        RFC #4021 will replace this transitional path as runtime consumers move
        to VllmOmniConfig.
        """
        user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            # See create_from_model: HF resolution needs a real bool.
            trust_remote_code=bool(trust_remote_code),
            deploy_config_path=deploy_config_path,
            user_deploy_config=user_deploy_config,
            **_optional_revision_kwargs(cli_overrides.get("revision")),
        )
        if pipeline_cfg is None:
            return None, None

        legacy_cli_overrides = with_trust_remote_code_override(cli_overrides, trust_remote_code)
        return cls._create_legacy_from_registry(
            pipeline_cfg,
            legacy_cli_overrides,
            deploy_config_path,
            user_deploy_config=user_deploy_config,
            strategy_specs=strategy_specs,
        )

    @classmethod
    def _create_legacy_from_registry(
        cls,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        user_deploy_config: DeployConfig | None = None,
        strategy_specs: Mapping[Any, Any] | None = None,
    ) -> tuple[list[StageConfig], str | None]:
        """Create current runtime StageConfigs from registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.

        Returns ``(stages, omni_lb_policy)`` — the strategy-derived pipeline-wide
        load-balancer policy (``None`` when no strategy set one) travels with the
        stages instead of through a mutable out-param.
        """
        deploy_cfg, _ = cls._select_deploy_config(
            pipeline_cfg,
            deploy_config_path,
            user_deploy_config,
        )
        pipeline_cfg, deploy_cfg = cls._prepare_registry_inputs(
            pipeline_cfg,
            deploy_cfg,
            cli_overrides,
        )
        stages, applied = cls._create_legacy_from_prepared_registry(
            pipeline_cfg,
            deploy_cfg,
            cli_overrides,
            strategy_specs,
        )
        omni_lb_policy = applied.omni_lb_policy if applied is not None else None
        return stages, omni_lb_policy

    @classmethod
    def _create_legacy_from_prepared_registry(
        cls,
        pipeline_cfg: PipelineConfig,
        deploy_cfg: DeployConfig,
        cli_overrides: dict[str, Any],
        strategy_specs: Mapping[Any, Any] | None,
    ) -> tuple[list[StageConfig], Any]:
        """Build the legacy view from already selected/transformed inputs."""
        stages = merge_pipeline_deploy(
            pipeline_cfg,
            cls._deploy_for_materialized_builder(deploy_cfg),
            cli_overrides,
        )

        # Overlay declarative parallel strategies (opt-in) before CLI overrides.
        applied = cls._apply_strategy_specs(stages, strategy_specs)

        explicit_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        for stage in stages:
            stage.runtime_overrides = cls._merge_cli_overrides(stage, explicit_overrides)
            if StageType(stage.stage_type) != StageType.DIFFUSION:
                continue

            # These diffusion-only CLI values were historically finalized by
            # AsyncOmniEngine after the legacy factory returned. Resolve them
            # here so the compatibility and typed views see the same input.
            stage_num_gpus = explicit_overrides.get(f"stage_{stage.stage_id}_num_gpus")
            global_num_gpus = explicit_overrides.get("num_gpus")
            if stage_num_gpus is not None or global_num_gpus is not None:
                stage.runtime_overrides["num_gpus"] = stage_num_gpus if stage_num_gpus is not None else global_num_gpus

            yaml_diffusion_quantization = stage.yaml_engine_args.pop(
                "diffusion_quantization_config",
                None,
            )
            if yaml_diffusion_quantization is not None and stage.yaml_engine_args.get("quantization_config") is None:
                stage.yaml_engine_args["quantization_config"] = yaml_diffusion_quantization

            cli_diffusion_quantization = explicit_overrides.get(
                f"stage_{stage.stage_id}_diffusion_quantization_config",
                explicit_overrides.get("diffusion_quantization_config"),
            )
            if cli_diffusion_quantization is not None and stage.runtime_overrides.get("quantization_config") is None:
                stage.runtime_overrides["quantization_config"] = cli_diffusion_quantization
            stage.runtime_overrides.pop("diffusion_quantization_config", None)

        # Re-validate the resolved layout now that CLI overrides are on top.
        cls._reconcile_strategy_with_cli(stages, applied)

        return stages, applied

    @staticmethod
    def _apply_strategy_specs(
        stages: list[StageConfig],
        strategy_specs: Mapping[Any, Any] | None,
    ) -> Any:
        """Overlay derived parallel sizing onto merged stages (opt-in).

        ``omni_lb_policy`` cannot be set from stage configs (omni reads it once
        at orchestrator construction), so a derived non-default policy is logged
        here and carried on the returned ``StrategyApplyResult`` for the caller
        to hand to the orchestrator.

        Returns the ``StrategyApplyResult`` (or ``None`` when no strategy was
        supplied) so the caller can re-validate the resolved layout once CLI
        overrides have been merged on top, and read its ``omni_lb_policy``.
        """
        if not strategy_specs:
            return None
        from vllm_omni.config.composable_parallel import apply_strategy_specs

        applied = apply_strategy_specs(stages, strategy_specs)
        if applied.omni_lb_policy is not None:
            logger.info(
                "[composable_parallel] strategy derived omni_lb_policy=%r; it will be applied "
                "to the orchestrator unless an explicit --omni-lb-policy was given.",
                applied.omni_lb_policy,
            )
        return applied

    @staticmethod
    def _reconcile_strategy_with_cli(
        stages: list[StageConfig],
        applied: Any,
    ) -> None:
        """Reconcile CLI overrides applied *after* a strategy overlay.

        CLI overrides are applied last (in ``to_omegaconf``), so for any stage a
        strategy touched we must (a) warn loudly when a CLI arg overrides a
        strategy-declared axis — the strategy is meant to be the single writer
        for the axes it declares, so a silent CLI win is surprising — and
        (b) re-run the device-count check against the *effective* (post-CLI)
        ``tp``/``dp``/``pp``/``num_replicas``/``devices`` so a CLI
        ``--devices``/``--tensor-parallel-size``/``--num-replicas`` cannot slip
        past the pre-spawn guard that exists to prevent silent OOMs.
        """
        if applied is None:
            return
        from vllm_omni.config.composable_parallel import check_device_layout

        # Axis kind -> (engine-arg field, strategy-derived attribute on the cfg).
        axis_fields = {
            "tp": ("tensor_parallel_size", "tensor_parallel_size"),
            "dp": ("data_parallel_size", "data_parallel_size"),
            "pp": ("pipeline_parallel_size", "pipeline_parallel_size"),
        }

        for stage in stages:
            cfg = applied.per_stage_config.get(stage.stage_id)
            if cfg is None:
                continue
            overrides = stage.runtime_overrides or {}
            declared = set(cfg.l1_owners.keys())

            for kind, (field_name, attr) in axis_fields.items():
                if kind in declared and overrides.get(field_name) is not None:
                    cli_val = overrides[field_name]
                    derived = getattr(cfg, attr)
                    if cli_val != derived:
                        logger.warning(
                            "[composable_parallel] stage %s: CLI %s=%s overrides the "
                            "strategy-derived %s=%s. The CLI value wins; remove one to avoid ambiguity.",
                            stage.stage_id,
                            field_name,
                            cli_val,
                            field_name,
                            derived,
                        )
            if "stage_replica" in declared and overrides.get("num_replicas") is not None:
                cli_val = overrides["num_replicas"]
                if cli_val != cfg.stage_replica_size:
                    logger.warning(
                        "[composable_parallel] stage %s: CLI num_replicas=%s overrides the "
                        "strategy-derived num_replicas=%s. The CLI value wins; remove one to avoid ambiguity.",
                        stage.stage_id,
                        cli_val,
                        cfg.stage_replica_size,
                    )

            def _eff(field_name: str, fallback: Any) -> Any:
                val = overrides.get(field_name)
                return val if val is not None else fallback

            def _eff_degree(field_name: str, source: dict[str, Any]) -> int:
                # Single place the per-axis "default to 1" fallback is applied,
                # for both the override and the YAML-default sides (see the
                # _DEFAULT_PARALLEL_DEGREE TODO).
                value = _eff(field_name, source.get(field_name, _DEFAULT_PARALLEL_DEGREE))
                return int(value or _DEFAULT_PARALLEL_DEGREE)

            check_device_layout(
                _eff("devices", stage.yaml_runtime.get("devices")),
                tensor_parallel_size=_eff_degree("tensor_parallel_size", stage.yaml_engine_args),
                data_parallel_size=_eff_degree("data_parallel_size", stage.yaml_engine_args),
                pipeline_parallel_size=_eff_degree("pipeline_parallel_size", stage.yaml_engine_args),
                num_replicas=_eff_degree("num_replicas", stage.yaml_runtime),
                role=stage.model_stage,
            )

    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Single-stage diffusion - no YAML needed.

        Creates a default diffusion stage configuration for single-stage
        diffusion models. Returns a legacy OmegaConf-compatible dict for
        backward compatibility with OmniStage.

        Args:
            kwargs: Engine arguments from CLI/API.

        Returns:
            List containing a single config dict for the diffusion stage.
        """
        # Calculate devices based on parallel config
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"

        engine_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("parallel_config",):
                continue
            engine_args[key] = value

        # Serialize parallel_config as dict for OmegaConf. Test helpers
        # sometimes pass SimpleNamespace rather than a dataclass instance.
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            if dataclasses.is_dataclass(parallel_config) and not isinstance(parallel_config, type):
                engine_args["parallel_config"] = asdict(parallel_config)
            elif hasattr(parallel_config, "__dict__"):
                engine_args["parallel_config"] = dict(vars(parallel_config))
            else:
                engine_args["parallel_config"] = parallel_config

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        engine_args.setdefault("max_num_seqs", 1)
        model_class_name = engine_args.get("model_class_name")

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": StageType.DIFFUSION.value,
            "runtime": {
                "process": True,
                "devices": devices,
            },
            "engine_args": create_config(engine_args),
            "final_output": True,
            "final_output_type": get_diffusion_output_type(model_class_name),
        }

        return [config_dict]

    @classmethod
    def _merge_cli_overrides(
        cls,
        stage: StageConfig,
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge global and per-stage (``stage_N_*``) CLI overrides.

        Orchestrator-owned keys are filtered by ``build_stage_runtime_overrides``
        using ``OrchestratorArgs`` as the single source of truth; unknown
        server/uvicorn keys are dropped downstream by
        ``filter_dataclass_kwargs(OmniEngineArgs, ...)``.
        """
        return build_stage_runtime_overrides(stage.stage_id, cli_overrides)
