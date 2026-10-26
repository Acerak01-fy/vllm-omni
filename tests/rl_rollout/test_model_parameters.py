# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model recipe scopes must preserve lazy sources and frozen components."""

import pytest

from vllm_omni.rl_rollout.models.hunyuan_image3 import select_stage_parameters
from vllm_omni.rl_rollout.models.qwen_image import select_transformer_parameters

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen_selection_keeps_source_lazy_and_preserves_storage() -> None:
    value = object()

    def source():
        yield "vae.decoder.weight", object()
        yield "transformer.img_in.weight", value
        raise RuntimeError("next tensor has not been materialized")

    selected = select_transformer_parameters(source())
    name, actual = next(selected)
    assert name == "transformer.img_in.weight"
    assert actual is value
    with pytest.raises(RuntimeError, match="not been materialized"):
        next(selected)


def test_qwen_selection_excludes_conditioning_components() -> None:
    parameters = {
        "transformer.img_in.weight": object(),
        "vae.decoder.weight": object(),
        "text_encoder.layer.weight": object(),
        "vision_model.weight": object(),
    }
    assert dict(select_transformer_parameters(parameters.items())) == {
        "transformer.img_in.weight": parameters["transformer.img_in.weight"]
    }


@pytest.mark.parametrize("stage", ["ar", "diffusion"])
def test_hunyuan_scopes_share_backbone_and_keep_heads_separate(stage) -> None:
    parameters = {
        "model.layers.0.weight": object(),
        "lm_head.weight": object(),
        "patch_embed.proj.weight": object(),
        "time_embed.weight": object(),
        "time_embed_2.weight": object(),
        "timestep_emb.weight": object(),
        "final_layer.linear.weight": object(),
        "guidance_emb.weight": object(),
        "timestep_r_emb.weight": object(),
        "vae.decoder.weight": object(),
        "vision_model.weight": object(),
        "vision_aligner.weight": object(),
    }
    selected = dict(select_stage_parameters(parameters.items(), stage=stage))
    assert selected["model.layers.0.weight"] is parameters["model.layers.0.weight"]
    if stage == "ar":
        assert set(selected) == {"model.layers.0.weight", "lm_head.weight"}
    else:
        assert set(selected) == set(parameters) - {
            "lm_head.weight",
            "vae.decoder.weight",
            "vision_model.weight",
            "vision_aligner.weight",
        }


def test_hunyuan_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        list(select_stage_parameters([], stage="unknown"))
