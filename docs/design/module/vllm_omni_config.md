---
title: vLLM-Omni Configuration
kind: module
status: draft
architecture_state: deferred-pending-refactor
ownership_status: provisional
owners:
  - "@lishunyang12"
  - "@alex-jw-brooks"
primary_code_paths:
  - vllm_omni/config/**
  - vllm_omni/deploy/**
  - vllm_omni/model_executor/stage_configs/**
related_code_paths:
  - vllm_omni/platforms/*/stage_configs/**
depends_on: []
validation_paths:
  - tests/config/**
upstream_refs:
  - vllm.config
last_reviewed: 2026-08-11
last_verified_commit: e356708da4cd39992405686dacfc32b18fcccc7f
---

# vLLM-Omni configuration

This page intentionally records discovery paths only. Substantive
configuration ownership and contract work is deferred until the active
configuration refactoring has settled.

## Deferred contract scope

The current code and tests remain authoritative for parsing, defaults,
overrides, deployment topology, stage construction, and runtime projection.
This draft does not define a stable precedence model, environment-variable
capture rule, canonical configuration object, or invariant namespace.

The PR review proposal to capture environment-derived values in configuration
objects at initialization is an open design question, not a current
invariant.

## Current structured-stage reuse behavior

The current RFC #4021 implementation uses upstream vLLM config classes as the
schema source for AR and generation stages:

- `OmniStageLoadConfig` inherits `vllm.config.LoadConfig`.
- `OmniStageCacheConfig` inherits `vllm.config.CacheConfig`.
- `OmniStageSchedulerConfig` inherits `vllm.config.SchedulerConfig`.
- `OmniStageParallelConfig` inherits `vllm.config.ParallelConfig`.

This inheritance preserves the existing `VllmOmniConfig` paths while reducing
duplicate downstream declarations. It does not, by itself, make every
inherited field an effective vLLM-Omni runtime option. Construction and
runtime consumption are separate surfaces:

| Surface | Current behavior |
| --- | --- |
| Structured schema | Upstream dataclass fields are inherited. Their defaults, default factories, and applicable Pydantic validation can participate when the structured object is constructed. |
| Omni input ownership | Pipeline/deploy/CLI construction continues to accept only fields with an existing structured owner. Direct construction of an inherited sub-config can expose a wider upstream schema. |
| Engine projection | The explicit load, cache, scheduler, and parallel engine-field sets define which values are emitted to flat `EngineArgs`. An inherited field is not guaranteed to affect execution unless it is part of that projection. |
| Terminal materialization | The engine-owning process constructs the final upstream `VllmConfig` and performs model-, platform-, rank-, port-, and backend-dependent initialization. |

The distinction is intentional for this reuse step: upstream schema can be
shared without silently expanding current backend behavior. Adopting an
additional upstream field requires an explicit ownership and projection
decision, plus an effective-engine-argument parity test. Code must not infer
runtime support solely from `dataclasses.fields()` or an inherited constructor
parameter.

Existing nested paths for `CompilationConfig`, `ProfilerConfig`, and
`QuantizationConfigArgs` also accept complete upstream value objects. This is
object pass-through reuse: the outer Omni field remains defined, mappings
remain supported, and prebuilt upstream objects retain their type across the
structured projection boundary.

Diffusion keeps its existing engine-field ownership and projection surface.
Shared Python inheritance must not be treated as evidence that an LLM-only
parallel input is effective for a diffusion stage.

## Safe-change guide while deferred

Changes should trace every affected producer to its runtime consumer and test
the structured, legacy, CLI, and deployment paths that are actually supported.
For inherited vLLM fields, tests should separately cover constructor behavior
and effective `EngineArgs` projection. Do not infer a stable contract from
this placeholder page.

## Promotion gate

- Finish or stabilize the configuration refactor and identify the canonical
  runtime configuration representation.
- Confirm technical owners, primary path exceptions, and validation owners.
- Document parsing and override precedence, including when environment values
  are captured.
- Allocate an invariant namespace only after those behaviors are enforced and
  owner-approved.
