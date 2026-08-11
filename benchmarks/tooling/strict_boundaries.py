"""Strict typed Pydantic boundaries for benchmark control-plane loaders.

These models are the closed, ``extra="forbid"`` validation front door for the
four authored configuration roots that the benchmark tooling parses:

* Harbor job dataset/task selections (``benchmark_contracts``),
* Harbor ``task.toml`` task/environment sections (``harbor_suite``),
* held-out run plans and run entries (``heldout_runner``), and
* observation job normalization roots (``observation_results``).

Each model uses strict scalar fields (``StrictBool``/``StrictInt``/``StrictStr``
where a scalar is expected) and rejects unknown fields.  Loaders validate raw
payloads through these models *before* any semantic ``.get``/iteration/set
conversion, so a structurally malformed config fails closed with a field-path
diagnostic instead of reaching backend or artifact side effects.

Diagnostics are returned as ``HarborSuiteError`` messages carrying a
``field.path`` prefix and a short safe code, never the raw input value.  The
helpers never import product internals; ``benchmarks.tooling`` stays
self-contained.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)

from benchmarks.tooling.errors import HarborSuiteError


class _StrictModel(BaseModel):
    """Closed base: forbid extras and reject loose scalar coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# Harbor job dataset/task selections
# ---------------------------------------------------------------------------


class HarborJobDatasetEntry(_StrictModel):
    path: StrictStr
    task_names: list[StrictStr] | None = None


class HarborJobTaskEntry(_StrictModel):
    path: StrictStr


class HarborJobSelection(_StrictModel):
    """Exactly one of ``datasets`` or ``tasks``; validated structurally first."""

    datasets: list[HarborJobDatasetEntry] | None = None
    tasks: list[HarborJobTaskEntry] | None = None


# ---------------------------------------------------------------------------
# Harbor task.toml task/environment sections
# ---------------------------------------------------------------------------


class TaskEnvironmentSection(_StrictModel):
    network_mode: StrictStr
    cpus: StrictInt | None = None
    memory_mb: StrictInt | None = None
    storage_mb: StrictInt | None = None


class TaskVerifierSection(_StrictModel):
    timeout_sec: StrictFloat | None = None
    environment_mode: StrictStr | None = None
    environment: TaskEnvironmentSection | None = None


class TaskAgentSection(_StrictModel):
    timeout_sec: StrictFloat | None = None


class TaskSection(_StrictModel):
    name: StrictStr
    version: StrictStr
    description: StrictStr | None = None
    keywords: list[StrictStr] | None = None


class TaskManifestSections(_StrictModel):
    """Strict projection of the structural ``task.toml`` sections.

    The ``metadata`` section is intentionally not type-closed here: its schema
    keeps ``additionalProperties: true`` for provenance fields, so it stays on
    the existing metadata validation path.  Only the task/environment/verifier/
    agent sections are closed.  ``artifacts`` is admitted as a list of artifact
    URIs so the top-level ``extra="forbid"`` does not reject it.
    """

    schema_version: StrictStr
    task: TaskSection
    artifacts: list[StrictStr] | None = None
    metadata: dict[StrictStr, Any] | None = None
    agent: TaskAgentSection | None = None
    environment: TaskEnvironmentSection | None = None
    verifier: TaskVerifierSection | None = None


# ---------------------------------------------------------------------------
# Held-out run plans and run entries
# ---------------------------------------------------------------------------


class HeldoutRunBudget(_StrictModel):
    max_tokens: StrictInt
    max_cost_usd: StrictFloat
    enforcement: StrictStr
    missing_accounting: StrictStr
    overage: StrictStr


class HeldoutRunEntry(_StrictModel):
    pair_id: StrictStr
    condition: Literal["C1", "C2"]
    job: StrictStr
    jobs_dir: StrictStr
    pair_index: StrictInt | None = None
    task: StrictStr | None = None
    repetition: StrictInt | None = None
    jacobian_enabled: StrictBool | None = None
    runtime_snapshot: StrictStr | None = None


class HeldoutRunPlan(_StrictModel):
    schema_version: Literal["3"]
    manifest_digest: StrictStr
    pair_count: StrictInt
    budget: HeldoutRunBudget
    runs: list[HeldoutRunEntry]
    plan_digest: StrictStr
    stage: StrictStr | None = None


# ---------------------------------------------------------------------------
# Held-out manifest (v3)
# ---------------------------------------------------------------------------


class HeldoutSnapshotLock(_StrictModel):
    lock_id: StrictStr
    lock_uri: StrictStr
    lock_digest: StrictStr


class HeldoutArchive(_StrictModel):
    uri: StrictStr
    sha256: StrictStr


class HeldoutDataset(_StrictModel):
    id: StrictStr
    path: StrictStr
    manifest_digest: StrictStr
    minimum_independent_families: StrictInt


class HeldoutTask(_StrictModel):
    id: StrictStr
    family: StrictStr
    digest: StrictStr
    verifier_root: StrictStr
    verifier_tree_digest: StrictStr
    oracle_root: StrictStr
    oracle_tree_digest: StrictStr


class HeldoutControlCondition(_StrictModel):
    id: Literal["C1"]
    role: Literal["PRIMARY_CONTROL"]
    jacobian_enabled: Literal[False] = False


class HeldoutTreatmentCondition(_StrictModel):
    id: Literal["C2"]
    role: Literal["PRIMARY_TREATMENT"]
    jacobian_enabled: Literal[True] = True
    image: StrictStr
    source_sha: StrictStr
    platform: StrictStr
    server_version: StrictStr
    policy_profile: StrictStr
    catalog_digest: StrictStr
    policy_digest: StrictStr


class HeldoutAgent(_StrictModel):
    name: Literal["codex"]
    version: StrictStr


class HeldoutStage(_StrictModel):
    task_ids: list[StrictStr]
    repetitions: StrictInt


class HeldoutExperiment(_StrictModel):
    harbor_version: Literal["0.20.0"]
    agent: HeldoutAgent
    model: StrictStr
    prompt_path: StrictStr
    prompt_digest: StrictStr
    reasoning_effort: StrictStr
    randomization_seed: StrictInt
    max_tokens: StrictInt
    max_cost_usd: StrictFloat
    stages: dict[StrictStr, HeldoutStage]


class HeldoutManifest(_StrictModel):
    schema_version: Literal["3"]
    bundle_id: StrictStr
    bundle_version: StrictStr
    snapshot_lock: HeldoutSnapshotLock
    archive: HeldoutArchive
    dataset: HeldoutDataset
    tasks: list[HeldoutTask]
    conditions: list[HeldoutControlCondition | HeldoutTreatmentCondition]
    experiment: HeldoutExperiment


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def format_strict_errors(exc: ValidationError, *, label: str) -> list[str]:
    """Render a Pydantic ``ValidationError`` as field-path/code diagnostics.

    Each diagnostic carries the field path (relative to *label*), the Pydantic
    error message, and the short error type code.  Raw input values are never
    included, so the diagnostic is safe to surface through ``HarborSuiteError``.
    """

    failures: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        path = f"{label}.{loc}" if loc else label
        failures.append(f"{path}: {error['msg']} ({error['type']})")
    return failures


def strict_model_failures(
    model: type[BaseModel],
    payload: Any,
    *,
    label: str,
) -> list[str]:
    """Validate *payload* against *model* and return field-path diagnostics."""

    try:
        model.model_validate(payload)
    except ValidationError as exc:
        return format_strict_errors(exc, label=label)
    return []


def raise_strict_model(
    model: type[BaseModel],
    payload: Any,
    *,
    label: str,
) -> BaseModel:
    """Validate *payload* and raise ``HarborSuiteError`` on structural failure."""

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        failures = format_strict_errors(exc, label=label)
        raise HarborSuiteError(
            f"{label}: strict configuration validation failed: " + "; ".join(failures)
        ) from exc


__all__ = [
    "HarborJobDatasetEntry",
    "HarborJobSelection",
    "HarborJobTaskEntry",
    "HeldoutAgent",
    "HeldoutArchive",
    "HeldoutControlCondition",
    "HeldoutDataset",
    "HeldoutExperiment",
    "HeldoutManifest",
    "HeldoutRunBudget",
    "HeldoutRunEntry",
    "HeldoutRunPlan",
    "HeldoutSnapshotLock",
    "HeldoutStage",
    "HeldoutTask",
    "HeldoutTreatmentCondition",
    "TaskAgentSection",
    "TaskEnvironmentSection",
    "TaskManifestSections",
    "TaskSection",
    "TaskVerifierSection",
    "format_strict_errors",
    "raise_strict_model",
    "strict_model_failures",
]
