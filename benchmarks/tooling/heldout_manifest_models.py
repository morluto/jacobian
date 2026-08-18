"""Strict held-out bundle manifest models.

These models are the closed, ``extra="forbid"`` validation front door for
held-out manifest files parsed by ``heldout_manifest``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    StrictFloat,
    StrictInt,
    StrictStr,
)

from benchmarks.tooling.strict_boundaries import _StrictModel


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
    catalog_digest: StrictStr


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


__all__ = [
    "HeldoutAgent",
    "HeldoutArchive",
    "HeldoutControlCondition",
    "HeldoutDataset",
    "HeldoutExperiment",
    "HeldoutManifest",
    "HeldoutSnapshotLock",
    "HeldoutStage",
    "HeldoutTask",
    "HeldoutTreatmentCondition",
]
