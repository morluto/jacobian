"""Strict Harbor task.toml manifest section models.

These models are the closed, ``extra="forbid"`` validation front door for
the task/environment/verifier/agent sections of ``task.toml`` files parsed
by ``harbor_suite``.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    StrictFloat,
    StrictInt,
    StrictStr,
)

from benchmarks.tooling.strict_boundaries import _StrictModel


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


__all__ = [
    "TaskAgentSection",
    "TaskEnvironmentSection",
    "TaskManifestSections",
    "TaskSection",
    "TaskVerifierSection",
]
