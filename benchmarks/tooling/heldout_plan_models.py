"""Strict held-out run-plan models.

These models are the closed, ``extra="forbid"`` validation front door for
held-out run plan files parsed by ``heldout_runner``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from benchmarks.tooling.strict_boundaries import _StrictModel


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


__all__ = [
    "HeldoutRunBudget",
    "HeldoutRunEntry",
    "HeldoutRunPlan",
]
