"""Compile semantic x resource x profile into an execution profile.

Scaffolding for the orthogonal lane model (#1164 / #1167 / #1169). Full
scheduler integration lands after fixture contracts are universal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tests.support.resource_contracts import IsolationClass, ResourceKind

Scheduler = Literal["none", "worksteal", "load"]


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: str
    workers: int
    distribution: Scheduler
    timeout_seconds: int
    process_supervision: bool
    sqlite_serial: bool
    setup_affinity: str | None = None


_RESOURCE_DOMINANCE: dict[ResourceKind, ExecutionProfile] = {
    ResourceKind.PROCESS_GROUP: ExecutionProfile(
        name="process-isolated",
        workers=2,
        distribution="worksteal",
        timeout_seconds=120,
        process_supervision=True,
        sqlite_serial=False,
        setup_affinity="process-group",
    ),
    ResourceKind.SQLITE: ExecutionProfile(
        name="sqlite-serial",
        workers=0,
        distribution="none",
        timeout_seconds=120,
        process_supervision=False,
        sqlite_serial=True,
        setup_affinity="sqlite",
    ),
    ResourceKind.LEAN: ExecutionProfile(
        name="lean-serial",
        workers=0,
        distribution="none",
        timeout_seconds=300,
        process_supervision=True,
        sqlite_serial=False,
        setup_affinity="lean",
    ),
}


def compile_execution_profile(
    *,
    semantic_owner: str,
    resources: set[ResourceKind] | frozenset[ResourceKind],
    isolation: IsolationClass,
    default_workers: int = 2,
    default_timeout: int = 120,
    default_distribution: Scheduler = "worksteal",
) -> ExecutionProfile:
    """Apply resource dominance rules over a semantic lane default."""

    del isolation  # reserved for share-scope decisions
    for resource in (
        ResourceKind.LEAN,
        ResourceKind.PROCESS_GROUP,
        ResourceKind.SQLITE,
    ):
        if resource in resources:
            return _RESOURCE_DOMINANCE[resource]
    return ExecutionProfile(
        name=f"{semantic_owner}-default",
        workers=default_workers,
        distribution=default_distribution,
        timeout_seconds=default_timeout,
        process_supervision=ResourceKind.PROCESS_GROUP in resources,
        sqlite_serial=ResourceKind.SQLITE in resources,
        setup_affinity=None,
    )
