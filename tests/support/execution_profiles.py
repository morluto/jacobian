"""Bridge fixture ResourceKind sets into planner execution profiles."""

from __future__ import annotations

from tests.support.resource_contracts import IsolationClass, ResourceKind
from tools.test_plan.execution_profiles import (
    ExecutionProfile,
    compile_execution_profile as _compile_execution_profile,
)

__all__ = [
    "ExecutionProfile",
    "compile_execution_profile",
]


def compile_execution_profile(
    *,
    semantic_owner: str,
    resources: set[ResourceKind] | frozenset[ResourceKind],
    isolation: IsolationClass,
    default_workers: int = 2,
    default_timeout: int = 120,
    default_distribution: str = "worksteal",
) -> ExecutionProfile:
    """Apply resource dominance using fixture ResourceKind values."""

    del isolation
    return _compile_execution_profile(
        semantic_owner=semantic_owner,
        resources={resource.value for resource in resources},
        default_workers=default_workers,
        default_timeout=default_timeout,
        default_distribution=default_distribution,  # type: ignore[arg-type]
    )
