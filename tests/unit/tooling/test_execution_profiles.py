"""Unit tests for resource-dominance execution profiles."""

from __future__ import annotations

from tests.support.execution_profiles import compile_execution_profile
from tests.support.resource_contracts import IsolationClass, ResourceKind


def test_process_resource_dominates_semantic_defaults() -> None:
    profile = compile_execution_profile(
        semantic_owner="composition",
        resources={ResourceKind.COMPLETE_RUNTIME, ResourceKind.PROCESS_GROUP},
        isolation=IsolationClass.PRIVATE_MUTABLE,
    )
    assert profile.name == "process-isolated"
    assert profile.process_supervision is True
    assert profile.workers == 2
