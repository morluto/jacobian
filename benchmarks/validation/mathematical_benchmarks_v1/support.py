"""Backward-compatible facade for the mathematical-benchmarks-v1 validation harness.

The implementation is split into focused modules:

- ``_paths``: shared dataset path and suite references;
- ``_metadata``: task-local verifier contract metadata loading and validation;
- ``_fixtures``: task catalog constants and canonical fixture preparation;
- ``_verifier``: verifier execution and failure normalization.

This module re-exports the previous public surface so existing leaf and
generic tests that import ``support`` keep working unchanged. New code should
import from the focused module that owns the behavior.
"""

from __future__ import annotations

from benchmarks.validation.mathematical_benchmarks_v1._fixtures import (
    RATIONAL_TASK,
    RESOURCE_DERIVED_TASKS,
    SINGLE_EVIDENCE_TASKS,
    VERIFICATION_RECORD_TASKS,
    VERIFIER_TASKS,
    _bind_result_evidence,
    _digest,
    _prepare_case,
    _task,
    _task_tree_snapshot,
    _write_json,
)
from benchmarks.validation.mathematical_benchmarks_v1._metadata import (
    is_input_binding_decoupled,
    is_scope_independent_assurance,
    load_task_contract_metadata,
)
from benchmarks.validation.mathematical_benchmarks_v1._paths import (
    AGENT_TASKS,
    ROOT,
    TASKS,
)
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

__all__ = [
    "AGENT_TASKS",
    "RATIONAL_TASK",
    "RESOURCE_DERIVED_TASKS",
    "ROOT",
    "SINGLE_EVIDENCE_TASKS",
    "TASKS",
    "VERIFICATION_RECORD_TASKS",
    "VERIFIER_TASKS",
    "_bind_result_evidence",
    "_digest",
    "_prepare_case",
    "_run_verifier",
    "_task",
    "_task_tree_snapshot",
    "_write_json",
    "is_input_binding_decoupled",
    "is_scope_independent_assurance",
    "load_task_contract_metadata",
]
