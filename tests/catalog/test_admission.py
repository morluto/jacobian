"""Fail-closed admission checks for the curated public operation basis."""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path

import pytest

from jacobian.catalog.admission import (
    REVIEWED_BASE_REVISION,
    AdmissionDecision,
    curate_public_tools,
)
from jacobian.catalog.builtins import (
    _ALL_ADMISSIONS,
    _BUILTIN_CANDIDATES,
    BUILTIN_TOOLS,
)

OPERATION_ADMISSIONS = _ALL_ADMISSIONS


def test_every_frozen_candidate_has_exactly_one_admission_decision() -> None:
    candidate_ids = [tool.operation_id for tool in _BUILTIN_CANDIDATES]
    reviewed_ids = [record.operation_id for record in OPERATION_ADMISSIONS]

    assert REVIEWED_BASE_REVISION == "61589543bbbff546edbc51d34a07887982fa4ad6"
    assert len(candidate_ids) == len(set(candidate_ids)) == 406
    assert reviewed_ids == sorted(reviewed_ids)
    assert set(reviewed_ids) == set(candidate_ids)
    assert all(record.rationale.strip() for record in OPERATION_ADMISSIONS)
    assert Counter(record.decision for record in OPERATION_ADMISSIONS) == {
        AdmissionDecision.KEEP: 246,
        AdmissionDecision.NATIVE_ONLY: 56,
        AdmissionDecision.DROP: 104,
    }


def test_public_catalog_contains_only_admitted_atomic_operations() -> None:
    expected = {
        record.operation_id
        for record in OPERATION_ADMISSIONS
        if record.decision is AdmissionDecision.KEEP
    }

    assert {tool.operation_id for tool in BUILTIN_TOOLS} == expected
    assert len(BUILTIN_TOOLS) == 246


def test_catalog_construction_fails_closed_on_duplicate_candidates() -> None:
    duplicate_candidates = (*_BUILTIN_CANDIDATES, _BUILTIN_CANDIDATES[0])

    with pytest.raises(ValueError, match="candidate operation IDs must be unique"):
        curate_public_tools(duplicate_candidates, _ALL_ADMISSIONS)


def test_native_only_decisions_resolve_to_supported_public_symbols() -> None:
    native_records = [
        record
        for record in OPERATION_ADMISSIONS
        if record.decision is AdmissionDecision.NATIVE_ONLY
    ]

    assert native_records
    for record in native_records:
        assert record.native_symbol is not None
        module_name, _, symbol_name = record.native_symbol.rpartition(".")
        module = importlib.import_module(module_name)
        assert symbol_name in module.__all__, record.operation_id
        assert callable(getattr(module, symbol_name)), record.operation_id


def test_non_public_decisions_cannot_leak_into_discovery() -> None:
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    excluded = {
        record.operation_id
        for record in OPERATION_ADMISSIONS
        if record.decision
        in {
            AdmissionDecision.NATIVE_ONLY,
            AdmissionDecision.SPLIT,
            AdmissionDecision.DROP,
        }
    }

    assert public_ids.isdisjoint(excluded)


def test_public_guidance_does_not_advertise_excluded_operation_ids() -> None:
    excluded = {
        record.operation_id
        for record in OPERATION_ADMISSIONS
        if record.decision is not AdmissionDecision.KEEP
    }
    exempt_documents = {"public-operation-admission.md"}

    leaks: dict[str, list[str]] = {}
    for path in sorted(Path("docs").rglob("*.md")):
        if path.name in exempt_documents:
            continue
        found = sorted(
            operation_id
            for operation_id in excluded
            if operation_id in path.read_text()
        )
        if found:
            leaks[str(path)] = found

    assert not leaks
