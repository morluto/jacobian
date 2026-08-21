"""Fail-closed admission checks for the curated public operation basis."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationRegistration,
    curate_public_tools,
)
from jacobian.catalog.builtins import (
    _ALL_ADMISSIONS,
    _BUILTIN_CANDIDATES,
    _BUILTIN_REGISTRATION_MODULES,
    _BUILTIN_REGISTRATIONS,
    BUILTIN_TOOLS,
)

OPERATION_ADMISSIONS = _ALL_ADMISSIONS


def test_registration_discovery_is_deterministic_and_owner_local() -> None:
    assert tuple(sorted(_BUILTIN_REGISTRATION_MODULES)) == _BUILTIN_REGISTRATION_MODULES
    assert len(_BUILTIN_REGISTRATION_MODULES) == len(set(_BUILTIN_REGISTRATION_MODULES))
    assert all(
        module_name.startswith("jacobian.math.") and module_name.endswith("._admission")
        for module_name in _BUILTIN_REGISTRATION_MODULES
    )
    assert len(_BUILTIN_REGISTRATION_MODULES) == len(_BUILTIN_REGISTRATIONS)


def test_every_candidate_has_exactly_one_admission_decision() -> None:
    candidate_ids = [tool.operation_id for tool in _BUILTIN_CANDIDATES]
    reviewed_ids = [record.operation_id for record in OPERATION_ADMISSIONS]

    assert len(candidate_ids) == len(set(candidate_ids))
    assert reviewed_ids == sorted(reviewed_ids)
    assert set(reviewed_ids) == set(candidate_ids)
    assert all(record.rationale.strip() for record in OPERATION_ADMISSIONS)


def test_each_domain_owns_its_complete_registration() -> None:
    for registration in _BUILTIN_REGISTRATIONS:
        candidate_ids = {tool.operation_id for tool in registration.candidates}
        admission_ids = {record.operation_id for record in registration.admissions}

        assert candidate_ids == admission_ids


def test_domain_registration_rejects_admission_owned_by_another_domain() -> None:
    candidate = _BUILTIN_CANDIDATES[0]
    unrelated_admission = next(
        record
        for record in _ALL_ADMISSIONS
        if record.operation_id != candidate.operation_id
    )

    with pytest.raises(
        ValueError, match="domain registration admissions do not match candidates"
    ):
        OperationRegistration((candidate,), (unrelated_admission,))


def test_public_catalog_contains_only_admitted_atomic_operations() -> None:
    expected = {
        record.operation_id
        for record in OPERATION_ADMISSIONS
        if record.decision is AdmissionDecision.KEEP
    }

    assert {tool.operation_id for tool in BUILTIN_TOOLS} == expected


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
