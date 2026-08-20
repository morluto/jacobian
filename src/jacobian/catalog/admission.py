"""Explicit admission decisions for the frozen public math-operation basis.

This module owns admission policy types and curation/validation. The actual
admission row inventory is co-located with each owning math domain in
owner-local ``_admission.py`` modules.

New candidate declarations must receive a reviewed decision before they can
enter the public catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jacobian.catalog.models import MathTools


class AdmissionDecision(StrEnum):
    KEEP = "KEEP"
    NATIVE_ONLY = "NATIVE_ONLY"
    SPLIT = "SPLIT"
    DROP = "DROP"
    CONTRACT_FIX = "CONTRACT_FIX"


@dataclass(frozen=True, slots=True)
class OperationAdmission:
    operation_id: str
    decision: AdmissionDecision
    rationale: str
    native_symbol: str | None = None


@dataclass(frozen=True, slots=True)
class OperationRegistration:
    """One domain-owned unit of candidate tools and their admission decisions."""

    candidates: MathTools
    admissions: tuple[OperationAdmission, ...]

    def __post_init__(self) -> None:
        candidate_ids = tuple(tool.operation_id for tool in self.candidates)
        admission_ids = tuple(record.operation_id for record in self.admissions)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("domain registration candidate IDs must be unique")
        if len(set(admission_ids)) != len(admission_ids):
            raise ValueError("domain registration admission IDs must be unique")
        if set(candidate_ids) != set(admission_ids):
            missing = sorted(set(candidate_ids) - set(admission_ids))
            stale = sorted(set(admission_ids) - set(candidate_ids))
            raise ValueError(
                "domain registration admissions do not match candidates: "
                f"missing={missing}, stale={stale}"
            )


def curate_public_tools(
    candidates: MathTools, admissions: tuple[OperationAdmission, ...]
) -> MathTools:
    """Return only reviewed public operations and fail closed on ledger drift."""

    records = {record.operation_id: record for record in admissions}
    if len(records) != len(admissions):
        raise ValueError("operation admission IDs must be unique")
    candidate_sequence = tuple(tool.operation_id for tool in candidates)
    candidate_ids = set(candidate_sequence)
    if len(candidate_ids) != len(candidate_sequence):
        raise ValueError("candidate operation IDs must be unique")
    record_ids = set(records)
    if candidate_ids != record_ids:
        missing = sorted(candidate_ids - record_ids)
        stale = sorted(record_ids - candidate_ids)
        raise ValueError(
            "operation admission ledger does not match candidates: "
            f"missing={missing}, stale={stale}"
        )
    admitted = {AdmissionDecision.KEEP}
    return tuple(
        tool for tool in candidates if records[tool.operation_id].decision in admitted
    )


def admission_by_id(
    admissions: tuple[OperationAdmission, ...],
) -> dict[str, OperationAdmission]:
    return {record.operation_id: record for record in admissions}


__all__ = [
    "AdmissionDecision",
    "OperationAdmission",
    "OperationRegistration",
    "admission_by_id",
    "curate_public_tools",
]
