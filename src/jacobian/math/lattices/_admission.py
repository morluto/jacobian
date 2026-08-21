"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.lattices._tools import TOOLS

_ADMISSION_RATIONALE = (
    "distinct exact bounded mathematical value or invariant with material "
    "computational or reliability leverage"
)

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "lattice.basis.reduce",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.hermite_normal_form.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.rank_gram.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.canonical_basis.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.dual.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.saturation.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.sublattice_index.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.discriminant_group.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.orthogonal_complement.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.direct_sum.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
    OperationAdmission(
        "lattice.orthogonal_sum.compute",
        AdmissionDecision.KEEP,
        _ADMISSION_RATIONALE,
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
"""Bounded lattice-reduction and integer normal-form operations."""
