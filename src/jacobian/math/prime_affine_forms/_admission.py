"""Owner-local admission decisions for prime-affine operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.prime_affine_forms._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.prime_affine_forms.interval_count.compute",
        AdmissionDecision.KEEP,
        "exact count of a complete bounded positive-prime affine pattern",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.interval_enumerate.compute",
        AdmissionDecision.KEEP,
        "complete bounded family of positive-prime affine matches and values",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.interval_residue_profile.compute",
        AdmissionDecision.KEEP,
        "complete bounded interval profile for a supplied exact CRT wheel",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.local_admissibility.compute",
        AdmissionDecision.KEEP,
        "closed local-admissibility decision from the primitive-form finite cutoff theorem",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.local_factor.compute",
        AdmissionDecision.KEEP,
        "complete one-prime residue partition and exact Hardy-Littlewood local factor",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.local_factors.compute",
        AdmissionDecision.KEEP,
        "exact finite local-factor family and explicitly finite rational product",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.residue_wheel.compute",
        AdmissionDecision.KEEP,
        "compact exact CRT product with source-bound local residue factors and count",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.residue_wheel.enumerate.compute",
        AdmissionDecision.KEEP,
        "separately bounded complete CRT residue materialization with component transport",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.translation.compute",
        AdmissionDecision.KEEP,
        "typed affine-variable translation preserving stable form identity",
    ),
    OperationAdmission(
        "number_theory.prime_affine_forms.wheel_membership.compute",
        AdmissionDecision.KEEP,
        "exact modular membership with a replayable first local exclusion",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
