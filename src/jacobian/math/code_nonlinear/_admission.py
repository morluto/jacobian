"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.code_nonlinear._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "code.nonlinear.distance_profile.compute",
        AdmissionDecision.KEEP,
        "exact minimum Hamming distance and weight profile by brute-force enumeration",
    ),
    OperationAdmission(
        "code.nonlinear.constant_weight.compute",
        AdmissionDecision.KEEP,
        "exact generation of all constant-weight binary words",
    ),
    OperationAdmission(
        "code.binary.word_distance.compute",
        AdmissionDecision.KEEP,
        "exact Hamming distance between two equal-length binary words",
    ),
    OperationAdmission(
        "code.binary.explicit.profile.compute",
        AdmissionDecision.KEEP,
        "exact complete distance profile with histogram and extremal witnesses",
    ),
    OperationAdmission(
        "code.binary.constant_weight.profile.compute",
        AdmissionDecision.KEEP,
        "exact profile of a constant-weight binary code with support-intersection distances",
    ),
    OperationAdmission(
        "code.binary.explicit.to_set_system.compute",
        AdmissionDecision.NATIVE_ONLY,
        "trivial projection enumerating support indices already supplied by caller",
        native_symbol="jacobian.math.code_nonlinear.to_set_system",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
