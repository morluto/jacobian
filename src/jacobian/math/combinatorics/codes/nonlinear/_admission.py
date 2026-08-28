"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.codes.nonlinear._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "code.nonlinear.constant_weight.compute",
        AdmissionDecision.KEEP,
        "exact bounded generation of the canonical constant-weight binary code",
    ),
    OperationAdmission(
        "code.binary.word_distance.compute",
        AdmissionDecision.KEEP,
        "exact Hamming distance between two equal-length binary words",
    ),
    OperationAdmission(
        "code.binary.explicit.profile.compute",
        AdmissionDecision.KEEP,
        "exact source-replayed distance profile under derived pair-work and result bounds",
    ),
    OperationAdmission(
        "code.binary.constant_weight.profile.compute",
        AdmissionDecision.KEEP,
        "exact source-replayed distance and intersection profile under derived bounds",
    ),
    OperationAdmission(
        "code.binary.explicit.to_set_system.compute",
        AdmissionDecision.NATIVE_ONLY,
        "source-bound projection to the complete coordinate axis and support blocks",
        native_symbol="jacobian.math.combinatorics.codes.nonlinear.to_set_system",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
