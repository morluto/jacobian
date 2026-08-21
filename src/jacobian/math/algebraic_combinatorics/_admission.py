"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.algebraic_combinatorics._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "combinatorics.conjugate_partition.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.conjugate_partition",
    ),
    OperationAdmission(
        "combinatorics.hook_length.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.hook_lengths",
    ),
    OperationAdmission(
        "combinatorics.standard_young_tableaux.count",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.algebraic_combinatorics.standard_young_tableaux_count",
    ),
    OperationAdmission(
        "combinatorics.rsk.permutation.compute",
        AdmissionDecision.KEEP,
        "RSK correspondence with insertion/recording tableaux, shape, and LIS/LDS lengths",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
