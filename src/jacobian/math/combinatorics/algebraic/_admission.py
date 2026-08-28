"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.algebraic._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "combinatorics.conjugate_partition.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.combinatorics.algebraic.conjugate_partition",
    ),
    OperationAdmission(
        "combinatorics.hook_length.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.combinatorics.algebraic.hook_lengths",
    ),
    OperationAdmission(
        "combinatorics.standard_young_tableaux.count",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.combinatorics.algebraic.standard_young_tableaux_count",
    ),
    OperationAdmission(
        "combinatorics.rsk.permutation.compute",
        AdmissionDecision.KEEP,
        "source-bound strict permutation RSK with canonical standard tableaux, shape, and derived LIS/LDS lengths",
    ),
    OperationAdmission(
        "tableau.rsk.word.compute",
        AdmissionDecision.KEEP,
        "compact exact word RSK pair with an explicit alphabet and row-insertion convention",
    ),
    OperationAdmission(
        "tableau.rsk.inverse_word.compute",
        AdmissionDecision.KEEP,
        "exact inverse correspondence from a compatible canonical tableau pair to its ordered word",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
