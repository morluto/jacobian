"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "combinatorics.compute.bell",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.bell_number",
    ),
    OperationAdmission(
        "combinatorics.compute.bernoulli",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.bernoulli_number",
    ),
    OperationAdmission(
        "combinatorics.compute.binomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.catalan",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.catalan_number",
    ),
    OperationAdmission(
        "combinatorics.compute.central_binomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.compositions",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.derangements",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.derangement_number",
    ),
    OperationAdmission(
        "combinatorics.compute.double_factorial",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.double_factorial",
    ),
    OperationAdmission(
        "combinatorics.compute.factorial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.fibonacci",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.fibonacci_number",
    ),
    OperationAdmission(
        "combinatorics.compute.fibonacci_pair",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.lucas",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.lucas_number",
    ),
    OperationAdmission(
        "combinatorics.compute.motzkin",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.motzkin_number",
    ),
    OperationAdmission(
        "combinatorics.compute.multinomial",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.partition_number",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.partition_number",
    ),
    OperationAdmission(
        "combinatorics.compute.permutations",
        AdmissionDecision.DROP,
        "ordinary scalar or finite enumeration better authored directly in Python",
    ),
    OperationAdmission(
        "combinatorics.compute.stirling_first",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.stirling_first",
    ),
    OperationAdmission(
        "combinatorics.compute.stirling_second",
        AdmissionDecision.NATIVE_ONLY,
        "useful classical number retained without a scalar catalog slot",
        native_symbol="jacobian.math.combinatorics.stirling_second",
    ),
    OperationAdmission(
        "combinatorics.cyclic_difference_set.extension.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.cyclic_difference_set.perfect.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.enumerate.integer_partitions",
        AdmissionDecision.NATIVE_ONLY,
        "useful finite enumeration retained without a scalar-family catalog slot",
        native_symbol="jacobian.math.combinatorics.integer_partitions",
    ),
    OperationAdmission(
        "combinatorics.generating_function.coefficients.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.integer_set.sidon.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "combinatorics.recurrence.linear.evaluate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.recurrence.p_recursive.evaluate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "combinatorics.recurrence.p_recursive.table_residuals.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
