"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "arithmetic.real_quadratic.order.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.absolute_value",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.absolute_value",
    ),
    OperationAdmission(
        "integer.compute.decimal_digit_count",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.decimal_digit_sum",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.nth_root",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.sign",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.sign",
    ),
    OperationAdmission(
        "integer.transform.base_digits",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "rational.compute.absolute_value",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.ceiling",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.continued_fraction",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.difference",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.floor",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.maximum",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.minimum",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.negation",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.product",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.compute.quotient",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.quotient",
    ),
    OperationAdmission(
        "rational.compute.reciprocal",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.reciprocal",
    ),
    OperationAdmission(
        "rational.compute.sum",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.arithmetic.sum_rationals",
    ),
    OperationAdmission(
        "rational.decide.equal",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
    OperationAdmission(
        "rational.decide.less_than",
        AdmissionDecision.DROP,
        "ordinary Fraction operation without catalog-level leverage",
    ),
)
