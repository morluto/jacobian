"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "finite_abelian_group.exact_factorization.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.aliquot_sum",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.divisor_count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.divisor_sum",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.divisors",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.euler_totient",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.extended_gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.floor_square_root",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.gcd",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.lcm",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.mobius",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.next_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.nth_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.previous_prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.prime_count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.prime_factorization",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.primorial",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.compute.proper_divisors",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.radical",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.compute.valuation",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "integer.decide.abundant",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.coprime",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.deficient",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.divides",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.even",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.odd",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.perfect",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.powerful",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.prime",
        AdmissionDecision.KEEP,
        "distinct exact bounded predicate or candidate check with typed semantics",
    ),
    OperationAdmission(
        "integer.decide.square",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "integer.decide.squarefree",
        AdmissionDecision.DROP,
        "ordinary arithmetic or cheap projection of a retained exact factorization/divisor result",
    ),
    OperationAdmission(
        "modular.compute.discrete_logarithm",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.compute.inverse",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.compute.multiplicative_order",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.enumerate.quadratic_residues",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "modular.polynomial_identity.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.polynomial_residue_image.assignments.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.polynomial_residue_image.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "modular.solve.chinese_remainder",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "number_theory.compute.factorial_valuation",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.compute.jacobi_symbol",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "number_theory.compute.legendre_symbol",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)
