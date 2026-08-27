"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "congruence.periodic_union.measure.compute",
        AdmissionDecision.KEEP,
        "distinct exact source-bound union count and density with a compressed generalized-CRT regime that avoids materializing a large common period",
    ),
    OperationAdmission(
        "congruence.periodic_union.profile.compute",
        AdmissionDecision.KEEP,
        "distinct complete exact finite-period union profile with composable count, density, and residues",
    ),
    OperationAdmission(
        "integer.factor.certified_compute",
        AdmissionDecision.KEEP,
        "distinct bounded subexponential factorization with per-factor Pratt primality certificates",
    ),
    OperationAdmission(
        "integer.primality.certificate.compute",
        AdmissionDecision.KEEP,
        "distinct bounded Pratt primality certificate with independent verification",
    ),
    OperationAdmission(
        "finite_abelian_group.exact_factorization.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "finite_abelian_group.spectral_pair.decide",
        AdmissionDecision.KEEP,
        "exact spectral-basis predicate not supplied by Boolean Walsh transforms or additive factorization",
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
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage; "
        "re-admitted for the v4 result-derived envelope: PrimorialRequest bounds n <= 1001 from the declared "
        "3,400-digit result budget (primorial(1001) carries 3397 digits, primorial(1002) 3401), so admitted "
        "work is n-1 big-integer multiplications whose intermediates and exact result stay inside that budget",
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
        AdmissionDecision.KEEP,
        "bounded partial-factor and rough-cofactor certificate decides 25-digit inputs without complete factorization",
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
    OperationAdmission(
        "number_theory.ramanujan_sum.compute",
        AdmissionDecision.KEEP,
        "classical exact arithmetic function whose factorization-backed evaluation replaces a variable-length composition of divisor and Mobius calls",
    ),
    OperationAdmission(
        "number_theory.friable.count.compute",
        AdmissionDecision.KEEP,
        "complete exact friable count with result-sensitive materialized and generated regimes, avoiding caller-side factorization of every source integer",
    ),
    OperationAdmission(
        "number_theory.gcd_quotient.profile.compute",
        AdmissionDecision.KEEP,
        "distinct complete pairwise gcd profile that a loop of scalar gcd calls cannot establish as one source-bound value",
    ),
    OperationAdmission(
        "number_theory.product_divisibility.profile.compute",
        AdmissionDecision.KEEP,
        "distinct complete pairwise divisibility profile that a loop of scalar divides checks cannot establish as one source-bound value",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
