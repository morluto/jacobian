"""Supported exact polynomial API."""

from jacobian.math.polynomials._elementary_kernel import (
    integer_polynomial_compose,
    integer_polynomial_content,
    integer_polynomial_evaluate,
    integer_polynomial_gcd,
    integer_polynomial_primitive_part,
    integer_polynomial_shift,
    rational_partial_fraction_decomposition,
    rational_polynomial_derivative,
    rational_polynomial_division,
    rational_polynomial_evaluate,
    rational_polynomial_integral,
)
from jacobian.math.polynomials.operations import (
    derivative,
    discriminant,
    divide,
    evaluate,
    factorization,
    gcdex,
    groebner_basis,
    hermite_reduction,
    integral,
    multiply,
    partial_fractions,
    polynomial_discriminant,
    polynomial_factorization,
    polynomial_gcd,
    polynomial_groebner_basis,
    polynomial_resultant,
    polynomial_square_free_decomposition,
    resultant,
    square_free_decomposition,
    verify_polynomial_discriminant,
    verify_polynomial_factorization,
    verify_polynomial_gcd,
    verify_polynomial_resultant,
    verify_polynomial_square_free_decomposition,
)
from jacobian.math.polynomials.rational_functions.operations import (
    verify_hermite_reduction,
)
from jacobian.math.polynomials.unit_circle import (
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
    unit_circle_arc_energy,
    verify_unit_circle_arc_energy,
)


def __getattr__(name: str) -> object:
    if name not in {
        "ideal_containment",
        "ideal_equality",
        "ideal_membership_certificate",
        "ideal_normal_form",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from jacobian.math.polynomials.ideals import operations

    value = getattr(operations, name)
    globals()[name] = value
    return value


__all__ = [
    "UnitCircleArcEnergyRequest",
    "UnitCircleArcEnergyResult",
    "derivative",
    "discriminant",
    "divide",
    "evaluate",
    "factorization",
    "gcdex",
    "groebner_basis",
    "hermite_reduction",
    "ideal_containment",
    "ideal_equality",
    "ideal_membership_certificate",
    "ideal_normal_form",
    "integer_polynomial_compose",
    "integer_polynomial_content",
    "integer_polynomial_evaluate",
    "integer_polynomial_gcd",
    "integer_polynomial_primitive_part",
    "integer_polynomial_shift",
    "integral",
    "multiply",
    "partial_fractions",
    "polynomial_discriminant",
    "polynomial_factorization",
    "polynomial_gcd",
    "polynomial_groebner_basis",
    "polynomial_resultant",
    "polynomial_square_free_decomposition",
    "rational_partial_fraction_decomposition",
    "rational_polynomial_derivative",
    "rational_polynomial_division",
    "rational_polynomial_evaluate",
    "rational_polynomial_integral",
    "resultant",
    "square_free_decomposition",
    "unit_circle_arc_energy",
    "verify_hermite_reduction",
    "verify_unit_circle_arc_energy",
    "verify_polynomial_discriminant",
    "verify_polynomial_factorization",
    "verify_polynomial_gcd",
    "verify_polynomial_resultant",
    "verify_polynomial_square_free_decomposition",
]
