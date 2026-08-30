"""Independent defining-invariant checks for embedding-profile tests."""

from __future__ import annotations

from fractions import Fraction
from functools import cmp_to_key
from typing import Any, Literal

from jacobian.canonical import parse_canonical_integer
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
    RationalComplexIsolatingRectangle,
    algebraic_real_part_separation_denominator_bound,
    algebraic_root_separation_denominator_bound,
)
from jacobian.math.number_theory.algebraic_numbers.real import RationalIsolatingInterval
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbedding,
)


def _backend_fraction(value: Any) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _sympy_polynomial(coefficients: tuple[str, ...]) -> Any:
    import sympy

    variable = sympy.Symbol("x")
    return sympy.Poly.from_list(
        [parse_canonical_integer(coefficient) for coefficient in coefficients],
        gens=variable,
        domain=sympy.ZZ,
    )


def _indexed_root_approximation(
    polynomial: Any,
    root_index: int,
    error: Fraction,
) -> tuple[Fraction, Fraction]:
    import sympy

    root = sympy.CRootOf(polynomial.as_expr(), root_index, radicals=False)
    tolerance = sympy.Rational(error.numerator, error.denominator)
    approximation = root.eval_rational(dx=tolerance, dy=tolerance)
    real_part, imaginary_part = approximation.as_real_imag()
    return _backend_fraction(real_part), _backend_fraction(imaginary_part)


def _public_to_backend_root_indices(
    value: ComplexAlgebraicValue,
) -> tuple[int, ...]:
    import sympy

    polynomial = _sympy_polynomial(value.polynomial)
    degree = polynomial.degree()
    real_count = int(polynomial.count_roots(-sympy.oo, sympy.oo))
    if real_count == degree:
        return tuple(range(degree))

    root_separation = algebraic_root_separation_denominator_bound(value.polynomial)
    pair_count = (degree - real_count) // 2
    real_part_separation = (
        algebraic_real_part_separation_denominator_bound(value.polynomial)
        if pair_count > 1
        else 1
    )
    root_error = Fraction(1, 16 * root_separation)
    real_part_error = Fraction(1, 16 * real_part_separation)
    ordering_error = min(root_error, real_part_error)
    backend_roots = tuple(
        sympy.CRootOf(polynomial.as_expr(), index, radicals=False)
        for index in range(degree)
    )
    unused = set(range(real_count, degree))
    positive_representatives: list[tuple[int, Fraction, Fraction]] = []
    negative_for_positive: dict[int, int] = {}
    while unused:
        backend_index = min(unused)
        conjugate = sympy.conjugate(backend_roots[backend_index])
        partner = next(
            candidate
            for candidate in unused
            if candidate != backend_index and backend_roots[candidate] == conjugate
        )
        _real_part, imaginary_part = _indexed_root_approximation(
            polynomial,
            backend_index,
            root_error,
        )
        positive_index = backend_index if imaginary_part > 0 else partner
        negative_index = partner if positive_index == backend_index else backend_index
        positive_real, positive_imaginary = _indexed_root_approximation(
            polynomial,
            positive_index,
            ordering_error,
        )
        positive_representatives.append(
            (positive_index, positive_real, positive_imaginary)
        )
        negative_for_positive[positive_index] = negative_index
        unused.remove(backend_index)
        unused.remove(partner)

    def compare(
        left: tuple[int, Fraction, Fraction],
        right: tuple[int, Fraction, Fraction],
    ) -> int:
        _left_index, left_real, left_imaginary = left
        _right_index, right_real, right_imaginary = right
        if left_real + real_part_error < right_real - real_part_error:
            return -1
        if right_real + real_part_error < left_real - real_part_error:
            return 1
        if left_imaginary + root_error < right_imaginary - root_error:
            return -1
        if right_imaginary + root_error < left_imaginary - root_error:
            return 1
        return -1 if left_imaginary < right_imaginary else 1

    positive_representatives.sort(key=cmp_to_key(compare))
    public_to_backend = list(range(real_count))
    for positive_index, _real_part, _imaginary_part in positive_representatives:
        public_to_backend.extend(
            (negative_for_positive[positive_index], positive_index)
        )
    return tuple(public_to_backend)


def require_rectangle_selects_root(
    value: ComplexAlgebraicValue,
    rectangle: RationalComplexIsolatingRectangle,
) -> Literal["NEGATIVE_IMAGINARY", "POSITIVE_IMAGINARY"]:
    """Check root identity, count, and half-plane for test evidence."""

    import sympy

    polynomial = _sympy_polynomial(value.polynomial)
    root_separation = algebraic_root_separation_denominator_bound(value.polynomial)
    evidence_grid_denominator = 1 << (root_separation.bit_length() + 4)
    error = Fraction(1, 16 * evidence_grid_denominator)
    backend_root_index = _public_to_backend_root_indices(value)[value.root_index]
    real_part, imaginary_part = _indexed_root_approximation(
        polynomial,
        backend_root_index,
        error,
    )
    if not (
        rectangle.real_lower.as_fraction() < real_part - error
        and real_part + error < rectangle.real_upper.as_fraction()
        and rectangle.imaginary_lower.as_fraction() < imaginary_part - error
        and imaginary_part + error < rectangle.imaginary_upper.as_fraction()
    ):
        raise ValueError("complex isolator does not certify the selected indexed root")

    lower = sympy.Rational(*rectangle.real_lower.as_integer_ratio()) + sympy.I * (
        sympy.Rational(*rectangle.imaginary_lower.as_integer_ratio())
    )
    upper = sympy.Rational(*rectangle.real_upper.as_integer_ratio()) + sympy.I * (
        sympy.Rational(*rectangle.imaginary_upper.as_integer_ratio())
    )
    try:
        root_count = int(polynomial.count_roots(lower, upper))
    except NotImplementedError as exc:
        raise ValueError(
            "complex isolator boundaries must contain no additional polynomial root"
        ) from exc
    if root_count != 1:
        raise ValueError("a complex algebraic isolator must contain exactly one root")

    if rectangle.imaginary_upper.as_fraction() < 0:
        return "NEGATIVE_IMAGINARY"
    if rectangle.imaginary_lower.as_fraction() > 0:
        return "POSITIVE_IMAGINARY"
    raise ValueError("a nonreal root isolator must lie wholly in one open half-plane")


def require_real_interval_selects_root(
    embedding: RealNumberFieldEmbedding,
    interval: RationalIsolatingInterval,
) -> None:
    """Check real root identity and count for test evidence."""

    import sympy

    polynomial = _sympy_polynomial(embedding.presentation.coefficients_descending)
    lower = sympy.Rational(*interval.lower.as_integer_ratio())
    upper = sympy.Rational(*interval.upper.as_integer_ratio())
    if strict_root_count(polynomial, lower, upper) != 1:
        raise ValueError("a real embedding interval must isolate exactly one root")
    roots_below = int(polynomial.count_roots(-sympy.oo, lower))
    if polynomial.eval(lower) == 0:
        roots_below -= 1
    if roots_below != embedding.root.real_root_index:
        raise ValueError(
            "real isolating interval does not select the embedding's indexed root"
        )


__all__ = [
    "require_real_interval_selects_root",
    "require_rectangle_selects_root",
]
