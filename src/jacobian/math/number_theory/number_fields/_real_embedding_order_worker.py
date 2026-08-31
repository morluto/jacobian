"""One-shot SymPy kernel for selected-image isolation behind a killable deadline."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.number_fields._real_embedding_order_protocol import (
    SelectedImageWorkerComplete,
    SelectedImageWorkerError,
    SelectedImageWorkerRequest,
)


def _normalize_minimal_polynomial(polynomial: Any) -> Any:
    _denominator, integral = polynomial.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    return -primitive if primitive.LC() < 0 else primitive


def compute_selected_image_isolation(
    request: SelectedImageWorkerRequest,
) -> SelectedImageWorkerComplete | SelectedImageWorkerError:
    import sympy

    field = request.field
    alpha = sympy.Symbol("alpha")
    polynomial = sympy.Poly.from_list(
        [parse_canonical_integer(value) for value in field.coefficients_descending],
        gens=alpha,
        domain=sympy.QQ,
    )
    algebraic_field = sympy.QQ.alg_field_from_poly(
        polynomial,
        alias="alpha",
        root_index=request.real_root_index,
    )

    coefficients = [
        sympy.Rational(Fraction(coefficient))
        for coefficient in request.value_coefficients_descending
    ]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    value = algebraic_field.new(coefficients)

    if value == algebraic_field.zero:
        return SelectedImageWorkerComplete(
            kind="complete",
            order="EQ",
            isolating_interval={
                "lower": "0",
                "upper": "0",
                "interval_type": "SINGLETON",
            },
        )

    descending = list(value.to_list())
    if len(descending) == 1:
        rational = Fraction(
            int(descending[0].numerator), int(descending[0].denominator)
        )
        endpoint = f"{rational.numerator}/{rational.denominator}"
        order = "LT" if rational < 0 else "GT"
        return SelectedImageWorkerComplete(
            kind="complete",
            order=order,
            isolating_interval={
                "lower": endpoint,
                "upper": endpoint,
                "interval_type": "SINGLETON",
            },
        )

    image = algebraic_field.to_sympy(value)
    variable = sympy.Symbol("image")
    minimal_polynomial = _normalize_minimal_polynomial(
        sympy.minpoly(image, variable, polys=True)
    )
    if minimal_polynomial.degree() > field.degree:
        return SelectedImageWorkerError(
            kind="error",
            reason="over_degree_polynomial",
            message="SymPy returned an over-degree selected-image polynomial",
        )
    actual_height = max(
        abs(int(coefficient)) for coefficient in minimal_polynomial.all_coeffs()
    )
    coefficient_bound = parse_canonical_integer(
        request.minimal_polynomial_coefficient_bound
    )
    if actual_height > coefficient_bound:
        return SelectedImageWorkerError(
            kind="error",
            reason="polynomial_height_exceeded",
            message="selected-image polynomial exceeded its admitted height bound",
        )

    real_roots = minimal_polynomial.real_roots(radicals=False)
    matches = tuple(
        index
        for index, root in enumerate(real_roots)
        if minimal_polynomial.same_root(root, image)
    )
    if len(matches) != 1:
        return SelectedImageWorkerError(
            kind="error",
            reason="isolation_not_unique",
            message="exact selected-image isolation did not identify one real root",
        )

    constant = abs(int(minimal_polynomial.TC()))
    if constant == 0:
        return SelectedImageWorkerError(
            kind="error",
            reason="zero_root_polynomial",
            message="a nonzero field element received a zero-root minimal polynomial",
        )
    height = max(
        abs(int(coefficient)) for coefficient in minimal_polynomial.all_coeffs()
    )
    epsilon = sympy.Rational(constant, 2 * (constant + height))
    intervals = minimal_polynomial.intervals(eps=epsilon)
    (lower, upper), _multiplicity = intervals[matches[0]]
    lower_fraction = Fraction(int(lower.p), int(lower.q))
    upper_fraction = Fraction(int(upper.p), int(upper.q))
    if upper_fraction < 0:
        order = "LT"
    elif lower_fraction > 0:
        order = "GT"
    else:
        return SelectedImageWorkerError(
            kind="error",
            reason="sign_not_established",
            message="selected-image isolation did not establish a strict sign",
        )
    return SelectedImageWorkerComplete(
        kind="complete",
        order=order,
        isolating_interval={
            "lower": f"{lower_fraction.numerator}/{lower_fraction.denominator}",
            "upper": f"{upper_fraction.numerator}/{upper_fraction.denominator}",
            "interval_type": (
                "SINGLETON" if lower_fraction == upper_fraction else "OPEN"
            ),
        },
    )


def main() -> int:
    request = SelectedImageWorkerRequest.model_validate_json(
        __import__("sys").stdin.buffer.read(),
        strict=True,
    )
    response = compute_selected_image_isolation(request)
    __import__("sys").stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
