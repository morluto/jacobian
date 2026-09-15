"""Exact primitive-element splitting-field construction and field arithmetic.

The splitting field of the square-free support of ``p * p'`` is presented by
one primitive element ``theta`` with monic minimal polynomial over QQ. Every
distinct root is an exact rational polynomial in ``theta``, and complex
conjugation is represented as the exact Q-algebra automorphism of the field
sending ``theta`` to its conjugate element. All arithmetic here is exact
rational polynomial arithmetic reduced modulo the defining polynomial.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_SPLITTING_FIELD_DEGREE = 16


def _error(reason: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("polynomial",),
        code=f"polynomial.root_critical.{reason}",
        message=message,
    )


def _fraction(value: Any) -> Fraction:
    rational = value
    return Fraction(int(rational.p), int(rational.q))


def _ascending_coefficients(items: Any) -> list[Fraction]:
    """Convert SymPy falling-power ``coeffs()`` into ascending coefficients."""

    return list(reversed([_fraction(item) for item in items]))


def reduce_modulus(
    coefficients: list[Fraction], modulus: list[Fraction]
) -> list[Fraction]:
    """Reduce ascending coefficients modulo a monic polynomial."""

    degree = len(modulus) - 1
    reduced = list(coefficients) + [Fraction(0)] * max(0, degree - len(coefficients))
    while len(reduced) > degree:
        top = reduced.pop()
        if top:
            offset = len(reduced) - degree
            for position in range(degree):
                reduced[offset + position] -= top * modulus[position]
    return reduced + [Fraction(0)] * (degree - len(reduced))


def multiply(
    left: list[Fraction], right: list[Fraction], modulus: list[Fraction]
) -> list[Fraction]:
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        if not left_value:
            continue
        for right_index, right_value in enumerate(right):
            if right_value:
                product[left_index + right_index] += left_value * right_value
    return reduce_modulus(product, modulus)


def add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    size = max(len(left), len(right))
    padded_left = list(left) + [Fraction(0)] * (size - len(left))
    padded_right = list(right) + [Fraction(0)] * (size - len(right))
    return [a + b for a, b in zip(padded_left, padded_right, strict=True)]


def subtract(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    return add(left, [-value for value in right])


def conjugate_element(
    element: list[Fraction], conjugation: list[Fraction], modulus: list[Fraction]
) -> list[Fraction]:
    """Apply the conjugation automorphism ``theta -> conjugation`` to ``element``."""

    degree = len(modulus) - 1
    total = [Fraction(0)] * degree
    power: list[Fraction] = [Fraction(1)] + [Fraction(0)] * (degree - 1)
    for coefficient in element:
        if coefficient:
            total = add(total, [coefficient * value for value in power])
        power = multiply(power, conjugation, modulus)
    return reduce_modulus(total, modulus)




def _primitive_integer_monic(sympy: Any, polynomial: Any, variable: Any) -> Any:
    """Return a monic integer ``Poly`` equal up to the zero set of ``polynomial``."""

    cleared = sympy.expand(polynomial.as_expr() * sympy.denom(polynomial.LC()))
    integer_poly = sympy.Poly(cleared, variable)
    coefficients = [int(value) for value in integer_poly.all_coeffs()]
    content = 0
    for value in coefficients:
        content = gcd(content, abs(value))
    if content > 1:
        coefficients = [value // content for value in coefficients]
    if coefficients[0] < 0:
        coefficients = [-value for value in coefficients]
    return sympy.Poly(sum(c * variable ** (len(coefficients) - 1 - i) for i, c in enumerate(coefficients)), variable)


def _polynomial_from_ascending(
    coefficients: list[Fraction], variable: str
) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(value),
            exponents=(power,),
        )
        for power, value in reversed(tuple(enumerate(coefficients)))
        if value
    )
    return RationalPolynomial(
        variables=(variable,), polynomial=SparseRationalPolynomial(terms=terms)
    )


def _canonical(coefficients: list[Fraction]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(value) for value in coefficients)


def rectangle_for_root(root: Any) -> Any:
    """Return an exact isolating rectangle for one SymPy algebraic root."""

    from jacobian.math.polynomials.root_critical.operations import _isolating_rectangle

    return _isolating_rectangle(root, None, ())


def compute_splitting_field(
    polynomial: RationalPolynomial,
    embedding_index: int,
) -> dict[str, object]:
    """Build one bounded exact splitting field for the square-free ``p * p'``."""

    import sympy

    if len(polynomial.variables) != 1:
        raise _error("univariate", "a splitting field needs one polynomial variable")
    if type(embedding_index) is not int or embedding_index < 0:
        raise _error("embedding_index", "the embedding index must be nonnegative")
    variable_name = polynomial.variables[0]
    variable = sympy.Symbol(variable_name)
    expression = sum(
        term.coefficient.as_fraction() * variable ** term.exponents[0]
        for term in polynomial.polynomial.terms
    )
    source = sympy.Poly(expression, variable)
    if source.degree() < 1:
        raise _error("constant", "a splitting field needs a nonconstant polynomial")
    if source.degree() > 8:
        raise _error(
            "degree_bound", "the splitting-field source admits degree at most eight"
        )
    derivative = source.diff(variable)
    product = sympy.Poly(source.as_expr() * derivative.as_expr(), variable)
    repeated = sympy.gcd(product, product.diff(variable))
    support = sympy.cancel(product.as_expr() / repeated.as_expr())
    support_monic = _primitive_integer_monic(sympy, sympy.Poly(support, variable), variable)
    roots = support_monic.all_roots()
    if not roots:
        raise _error("no_roots", "the splitting-field support has no roots")

    from sympy.polys.numberfields import to_number_field

    primitive = to_number_field(roots)
    theta = primitive.to_root()
    modulus = [_fraction(value) for value in reversed(primitive.minpoly.all_coeffs())]
    degree = len(modulus) - 1
    if degree > MAX_SPLITTING_FIELD_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.root_critical.splitting_field_degree",
            message="the exact splitting-field degree exceeds the admitted bound",
        )
    root_coefficients = [
        _ascending_coefficients(to_number_field(root, theta).coeffs()) for root in roots
    ]
    conjugation = _ascending_coefficients(
        to_number_field(sympy.conjugate(theta), theta).coeffs()
    )

    def pad(values: list[Fraction]) -> list[Fraction]:
        return list(values) + [Fraction(0)] * (degree - len(values))

    root_coefficients = [pad(values) for values in root_coefficients]
    conjugation = pad(conjugation)
    embedding_roots = primitive.minpoly.all_roots()
    if embedding_index >= len(embedding_roots):
        raise _error(
            "embedding_index",
            "the embedding index exceeds the number of field embeddings",
        )
    return {
        "squarefree_support": _polynomial_from_ascending(
            [_fraction(value) for value in reversed(support_monic.all_coeffs())],
            variable_name,
        ),
        "defining_polynomial": _polynomial_from_ascending(modulus, "t"),
        "conjugation_coefficients": _canonical(conjugation),
        "embedding_index": embedding_index,
        "embedding_rectangle": rectangle_for_root(embedding_roots[embedding_index]),
        "roots": [
            {
                "axis_index": index,
                "coefficients_ascending": _canonical(coefficients),
                "multiplicity": 1,
                "rectangle": rectangle_for_root(root),
            }
            for index, (root, coefficients) in enumerate(
                zip(roots, root_coefficients, strict=True)
            )
        ],
        "_theta": theta,
        "_modulus": modulus,
        "_conjugation": conjugation,
    }


def compute_bound_profile(
    polynomial: RationalPolynomial,
    *,
    embedding_index: int,
    max_pair_rows: object,
) -> dict[str, object]:
    """Compute the splitting field and the exact distances inside it."""

    from sympy.polys.numberfields import to_number_field

    from jacobian.math.polynomials.root_critical.operations import (
        _admit,
        _compute_profile,
        _family,
    )

    source, _root_count, _critical_count = _admit(
        polynomial, max_pair_rows=max_pair_rows
    )
    profile = _compute_profile(polynomial, max_pair_rows=max_pair_rows)
    field_data = compute_splitting_field(polynomial, embedding_index)
    theta = field_data["_theta"]
    modulus = cast(list[Fraction], field_data["_modulus"])
    conjugation = cast(list[Fraction], field_data["_conjugation"])

    derivative_backend = source.diff()
    _root_records, root_values = _family(source)
    if derivative_backend.degree() > 0:
        _critical_records, critical_values = _family(derivative_backend)
    else:
        critical_values = ()
    root_coefficients = [
        _ascending_coefficients(to_number_field(value, theta).coeffs())
        for value in root_values
    ]
    critical_coefficients = [
        _ascending_coefficients(to_number_field(value, theta).coeffs())
        for value in critical_values
    ]

    def pad(values: list[Fraction]) -> list[Fraction]:
        return list(values) + [Fraction(0)] * (len(modulus) - 1 - len(values))

    root_coefficients = [pad(values) for values in root_coefficients]
    critical_coefficients = [pad(values) for values in critical_coefficients]
    distance_coefficients: list[list[Fraction]] = []
    for root_element in root_coefficients:
        for critical_element in critical_coefficients:
            difference = subtract(root_element, critical_element)
            conjugate = conjugate_element(difference, conjugation, modulus)
            distance_coefficients.append(
                reduce_modulus(multiply(difference, conjugate, modulus), modulus)
            )
    field_data["root_field_coefficients"] = _canonical_many(root_coefficients)
    field_data["critical_field_coefficients"] = _canonical_many(critical_coefficients)
    field_data["distance_field_coefficients"] = _canonical_many(distance_coefficients)
    field_data["profile"] = profile
    return field_data


def _canonical_many(values: list[list[Fraction]]) -> tuple[tuple[CanonicalRational, ...], ...]:
    return tuple(_canonical(value) for value in values)


__all__ = [
    "MAX_SPLITTING_FIELD_DEGREE",
    "compute_bound_profile",
    "compute_splitting_field",
    "conjugate_element",
    "multiply",
    "rectangle_for_root",
    "reduce_modulus",
    "subtract",
]
