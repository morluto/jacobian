"""Exact monic quartic cubic-resolvent contract tests."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.polynomials._quartic_resolvent import (
    QuarticCubicResolventRequest,
    QuarticCubicResolventResult,
    compute_quartic_cubic_resolvent,
)
from jacobian.math.polynomials._quartic_resolvent_tools import (
    QUARTIC_CUBIC_RESOLVENT_OPERATION,
)
from jacobian.math.polynomials.values import (
    MonicPolynomial,
    monic_polynomial_from_coefficients,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _monic(coefficients: tuple[Fraction, ...]) -> MonicPolynomial:
    return monic_polynomial_from_coefficients(
        tuple(CanonicalRational.from_fraction(value) for value in coefficients),
        variable="t",
    )


def _mul(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return product


def _expand_roots(roots: tuple[Fraction, ...]) -> list[Fraction]:
    monic = [Fraction(1)]
    for root in roots:
        monic = _mul(monic, [-root, Fraction(1)])
    return monic


def _evaluate(poly: list[Fraction], value: Fraction) -> Fraction:
    total = Fraction(0)
    for coefficient in reversed(poly):
        total = total * value + coefficient
    return total


def _resultant(f: list[Fraction], g: list[Fraction]) -> Fraction:
    # Sylvester matrix determinant over QQ by exact Gaussian elimination.
    m = len(f) - 1
    n = len(g) - 1
    size = m + n
    rows: list[list[Fraction]] = []
    for shift in range(n):
        row = [Fraction(0)] * size
        row[shift : shift + m + 1] = list(f)
        rows.append(row)
    for shift in range(m):
        row = [Fraction(0)] * size
        row[shift : shift + n + 1] = list(g)
        rows.append(row)
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((r for r in range(column, size) if rows[r][column] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        determinant *= rows[column][column]
        inverse = Fraction(1) / rows[column][column]
        for other in range(column + 1, size):
            factor = rows[other][column] * inverse
            if factor != 0:
                rows[other] = [
                    a - factor * b
                    for a, b in zip(rows[other], rows[column], strict=True)
                ]
    return determinant


def _discriminant(poly: list[Fraction]) -> Fraction:
    degree = len(poly) - 1
    derivative = [i * poly[i] for i in range(1, len(poly))]
    sign = -1 if degree * (degree - 1) // 2 % 2 else 1
    return sign * _resultant(poly, derivative) / poly[-1]


def test_roots_one_through_four_give_pair_product_resolvent() -> None:
    # Independent oracle: form the three root-pair sums from the known roots,
    # then multiply (Y-u)(Y-v)(Y-w), without using the coefficient formula.
    roots = (1, 2, 3, 4)
    pair_sums = (
        roots[0] * roots[1] + roots[2] * roots[3],
        roots[0] * roots[2] + roots[1] * roots[3],
        roots[0] * roots[3] + roots[1] * roots[2],
    )
    assert pair_sums == (14, 11, 10)
    expected = (
        -pair_sums[0] * pair_sums[1] * pair_sums[2],
        pair_sums[0] * pair_sums[1]
        + pair_sums[0] * pair_sums[2]
        + pair_sums[1] * pair_sums[2],
        -sum(pair_sums),
        1,
    )
    source = monic_polynomial_from_coefficients(
        tuple(map(_q, (24, -50, 35, -10, 1))), variable="t"
    )
    result = compute_quartic_cubic_resolvent(source)
    assert result.source == source
    assert result.convention == "MONIC_QUARTIC_ROOT_PAIR_PRODUCTS"
    assert tuple(c.as_fraction() for c in result.resolvent.coefficients) == tuple(
        Fraction(value) for value in expected
    )
    assert result.resolvent.variables == ("y",)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_operation_is_catalogued_with_executable_independent_example() -> None:
    assert QUARTIC_CUBIC_RESOLVENT_OPERATION.operation_id == (
        "polynomial.quartic.cubic_resolvent.compute"
    )
    assert QUARTIC_CUBIC_RESOLVENT_OPERATION in BUILTIN_TOOLS
    result = QUARTIC_CUBIC_RESOLVENT_OPERATION.run(
        QuarticCubicResolventRequest.model_validate_json(
            json.dumps(QUARTIC_CUBIC_RESOLVENT_OPERATION.examples[0].input)
        )
    )
    assert tuple(c.as_fraction() for c in result.resolvent.coefficients) == tuple(
        map(Fraction, (-1540, 404, -35, 1))
    )


def test_request_requires_degree_four() -> None:
    cubic = monic_polynomial_from_coefficients(tuple(map(_q, (1, 0, 0, 1))))
    with pytest.raises(ValidationError, match="degree-four"):
        QuarticCubicResolventRequest(polynomial=cubic)


def test_result_retains_revalidated_canonical_source() -> None:
    canonical = _monic(tuple(map(Fraction, (24, -50, 35, -10, 1))))
    noncanonical = canonical.model_copy(update={"variables": ["t"]})
    assert isinstance(noncanonical.variables, list)

    result = compute_quartic_cubic_resolvent(noncanonical)

    assert isinstance(result.source.variables, tuple)
    assert result.source == canonical
    assert (
        QuarticCubicResolventResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_resolvent_roots_are_the_three_pair_product_sums() -> None:
    roots = (Fraction(1, 2), Fraction(2), Fraction(-3), Fraction(4))
    pair_sums = (
        roots[0] * roots[1] + roots[2] * roots[3],
        roots[0] * roots[2] + roots[1] * roots[3],
        roots[0] * roots[3] + roots[1] * roots[2],
    )
    assert pair_sums == (Fraction(-11), Fraction(13, 2), Fraction(-4))
    source = _monic(tuple(_expand_roots(roots)))

    resolvent = compute_quartic_cubic_resolvent(source).resolvent
    coefficients = tuple(c.as_fraction() for c in resolvent.coefficients)
    assert coefficients == tuple(_expand_roots(pair_sums))
    for root in pair_sums:
        assert _evaluate(list(coefficients), root) == 0


@pytest.mark.parametrize(
    "coefficients",
    [
        (Fraction(24), Fraction(-50), Fraction(35), Fraction(-10), Fraction(1)),
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(-10), Fraction(0), Fraction(1)),
        (Fraction(2), Fraction(-3), Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(-7, 3), Fraction(1, 2), Fraction(-5), Fraction(3), Fraction(1)),
    ],
)
def test_resolvent_discriminant_equals_quartic_discriminant(
    coefficients: tuple[Fraction, ...],
) -> None:
    result = compute_quartic_cubic_resolvent(_monic(coefficients))
    quartic_discriminant = _discriminant(list(coefficients))
    resolvent_coefficients = [
        coefficient.as_fraction() for coefficient in result.resolvent.coefficients
    ]
    assert _discriminant(resolvent_coefficients) == quartic_discriminant


def test_coefficient_beyond_digit_admission_is_resource_refusal() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.polynomials._quartic_resolvent import (
        MAX_QUARTIC_RESOLVENT_INPUT_DIGITS,
    )

    value = 10**MAX_QUARTIC_RESOLVENT_INPUT_DIGITS
    with pytest.raises(OperationResourceAdmissionError):
        compute_quartic_cubic_resolvent(
            _monic(
                (Fraction(value), Fraction(0), Fraction(0), Fraction(0), Fraction(1))
            )
        )


def test_maximally_admitted_coefficient_digits_fit_preflight() -> None:
    from jacobian.math.polynomials._quartic_resolvent import (
        MAX_QUARTIC_RESOLVENT_INPUT_DIGITS,
        MAX_QUARTIC_RESOLVENT_OUTPUT_DIGITS,
        MAX_QUARTIC_RESOLVENT_WORK,
    )

    digits = MAX_QUARTIC_RESOLVENT_INPUT_DIGITS
    assert digits == 306
    assert 32 * digits * digits <= MAX_QUARTIC_RESOLVENT_WORK
    assert 8 * digits + 32 <= MAX_QUARTIC_RESOLVENT_OUTPUT_DIGITS
    # 257-digit components, previously rejected despite fitting both bounds.
    value = 10**256
    compute_quartic_cubic_resolvent(
        _monic((Fraction(value), Fraction(0), Fraction(0), Fraction(0), Fraction(1)))
    )
