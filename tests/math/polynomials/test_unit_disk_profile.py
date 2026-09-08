"""Exact independent root-location and admission regressions."""

from collections.abc import Sequence
from fractions import Fraction
from random import Random
from time import monotonic

import pytest

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.unit_circle import UnitDiskProfile, unit_disk_profile
from jacobian.math.polynomials.values import RationalPolynomial


def polynomial(
    coefficients: Sequence[int | Fraction], *, stride: int = 1, shift: int = 0
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["z"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {
                            "num": Fraction(c).numerator,
                            "den": Fraction(c).denominator,
                        },
                        "exponents": [i * stride + shift],
                    }
                    for i, c in reversed(list(enumerate(coefficients)))
                    if c
                ]
            },
        }
    )


def counts(source: RationalPolynomial) -> tuple[int, int, int]:
    result = unit_disk_profile(source)
    assert result.inside + result.on + result.outside == result.degree
    assert UnitDiskProfile.model_validate_json(result.model_dump_json()) == result
    assert result.polynomial == source
    return result.inside, result.on, result.outside


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([2, -5, 2], (1, 0, 1)),
        ([1, 1, -1, 1, 1], (1, 2, 1)),
        ([7], (0, 0, 0)),
        ([0, 0, 0, 1], (3, 0, 0)),
        ([1, 0, 1], (0, 2, 0)),
        ([1, 2, 1], (0, 2, 0)),
        ([1, -2, 1], (0, 2, 0)),
        ([1, 0, 2], (2, 0, 0)),
        ([2, 0, 1], (0, 0, 2)),
    ],
)
def test_exact_basic_and_comment_fixtures(
    coefficients: list[int], expected: tuple[int, int, int]
) -> None:
    assert counts(polynomial(coefficients)) == expected


def test_repeated_rational_and_complex_factors_with_known_roots() -> None:
    from flint import fmpz_poly

    z = fmpz_poly([0, 1])
    p = (2 * z - 1) ** 3 * (z - 1) ** 4 * (z + 1) ** 2 * (z - 2) ** 5 * (z * z + 1) ** 3
    p *= (2 * z * z + 1) ** 2 * (z * z + 2) ** 3
    assert counts(polynomial([int(c) for c in p.coeffs()])) == (7, 12, 11)


def test_rational_scalar_reciprocal_and_rational_radius_composition() -> None:
    a = [Fraction(1, 7), Fraction(-2, 3), Fraction(1, 5), Fraction(3, 2)]
    inside, on, outside = counts(polynomial(a))
    assert counts(polynomial(list(reversed(a)))) == (outside, on, inside)
    assert counts(polynomial([-Fraction(11, 13) * c for c in a])) == (
        inside,
        on,
        outside,
    )
    q = [2, -5, 2]
    small = unit_disk_profile(
        polynomial([c * Fraction(1, 2) ** i for i, c in enumerate(q)])
    )
    large = unit_disk_profile(
        polynomial([c * Fraction(2) ** i for i, c in enumerate(q)])
    )
    assert large.inside + large.on - small.inside == 2
    assert small.on == large.on == 1


def test_roots_arbitrarily_close_to_boundary_without_tolerance() -> None:
    epsilon = Fraction(1, 10**200)
    assert counts(polynomial([-(1 - epsilon), 1])) == (1, 0, 0)
    assert counts(polynomial([-(1 + epsilon), 1])) == (0, 0, 1)


def test_sparse_large_exponents_use_compressed_support() -> None:
    assert counts(polynomial([-1, 1], stride=32768)) == (0, 32768, 0)
    assert counts(polynomial([1], shift=32768)) == (32768, 0, 0)
    assert counts(polynomial([2, -5, 2], stride=10000, shift=17)) == (10017, 0, 10000)


def test_dense_degree_32_is_accepted() -> None:
    assert counts(polynomial([i * i % 7 + 1 for i in range(33)])) == (18, 0, 14)


def test_dense_degree_40_preserves_weighted_derivative_growth() -> None:
    # Independently checked with exact rectangle winding counts.
    assert counts(polynomial([i * i % 7 + 1 for i in range(41)])) == (20, 0, 20)


@pytest.mark.parametrize("shift", [0, 32768])
def test_single_term_large_scalar_needs_no_coefficient_arithmetic(shift: int) -> None:
    source = polynomial([Fraction(10**20000, 3)], shift=shift)
    assert counts(source) == (shift, 0, 0)


def test_independent_exact_rectangle_winding_oracle() -> None:
    from sympy import Poly, Symbol
    from sympy.polys.domains import QQ
    from sympy.polys.rootisolation import dup_count_complex_roots

    random = Random(3152)
    z = Symbol("z")
    for _ in range(12):
        coefficients = [random.randint(-5, 5) for _ in range(5)] + [1]
        # Independent symbolic substitution plus Collins--Krandick winding
        # numbers, rather than either of the production matrix signatures.
        p = Poly.from_list(list(reversed(coefficients)), z)
        q = Poly(
            sum(
                c * (1 + z) ** i * (1 - z) ** (5 - i)
                for i, c in enumerate(coefficients)
            ),
            z,
        )
        bound = QQ.convert(2 + 2 * max(abs(c / q.LC()) for c in q.all_coeffs()))
        dense = [QQ.convert(c) for c in q.all_coeffs()]
        left = dup_count_complex_roots(
            dense, QQ, (-bound, -bound), (0, bound), exclude=["E"]
        )
        right = dup_count_complex_roots(
            dense, QQ, (0, -bound), (bound, bound), exclude=["W"]
        )
        assert counts(polynomial(coefficients)) == (
            left,
            int(p.degree()) - left - right,
            right,
        )


def test_domain_and_resource_rejections_are_distinct() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonzero univariate"):
        unit_disk_profile(polynomial([]))
    multivariate = RationalPolynomial.model_validate(
        {
            "variables": ["x", "y"],
            "polynomial": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [1, 1]}]
            },
        }
    )
    with pytest.raises(OperationDomainValidationError, match="nonzero univariate"):
        unit_disk_profile(multivariate)
    with pytest.raises(OperationResourceAdmissionError, match="derived"):
        unit_disk_profile(polynomial([1] * 100))


def test_earlier_caller_deadline_is_preserved() -> None:
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            unit_disk_profile(polynomial([1, 1]))
