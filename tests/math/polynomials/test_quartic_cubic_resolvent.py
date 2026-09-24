"""Exact monic quartic cubic-resolvent contract tests."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.polynomials._quartic_resolvent import (
    QuarticCubicResolventRequest,
    compute_quartic_cubic_resolvent,
)
from jacobian.math.polynomials._quartic_resolvent_tools import (
    QUARTIC_CUBIC_RESOLVENT_OPERATION,
)
from jacobian.math.polynomials.values import monic_polynomial_from_coefficients


def _q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


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
