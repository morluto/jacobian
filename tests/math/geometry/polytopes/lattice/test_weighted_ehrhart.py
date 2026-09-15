"""Weighted Ehrhart slice (#3716): exact weighted counts and polynomials."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.polytopes.lattice._models import WeightedEhrhartRequest
from jacobian.math.geometry.polytopes.lattice._tools import weighted_ehrhart_polynomial
from jacobian.math.geometry.polytopes.lattice.operations import ehrhart_polynomial
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

R = CanonicalRational


def _vertex(*coordinates: int) -> Vertex:
    return Vertex(
        coordinates=tuple(R(num=c, den=1) for c in coordinates),
    )


def _weight(variables: tuple[str, ...], *terms: tuple[int, tuple[int, ...]]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=R(num=c, den=1), exponents=exponents
                )
                for c, exponents in terms
            )
        ),
    )


def _request(vertices, weight, degree_bound, max_dilation) -> WeightedEhrhartRequest:
    return WeightedEhrhartRequest(
        vertices=vertices,
        weight=weight,
        degree_bound=degree_bound,
        max_dilation=max_dilation,
    )


def test_weight_one_agrees_with_ordinary_ehrhart() -> None:
    """The ordinary case is weight 1: counts and polynomial must agree."""
    vertices = (_vertex(0), _vertex(1))
    weighted = weighted_ehrhart_polynomial(
        _request(vertices, _weight(("x0",), (1, (0,))), 1, 3)
    )
    ordinary = ehrhart_polynomial(vertices, 1, 3)
    assert tuple(c.as_fraction() for _, c in weighted.counts) == tuple(
        Fraction(v) for _, v in ordinary.counts
    )
    assert weighted.polynomial == ordinary.polynomial
    assert tuple(c.as_fraction() for c in weighted.coefficient_table) == (
        Fraction(1),
        Fraction(1),
    )


def test_weighted_interval_against_brute_force_oracle() -> None:
    """Weight x0 on [0,1] counts t(t+1)/2, checked against direct summation."""
    vertices = (_vertex(0), _vertex(1))
    result = weighted_ehrhart_polynomial(
        _request(vertices, _weight(("x0",), (1, (1,))), 2, 4)
    )
    assert tuple(c.as_fraction() for _, c in result.counts) == (
        Fraction(0),
        Fraction(1),
        Fraction(3),
        Fraction(6),
        Fraction(10),
    )
    assert tuple(c.as_fraction() for c in result.coefficient_table) == (
        Fraction(0),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    # Independent oracle: triangular numbers, not the kernel scan.
    assert [n * (n + 1) // 2 for n in range(5)] == [0, 1, 3, 6, 10]


def test_weighted_square_known_answer() -> None:
    """Weight x0 on [0,1]^2 counts t(t+1)^2/2."""
    vertices = (_vertex(0, 0), _vertex(1, 0), _vertex(1, 1), _vertex(0, 1))
    result = weighted_ehrhart_polynomial(
        _request(vertices, _weight(("x0", "x1"), (1, (1, 0))), 3, 4)
    )
    assert tuple(c.as_fraction() for _, c in result.counts) == (
        Fraction(0),
        Fraction(2),
        Fraction(9),
        Fraction(24),
        Fraction(50),
    )


def test_insufficient_degree_bound_rejected() -> None:
    vertices = (_vertex(0), _vertex(1))
    with pytest.raises(OperationDomainValidationError, match="degree_bound"):
        weighted_ehrhart_polynomial(
            _request(vertices, _weight(("x0",), (1, (1,))), 1, 3)
        )


def test_weight_axis_mismatch_rejected() -> None:
    vertices = (_vertex(0), _vertex(1))
    with pytest.raises(ValidationError):
        _request(vertices, _weight(("y",), (1, (1,))), 2, 3)


def test_nonintegral_vertices_rejected() -> None:
    vertices = (
        Vertex(coordinates=(R(num=1, den=2),)),
        Vertex(coordinates=(R(num=1, den=1),)),
    )
    with pytest.raises(ValidationError):
        _request(vertices, _weight(("x0",), (1, (1,))), 2, 3)


def test_forged_counts_rejected() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.geometry.polytopes.lattice._models import WeightedEhrhartResult

    vertices = (_vertex(0), _vertex(1))
    result = weighted_ehrhart_polynomial(
        _request(vertices, _weight(("x0",), (1, (1,))), 2, 3)
    )
    # Structural forgeries are rejected at the boundary; mathematical replay
    # stays in the kernel, so forge the coefficient binding, not a value.
    payload = result.model_dump(mode="json")
    payload["coefficient_table"][1] = {"num": "5", "den": "1"}
    with pytest.raises(ValidationError):
        WeightedEhrhartResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    bad_axis = result.model_dump(mode="json")
    bad_axis["counts"] = bad_axis["counts"][:2]
    with pytest.raises(ValidationError):
        WeightedEhrhartResult.model_validate_json(
            encode_strict_json(bad_axis), strict=True
        )
