"""Exact finite-degree invariant slices for diagonal G_m actions."""

from fractions import Fraction
from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightAction,
    PolynomialWeightActionRequest,
    PolynomialWeightInvariantRequest,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    diagonal_weight_action,
    gm_invariants_through_degree,
)


def _monomial_exponents(poly) -> tuple[tuple[int, ...], ...]:
    assert all(
        term.coefficient.as_fraction() == Fraction(1) for term in poly.polynomial.terms
    )
    return tuple(tuple(term.exponents) for term in poly.polynomial.terms)


def test_complete_invariant_basis_against_independent_cartesian_oracle() -> None:
    action = PolynomialWeightAction(variables=("x", "y", "z"), weights=(2, -3, 1))
    degree = 5
    result = gm_invariants_through_degree(
        PolynomialWeightInvariantRequest(action=action, degree=degree)
    )

    # Independent oracle enumerates the bounded exponent cube and filters by
    # total degree and the integer character pairing, not by the operation's
    # homogeneous-composition recursion.
    expected_by_degree = {
        d: tuple(
            exponent
            for exponent in product(range(degree + 1), repeat=3)
            if sum(exponent) == d
            and sum(
                weight * power
                for weight, power in zip(action.weights, exponent, strict=True)
            )
            == 0
        )
        for d in range(degree + 1)
    }
    expected = tuple(
        exponent
        for d in range(degree + 1)
        for exponent in sorted(expected_by_degree[d], reverse=True)
    )
    assert tuple(_monomial_exponents(poly)[0] for poly in result.basis) == expected
    assert tuple(row.dimension for row in result.hilbert_prefix) == tuple(
        len(expected_by_degree[d]) for d in range(degree + 1)
    )
    assert result.dimension == len(expected)


def test_invariants_compose_through_existing_coaction() -> None:
    action = PolynomialWeightAction(variables=("x", "y"), weights=(1, -1))
    result = gm_invariants_through_degree({"action": action.model_dump(), "degree": 4})
    assert tuple(row.dimension for row in result.hilbert_prefix) == (1, 0, 1, 0, 1)
    assert tuple(_monomial_exponents(poly)[0] for poly in result.basis) == (
        (0, 0),
        (1, 1),
        (2, 2),
    )

    for invariant in result.basis:
        coaction = diagonal_weight_action(
            PolynomialWeightActionRequest(action=action, polynomial=invariant)
        )
        assert len(coaction.components) == 1
        assert coaction.components[0].weight == 0
        assert coaction.weight_zero == invariant
        assert all(term.exponents[-1] == 0 for term in coaction.coaction.terms)


def test_combinatorial_work_is_admitted_before_enumeration() -> None:
    action = PolynomialWeightAction(
        variables=("a", "b", "c", "d", "e", "f", "g", "h"),
        weights=(0, 0, 0, 0, 0, 0, 0, 0),
    )
    # C(8+7, 7)=6435 candidates exceeds the 4096 envelope. In particular,
    # even an all-zero action is rejected before constructing its output basis.
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        gm_invariants_through_degree(
            PolynomialWeightInvariantRequest(action=action, degree=7)
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_weight_invariant.monomial_budget"
    )


def test_operation_is_catalogued_with_usable_example() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import OperationMatchRequest
    from jacobian.math.polynomials.derivations._tools import TOOLS

    operation_id = "algebraic_group.gm.invariants_through_degree.compute"
    tool = next(item for item in TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.dimension == sum(row.dimension for row in result.hilbert_prefix)
    found = Catalog.open().match(
        OperationMatchRequest(need="weight zero polynomial basis by degree")
    )
    assert any(item.operation_id == operation_id for item in found.matches)
