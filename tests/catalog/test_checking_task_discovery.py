"""Checking queries must retain the mathematical object and coefficient domain."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "check whether a multivariate polynomial factors over the rationals",
            "polynomial.multivariate.factor.compute",
        ),
        (
            "verify multivariate polynomial factorization over rationals",
            "polynomial.multivariate.factor.compute",
        ),
        (
            "verify finite field irreducibility of a degree 108 polynomial using Frobenius",
            "polynomial.galois.factor_mod_p.compute",
        ),
        (
            "check finite field polynomial irreducibility from coefficients",
            "polynomial.galois.factor_mod_p.compute",
        ),
        (
            "verify exact rational linear programming primal dual optimality",
            "optimization.linear.rational_optimality.check",
        ),
        (
            "check supplied primal dual candidates for a rational linear program without solving",
            "optimization.linear.rational_optimality.check",
        ),
        (
            "please check supplied primal dual candidates for a rational linear program over the rationals",
            "optimization.linear.rational_optimality.check",
        ),
        ("compute the dual of a linear code", "code.linear.dual.compute"),
        (
            "compute Frobenius cycle type from supplied factor degrees",
            "polynomial.galois.frobenius_cycle.compute",
        ),
        (
            "compute exact Bernstein coefficients of a polynomial",
            "polynomial.bernstein.coefficients.compute",
        ),
        (
            "check supplied edge clique partition certificate",
            "graph.edge_clique_partition.check",
        ),
    ],
)
def test_ranked_contract_matches_requested_input(query: str, expected: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=3)).matches
    assert matches[0].operation_id == expected


@pytest.mark.parametrize(
    "query",
    [
        "check polynomial identity over integers",
        "verify exact polynomial identity over the integers",
    ],
)
def test_missing_integer_identity_checker_does_not_claim_neighbor_support(
    query: str,
) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=5)).matches
    assert not matches


def test_discovered_factorization_consumes_the_actual_degree_108_polynomial() -> None:
    from jacobian.math.number_theory.galois._tools import TOOLS

    match = (
        Catalog.open()
        .match(
            OperationMatchRequest(
                need="verify finite field irreducibility of a degree 108 polynomial using Frobenius",
                limit=1,
            )
        )
        .matches[0]
    )
    operation = next(tool for tool in TOOLS if tool.operation_id == match.operation_id)
    coefficients = [1] + [0] * 107 + [1]
    request = operation.request_type.model_validate_json(
        json.dumps({"field_order": 3, "coefficients": coefficients})
    )
    result = operation.run(request)
    assert result.source_coefficients == tuple(coefficients)
    assert not result.is_irreducible
    assert any(factor.multiplicity > 1 for factor in result.factors)


def test_discovered_lp_checker_consumes_supplied_candidates_and_rejects_a_perturbation() -> (
    None
):
    from jacobian.math.optimization._tools import TOOLS

    match = (
        Catalog.open()
        .match(
            OperationMatchRequest(
                need="verify exact rational linear programming primal dual optimality",
                limit=1,
            )
        )
        .matches[0]
    )
    operation = next(tool for tool in TOOLS if tool.operation_id == match.operation_id)
    payload = dict(operation.examples[0].input)
    request = operation.request_type.model_validate_json(json.dumps(payload))
    assert operation.run(request).is_optimal
    payload["primal_candidate"] = [{"num": "0", "den": "1"}]
    invalid = operation.request_type.model_validate_json(json.dumps(payload))
    result = operation.run(invalid)
    assert not result.is_optimal
    assert not result.primal_feasible
