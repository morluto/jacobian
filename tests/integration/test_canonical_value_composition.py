"""Public-operation composition through canonical JSON value boundaries."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.polytopes._models import PolytopeVolumeRequest
from jacobian.math.geometry.polytopes._operations import (
    compute_polytope_support,
    compute_polytope_volume,
)
from jacobian.math.geometry.polytopes._tools import TOOLS
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    TreeAutomatonReachabilityRequest,
)
from jacobian.math.logic.automata.tree._operations import (
    compute_accepted_tree_count,
    compute_tree_automaton_reachability,
)
from jacobian.math.matrices._operation_models import (
    MatrixDeterminantRequest,
    RationalMatrixProductRequest,
)
from jacobian.math.matrices._operations import compute_determinant, compute_product
from jacobian.math.polynomials.series._models import (
    SeriesFromPolynomialRequest,
    SeriesTruncateRequest,
)
from jacobian.math.polynomials.series._operations import (
    compute_from_polynomial,
    compute_truncate,
)


def _canonical_json(value: Any) -> Any:
    """Cross the public value boundary as canonical JSON bytes, then decode it."""

    return json.loads(encode_strict_json(value.model_dump(mode="json")))


def test_matrix_product_value_composes_into_determinant_request() -> None:
    product = compute_product(
        RationalMatrixProductRequest.model_validate(
            {
                "left": {
                    "entries": [
                        [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                        [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                    ]
                },
                "right": {
                    "entries": [
                        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ]
                },
            }
        )
    )

    determinant_request = MatrixDeterminantRequest.model_validate(
        {"matrix": _canonical_json(product.product)}
    )

    assert compute_determinant(determinant_request).determinant.as_fraction() == -2


def test_support_polytope_value_composes_into_volume_request() -> None:
    support_tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polytope.rational.support.compute"
    )
    support = compute_polytope_support(
        support_tool.request_type.model_validate(support_tool.examples[0].input)
    )

    volume_request = PolytopeVolumeRequest.model_validate(
        {"vertices": _canonical_json(support.polytope)}
    )

    assert compute_polytope_volume(volume_request).volume.as_fraction() == 1


def test_formal_series_value_composes_into_truncation_request() -> None:
    producer_request = SeriesFromPolynomialRequest.model_validate(
        {
            "variable": "x",
            "coefficients": [
                {"num": "1", "den": "1"},
                {"num": "2", "den": "1"},
                {"num": "3", "den": "1"},
            ],
            "truncation_order": 3,
        }
    )
    produced = compute_from_polynomial(
        producer_request.variable,
        producer_request.coefficients,
        producer_request.truncation_order,
    )

    consumer_request = SeriesTruncateRequest.model_validate(
        {"series": _canonical_json(produced.result), "target_order": 2}
    )
    truncated = compute_truncate(consumer_request.series, consumer_request.target_order)

    assert tuple(value.as_fraction() for value in truncated.result.coefficients) == (
        Fraction(1),
        Fraction(2),
    )


def test_tree_automaton_value_composes_into_count_request() -> None:
    profile = compute_tree_automaton_reachability(
        TreeAutomatonReachabilityRequest.model_validate(
            {
                "automaton": {
                    "state_count": 1,
                    "arity": [0],
                    "transitions": [
                        {"symbol": 0, "child_states": [], "target_state": 0}
                    ],
                    "final_states": [0],
                }
            }
        )
    )

    count = compute_accepted_tree_count(
        AcceptedTreeCountRequest.model_validate(
            {"automaton": _canonical_json(profile.automaton), "tree_size": 1}
        )
    )

    assert int(count.count) == 1
