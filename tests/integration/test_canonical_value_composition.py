"""Public-operation composition through canonical JSON value boundaries."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from jacobian.canonical import encode_strict_json
from jacobian.math.geometry.polytopes._models import PolytopeVolumeRequest
from jacobian.math.geometry.polytopes._tools import (
    TOOLS,
    compute_polytope_support,
    compute_polytope_volume,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    TreeAutomatonReachabilityRequest,
)
from jacobian.math.logic.automata.tree._tools import (
    compute_accepted_tree_count,
    compute_tree_automaton_reachability,
)
from jacobian.math.matrices._operation_models import (
    MatrixDeterminantRequest,
    RationalMatrixProductRequest,
)
from jacobian.math.matrices._tools import compute_determinant, compute_product
from jacobian.math.polynomials.series._models import (
    SeriesFromPolynomialRequest,
    SeriesTruncateRequest,
)
from jacobian.math.polynomials.series.operations import (
    from_polynomial,
    truncate,
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
            "polynomial": {
                "variables": ["x"],
                "polynomial": {
                    "terms": [
                        {"coefficient": {"num": "3", "den": "1"}, "exponents": [2]},
                        {"coefficient": {"num": "2", "den": "1"}, "exponents": [1]},
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
                    ]
                },
            },
            "truncation_order": 3,
        }
    )
    produced = from_polynomial(
        producer_request.polynomial,
        producer_request.truncation_order,
    )

    consumer_request = SeriesTruncateRequest.model_validate(
        {"series": _canonical_json(produced.result), "target_order": 2}
    )
    truncated = truncate(consumer_request.series, consumer_request.target_order)

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


def test_matrix_polynomial_producers_compose_without_coefficient_rewriting() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    matrix = {
        "entries": [
            [{"num": str(v), "den": "1"} for v in row] for row in [[1, 2], [3, 4]]
        ]
    }
    characteristic = invoke_operation(
        "matrix.characteristic_polynomial.compute", {"matrix": matrix}, catalog
    ).output
    minimal = invoke_operation(
        "matrix.minimal_polynomial.compute", {"matrix": matrix}, catalog
    ).output
    assert characteristic["polynomial"] == minimal["characteristic_polynomial"]
    for polynomial in [characteristic["polynomial"], minimal["minimal_polynomial"]]:
        output = invoke_operation(
            "matrix.polynomial.evaluate.compute",
            {"matrix": matrix, "polynomial": json.loads(json.dumps(polynomial))},
            catalog,
        ).output
        assert all(
            value["num"] == "0" for row in output["value"]["entries"] for value in row
        )


def test_prime_field_matrix_coordinate_maps_retain_presentation_and_axes() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.finite_fields import Axis, AxisBoundMatrix, finite_field
    from jacobian.math.matrices.finite_fields import PrimeFieldMatrix
    from jacobian.math.matrices.finite_fields.presentations import (
        bind_prime_matrix,
        prime_matrix_coordinates,
    )

    matrix = PrimeFieldMatrix(prime=2, entries=((1, 1), (1, 1)), columns=2)
    presented = bind_prime_matrix(
        matrix,
        finite_field(2, (0, 1)),
        Axis(name="equations", labels=("second", "first")),
        Axis(name="unknowns", labels=("v", "u")),
    )
    wire = presented.model_dump(mode="json")
    assert (
        invoke_operation(
            "finite_field.matrix.rank.compute", {"matrix": wire}, Catalog.open()
        ).output["rank"]
        == 1
    )
    restored = prime_matrix_coordinates(AxisBoundMatrix.model_validate(wire))
    assert restored == matrix
    assert (
        bind_prime_matrix(
            restored, presented.presentation, presented.row_axis, presented.column_axis
        )
        == presented
    )


def test_smith_producers_share_the_same_normal_form_value() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    entries = [["2", "4"], ["0", "6"]]
    plain = invoke_operation(
        "matrix.normal_form.smith.compute", {"matrix": {"entries": entries}}, catalog
    ).output
    certified = invoke_operation(
        "matrix.normal_form.smith.certified.compute",
        {
            "matrix": {
                "domain": "ZZ",
                "entries": entries,
                "row_count": 2,
                "column_count": 2,
            }
        },
        catalog,
    ).output
    assert plain == certified["smith_form"]


def test_rational_factor_sturm_and_isolation_share_the_same_polynomial() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    polynomial = {
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "2", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
    factored = invoke_operation(
        "polynomial.factor.compute", {"polynomial": polynomial}, catalog
    ).output
    factor = factored["factors"][0]["factor"]
    assert factor["variables"] == ["z"]
    chain = invoke_operation(
        "polynomial.sturm_chain.compute", {"polynomial": factor}, catalog
    ).output["chain"]
    assert chain[0] == factor
    for value in [factor, chain[0]]:
        isolated = invoke_operation(
            "polynomial.roots.isolate", {"polynomial": value}, catalog
        ).output
        assert isolated["source_polynomial"] == value
        assert len(isolated["roots"]) == 1
        interval = isolated["roots"][0]["isolating_interval"]
        endpoints = [
            Fraction(int(endpoint["num"]), int(endpoint["den"]))
            for endpoint in interval
        ]
        assert endpoints[0] <= Fraction(-1, 2) <= endpoints[1]
        assert isolated["roots"][0]["algebraic_value"]["polynomial"] == ["2", "1"]
