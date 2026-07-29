from __future__ import annotations

from typing import Any

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.topology import TOPOLOGY_BUNDLE

_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}
_TORUS_FACETS = [
    [str(index), str((index + 1) % 7), str((index + 3) % 7)] for index in range(7)
] + [[str(index), str((index + 2) % 7), str((index + 3) % 7)] for index in range(7)]
_PROJECTIVE_PLANE_FACETS = [
    ["0", "1", "2"],
    ["0", "1", "3"],
    ["0", "2", "4"],
    ["0", "3", "5"],
    ["0", "4", "5"],
    ["1", "2", "5"],
    ["1", "3", "4"],
    ["1", "4", "5"],
    ["2", "3", "4"],
    ["2", "3", "5"],
]


def _materialize(runtime, presentation: dict[str, Any]) -> dict[str, Any]:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.materialize",
            input=presentation,
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return result.output["result"]["complex"]


def _betti(
    runtime,
    presentation: dict[str, Any],
    *,
    prime: int = 2,
    convention: str = "UNREDUCED",
) -> tuple[int, ...]:
    complex_ = _materialize(runtime, presentation)
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.compute",
            input={
                "complex": complex_,
                "prime": prime,
                "convention": convention,
            },
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    return tuple(group["betti_number"] for group in result.output["result"]["groups"])


def test_topology_bundle_exposes_three_atomic_capabilities(runtime) -> None:
    ids = tuple(operation.capability_id for operation in TOPOLOGY_BUNDLE.capabilities)

    assert ids == (
        "topology.simplicial_complex.materialize",
        "topology.simplicial_complex.chain_complex.compute",
        "topology.simplicial_homology.compute",
    )
    assert "topology" in runtime.portfolio.domain_bundles
    catalog_ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    assert set(ids).issubset(catalog_ids)


def test_homology_intent_discovers_the_domain_owned_operation(runtime) -> None:
    discovered = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="homology of a finite simplicial complex over F_2",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == (
        "topology.simplicial_homology.compute"
    )
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"


def test_materialization_is_canonical_complete_and_artifact_backed(runtime) -> None:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.materialize",
            input={
                "vertices": ["c", "a", "b"],
                "facets": [["b", "a"], ["c", "b"], ["c", "a"]],
            },
        )
    )

    complex_ = result.output["result"]["complex"]
    assert complex_["vertices"] == ["a", "b", "c"]
    assert complex_["maximal_simplices"] == [
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
    ]
    assert complex_["f_vector"] == [3, 3]
    assert complex_["closure_size"] == 6
    assert complex_["empty_simplex_stored"] is False
    assert result.output["result"]["completeness"] == "COMPLETE_FACE_CLOSURE"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert len(result.artifact_uris) == 2
    assert (
        runtime.core.store.get(result.output["result_uri"]).payload
        == (result.output["result"])
    )


def test_chain_complex_exposes_oriented_sparse_boundaries_and_augmentation(
    runtime,
) -> None:
    triangle = _materialize(
        runtime,
        {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]},
    )
    integer = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input={
                "complex": triangle,
                "coefficient_ring": "INTEGER",
                "convention": "UNREDUCED",
            },
        )
    ).output["result"]
    mod_two_reduced = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input={
                "complex": triangle,
                "coefficient_ring": "PRIME_FIELD",
                "prime": 2,
                "convention": "REDUCED",
            },
        )
    ).output["result"]

    assert integer["boundary_matrices"][2]["entries"] == [
        {"row": 0, "column": 0, "value": 1},
        {"row": 1, "column": 0, "value": -1},
        {"row": 2, "column": 0, "value": 1},
    ]
    assert integer["augmentation"] is None
    assert all(item["product_is_zero"] for item in integer["boundary_squared_zero"])
    assert mod_two_reduced["augmentation"]["entries"] == [
        {"row": 0, "column": 0, "value": 1},
        {"row": 0, "column": 1, "value": 1},
        {"row": 0, "column": 2, "value": 1},
    ]
    assert {
        entry["value"]
        for matrix in mod_two_reduced["boundary_matrices"]
        for entry in matrix["entries"]
    } == {1}


@pytest.mark.parametrize(
    ("presentation", "f_vector", "betti"),
    (
        ({"vertices": ["a"], "facets": [["a"]]}, (1,), (1,)),
        (
            {
                "vertices": ["a", "b", "c"],
                "facets": [["a"], ["b"], ["c"]],
            },
            (3,),
            (3,),
        ),
        ({"vertices": ["a", "b"], "facets": [["a", "b"]]}, (2, 1), (1, 0)),
        (_CIRCLE, (3, 3), (1, 1)),
        (
            {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]},
            (3, 3, 1),
            (1, 0, 0),
        ),
        (
            {
                "vertices": ["0", "1", "2", "3"],
                "facets": [
                    ["0", "1", "2"],
                    ["0", "1", "3"],
                    ["0", "2", "3"],
                    ["1", "2", "3"],
                ],
            },
            (4, 6, 4),
            (1, 0, 1),
        ),
        (
            {
                "vertices": ["0", "1", "2", "3"],
                "facets": [["0", "1", "2", "3"]],
            },
            (4, 6, 4, 1),
            (1, 0, 0, 0),
        ),
        (
            {
                "vertices": ["a", "b", "c", "p"],
                "facets": [
                    ["p", "a", "b"],
                    ["p", "b", "c"],
                    ["p", "a", "c"],
                ],
            },
            (4, 6, 3),
            (1, 0, 0),
        ),
        (
            {
                "vertices": ["a", "b", "c", "u", "v"],
                "facets": [
                    ["u", "a", "b"],
                    ["u", "b", "c"],
                    ["u", "a", "c"],
                    ["v", "a", "b"],
                    ["v", "b", "c"],
                    ["v", "a", "c"],
                ],
            },
            (5, 9, 6),
            (1, 0, 1),
        ),
    ),
)
def test_public_reference_cases(
    runtime,
    presentation: dict[str, Any],
    f_vector: tuple[int, ...],
    betti: tuple[int, ...],
) -> None:
    complex_ = _materialize(runtime, presentation)
    computed_betti = _betti(runtime, presentation)

    assert tuple(complex_["f_vector"]) == f_vector
    assert computed_betti == betti
    assert sum((-1) ** index * value for index, value in enumerate(f_vector)) == sum(
        (-1) ** index * value for index, value in enumerate(computed_betti)
    )


def test_torus_and_projective_plane_distinguish_coefficient_fields(runtime) -> None:
    torus = {"vertices": [str(index) for index in range(7)], "facets": _TORUS_FACETS}
    projective_plane = {
        "vertices": [str(index) for index in range(6)],
        "facets": _PROJECTIVE_PLANE_FACETS,
    }

    assert _betti(runtime, torus, prime=2) == (1, 2, 1)
    assert _betti(runtime, torus, prime=3) == (1, 2, 1)
    assert _betti(runtime, projective_plane, prime=2) == (1, 1, 1)
    assert _betti(runtime, projective_plane, prime=3) == (1, 0, 0)


def test_reduced_homology_and_disjoint_union_are_explicit(runtime) -> None:
    three_points = {
        "vertices": ["a", "b", "c"],
        "facets": [["a"], ["b"], ["c"]],
    }
    circle_and_point = {
        "vertices": ["a", "b", "c", "p"],
        "facets": [["a", "b"], ["b", "c"], ["a", "c"], ["p"]],
    }

    assert _betti(runtime, three_points, convention="REDUCED") == (2,)
    assert _betti(runtime, circle_and_point) == (2, 1)


def test_vertex_relabeling_and_input_orientation_preserve_betti_numbers(
    runtime,
) -> None:
    relabeled = {
        "vertices": ["z", "x", "y"],
        "facets": [["y", "x"], ["z", "y"], ["x", "z"]],
    }

    assert _betti(runtime, _CIRCLE) == _betti(runtime, relabeled) == (1, 1)


def test_invalid_topology_request_fails_before_artifact_writes(runtime) -> None:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.materialize",
            input={
                "vertices": ["a", "b"],
                "facets": [["a"], ["a", "b"]],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST"
    assert result.artifact_uris == ()
