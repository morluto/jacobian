from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.topology import build_topology_bundle


@pytest.fixture
def topology_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install only the topology bundle exercised by these behaviors."""

    with open_domain_services(tmp_path / "state", build_topology_bundle()) as services:
        yield services


def test_every_topology_operation_advertises_an_executable_example(
    topology_services: DomainTestServices,
) -> None:
    bundle = build_topology_bundle()

    for operation in bundle.capabilities:
        spec = operation.spec
        assert spec.invocation_examples, spec.operation_id
        example = spec.invocation_examples[0]
        result = topology_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=spec.operation_id,
                input=example.input,
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, (
            spec.operation_id,
            result.diagnostics,
        )


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


def _result_payload(_runtime, result) -> dict[str, Any]:
    return result.output["result"]


def _canonicalize(topology_services, presentation: dict[str, Any]) -> dict[str, Any]:
    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.canonicalize",
            input=presentation,
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return _result_payload(topology_services, result)["complex"]


def _betti(
    topology_services,
    presentation: dict[str, Any],
    *,
    prime: int = 2,
    convention: str = "UNREDUCED",
) -> tuple[int, ...]:
    complex_ = _canonicalize(topology_services, presentation)
    result = topology_services.core.capabilities.invoke(
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
    assert result.artifact_uris == ()
    return tuple(
        group["betti_number"]
        for group in _result_payload(topology_services, result)["groups"]
    )


def test_canonicalization_is_canonical_complete_inline_and_composable(
    topology_services,
) -> None:
    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.canonicalize",
            input={
                "vertices": ["c", "a", "b"],
                "facets": [["b", "a"], ["c", "b"], ["c", "a"]],
            },
        )
    )

    payload = _result_payload(topology_services, result)
    complex_ = payload["complex"]
    assert complex_["vertices"] == ["a", "b", "c"]
    assert complex_["maximal_simplices"] == [
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
    ]
    assert complex_["f_vector"] == [3, 3]
    assert complex_["closure_size"] == 6
    assert complex_["empty_simplex_stored"] is False
    assert payload["completeness"] == "COMPLETE_FACE_CLOSURE"
    assert result.artifact_uris == ()
    assert result.output["result"] == payload


def test_chain_complex_exposes_oriented_sparse_boundaries_and_augmentation(
    topology_services,
) -> None:
    triangle = _canonicalize(
        topology_services,
        {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]},
    )
    integer_result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input={
                "complex": triangle,
                "coefficient_ring": "INTEGER",
                "convention": "UNREDUCED",
            },
        )
    )
    integer = _result_payload(topology_services, integer_result)
    mod_two_reduced_result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.chain_complex.compute",
            input={
                "complex": triangle,
                "coefficient_ring": "PRIME_FIELD",
                "prime": 2,
                "convention": "REDUCED",
            },
        )
    )
    mod_two_reduced = _result_payload(topology_services, mod_two_reduced_result)

    assert integer_result.artifact_uris == ()
    assert mod_two_reduced_result.artifact_uris == ()

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
    topology_services,
    presentation: dict[str, Any],
    f_vector: tuple[int, ...],
    betti: tuple[int, ...],
) -> None:
    complex_ = _canonicalize(topology_services, presentation)
    computed_betti = _betti(topology_services, presentation)

    assert tuple(complex_["f_vector"]) == f_vector
    assert computed_betti == betti
    assert sum((-1) ** index * value for index, value in enumerate(f_vector)) == sum(
        (-1) ** index * value for index, value in enumerate(computed_betti)
    )


def test_torus_and_projective_plane_distinguish_coefficient_fields(
    topology_services,
) -> None:
    torus = {"vertices": [str(index) for index in range(7)], "facets": _TORUS_FACETS}
    projective_plane = {
        "vertices": [str(index) for index in range(6)],
        "facets": _PROJECTIVE_PLANE_FACETS,
    }

    assert _betti(topology_services, torus, prime=2) == (1, 2, 1)
    assert _betti(topology_services, torus, prime=3) == (1, 2, 1)
    assert _betti(topology_services, projective_plane, prime=2) == (1, 1, 1)
    assert _betti(topology_services, projective_plane, prime=3) == (1, 0, 0)


def test_integral_homology_exposes_free_and_torsion_generators(
    topology_services,
) -> None:
    circle = _canonicalize(topology_services, _CIRCLE)
    projective_plane = _canonicalize(
        topology_services,
        {
            "vertices": [str(index) for index in range(6)],
            "facets": _PROJECTIVE_PLANE_FACETS,
        },
    )

    circle_result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={"complex": circle, "convention": "UNREDUCED"},
        )
    )
    projective_result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={"complex": projective_plane, "convention": "UNREDUCED"},
        )
    )

    assert circle_result.execution.status is ExecutionStatus.COMPLETED
    circle_groups = _result_payload(topology_services, circle_result)["groups"]
    assert [group["betti_number"] for group in circle_groups] == [1, 1]
    assert [group["torsion_coefficients"] for group in circle_groups] == [[], []]
    assert all(
        group["generator_basis"]
        == "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"
        for group in circle_groups
    )

    assert projective_result.execution.status is ExecutionStatus.COMPLETED
    projective_groups = _result_payload(topology_services, projective_result)["groups"]
    assert [group["betti_number"] for group in projective_groups] == [1, 0, 0]
    assert [group["torsion_coefficients"] for group in projective_groups] == [
        [],
        ["2"],
        [],
    ]
    torsion = projective_groups[1]["torsion_generators"][0]
    assert torsion["order"] == "2"
    assert len(torsion["cycle"]["coefficients"]) == projective_plane["f_vector"][1]
    assert (
        len(torsion["bounding_chain"]["coefficients"])
        == (projective_plane["f_vector"][2])
    )
    assert projective_result.artifact_uris == ()


def test_reduced_integral_homology_uses_the_augmentation_kernel(
    topology_services,
) -> None:
    points = _canonicalize(
        topology_services,
        {
            "vertices": ["a", "b", "c"],
            "facets": [["a"], ["b"], ["c"]],
        },
    )
    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={"complex": points, "convention": "REDUCED"},
        )
    )

    group = _result_payload(topology_services, result)["groups"][0]
    assert group["outgoing_boundary_rank"] == 1
    assert group["betti_number"] == 2
    assert len(group["free_generators"]) == 2


def test_reduced_homology_and_disjoint_union_are_explicit(
    topology_services,
) -> None:
    three_points = {
        "vertices": ["a", "b", "c"],
        "facets": [["a"], ["b"], ["c"]],
    }
    circle_and_point = {
        "vertices": ["a", "b", "c", "p"],
        "facets": [["a", "b"], ["b", "c"], ["a", "c"], ["p"]],
    }

    assert _betti(topology_services, three_points, convention="REDUCED") == (2,)
    assert _betti(topology_services, circle_and_point) == (2, 1)


def test_vertex_relabeling_and_input_orientation_preserve_betti_numbers(
    topology_services,
) -> None:
    relabeled = {
        "vertices": ["z", "x", "y"],
        "facets": [["y", "x"], ["z", "y"], ["x", "z"]],
    }

    assert (
        _betti(topology_services, _CIRCLE)
        == _betti(topology_services, relabeled)
        == (1, 1)
    )


def test_invalid_topology_request_fails_before_artifact_writes(
    topology_services,
) -> None:
    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.canonicalize",
            input={
                "vertices": ["a", "b"],
                "facets": [["a"], ["a", "b"]],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST"
    assert result.artifact_uris == ()


def test_stale_complex_digest_surfaces_precise_field_in_diagnostic(
    topology_services,
) -> None:
    """A stale ``complex_digest`` must fail closed *and* surface the precise
    Pydantic validator message in the diagnostic ``hint`` and the nested
    field path in ``path`` — not just the generic bundle wording.
    """

    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_homology.integral.compute",
            input={
                "complex": {
                    "complex_format": "jacobian.finite-simplicial-complex/v1",
                    "vertices": ["x", "y", "z"],
                    "maximal_simplices": [["x", "y"], ["x", "z"], ["y", "z"]],
                    "faces_by_dimension": [
                        {
                            "dimension": 0,
                            "faces": [["x"], ["y"], ["z"]],
                        },
                        {
                            "dimension": 1,
                            "faces": [["x", "y"], ["x", "z"], ["y", "z"]],
                        },
                    ],
                    "dimension": 1,
                    "f_vector": [3, 3],
                    "closure_size": 6,
                    "orientation_convention": "LEXICOGRAPHIC_VERTEX_ORDER",
                    "empty_simplex_stored": False,
                    "complex_digest": (
                        "sha256:"
                        "6f797991bac967e2a8e572707df487061655df0f094c"
                        "bde0f52f82c5401fc043"
                    ),
                },
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST"
    assert result.diagnostics[0].path == "complex/complex_digest"
    assert "complex_digest" in (result.diagnostics[0].hint or "")
    assert result.artifact_uris == ()


def test_enriched_diagnostic_still_fails_closed_for_non_digest_error(
    topology_services,
) -> None:
    """A non-digest validation error (non-maximal facets) must still fail
    closed with the correct code and no artifacts, while the enriched
    ``hint`` surfaces the specific validator message.
    """

    result = topology_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="topology.simplicial_complex.canonicalize",
            input={
                "vertices": ["a", "b"],
                "facets": [["a"], ["a", "b"]],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST"
    assert "maximal" in (result.diagnostics[0].hint or "")
    assert result.artifact_uris == ()
