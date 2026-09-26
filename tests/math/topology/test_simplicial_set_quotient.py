"""Exact behavior for finite simplicial-set congruence quotients."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalizationError
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import parse_operation_input
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLEX_LABEL_LENGTH,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets._tools import TOOLS as ALL_TOOLS
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapRequest,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.quotient import simplicial_set_quotient
from jacobian.math.topology.simplicial_sets.quotient_models import (
    MAX_CLASS_ID,
    SimplicialSetQuotientRequest,
    SimplicialSetQuotientResult,
)
from jacobian.math.topology.simplicial_sets.quotient_tools import (
    TOOLS as QUOTIENT_TOOLS,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _request(
    degree_class_ids: list[list[int]],
) -> SimplicialSetQuotientRequest:
    return SimplicialSetQuotientRequest.model_validate(
        {
            "simplicial_set": standard_simplex(1, 2).model_dump(mode="json"),
            "degree_class_ids": degree_class_ids,
        }
    )


def test_endpoint_identification_produces_a_circle_prefix() -> None:
    request = _request([[0, 0], [0, 1, 0], [0, 1, 2, 0]])
    result = simplicial_set_quotient(request)
    projection = result.quotient_map

    assert projection.target.sets == (("q0",), ("q0", "q1"), ("q0", "q1", "q2"))
    assert projection.maps == ((0, 0), (0, 1, 0), (0, 1, 2, 0))
    assert projection.target.face_maps == (
        ((0, 0), (0, 0)),
        ((0, 1, 0), (0, 1, 1), (0, 0, 1)),
    )
    assert projection.target.degeneracy_maps == (
        ((0,),),
        ((0, 1), (0, 2)),
    )

    restored = SimplicialSetQuotientResult.model_validate(
        result.model_dump(mode="json")
    )
    assert restored == result
    # Independent public naturality replay after JSON round trip.
    replay = simplicial_map(
        SimplicialMapRequest(
            source=restored.quotient_map.source,
            target=restored.quotient_map.target,
            maps=restored.quotient_map.maps,
        )
    )
    assert replay.identities_preserved


def test_all_classes_merged_gives_a_point_and_empty_prefix_stays_empty() -> None:
    point = simplicial_set_quotient(_request([[0, 0], [0, 0, 0], [0, 0, 0, 0]]))
    assert point.quotient_map.target.sets == (("q0",), ("q0",), ("q0",))
    assert point.quotient_map.maps == ((0, 0), (0, 0, 0), (0, 0, 0, 0))

    from jacobian.math.topology.simplicial_sets.operations import from_tables

    empty = from_tables(0, ((),), (), ())
    assert empty.simplicial_set is not None
    empty_quotient = simplicial_set_quotient(
        SimplicialSetQuotientRequest(
            simplicial_set=empty.simplicial_set,
            degree_class_ids=((),),
        )
    )
    assert empty_quotient.quotient_map.target.sets == ((),)
    assert empty_quotient.quotient_map.maps == ((),)


def test_full_carrier_boundary_remains_admitted() -> None:
    from jacobian.math.topology.simplicial_sets.operations import from_tables

    labels = tuple(f"x{index}" for index in range(32))
    identity = tuple(range(32))
    source = from_tables(
        2,
        (labels, labels, labels),
        (
            (identity, identity),
            (identity, identity, identity),
        ),
        (
            (identity,),
            (identity, identity),
        ),
    )
    assert source.simplicial_set is not None
    assert source.simplicial_set.total_simplices == 96
    request = SimplicialSetQuotientRequest(
        simplicial_set=source.simplicial_set,
        degree_class_ids=(tuple(range(32)),) * 3,
    )
    result = simplicial_set_quotient(request)
    assert result.quotient_map.target.total_simplices == 96
    assert result.quotient_map.maps == (identity, identity, identity)


def test_class_labels_are_independent_of_the_simplex_count() -> None:
    # Labels beyond the 96-simplex carrier stay admissible; only their equality
    # defines the relation, so their magnitude changes neither the quotient nor
    # the transport envelope below the JSON-safe integer bound.
    request = _request([[97, 97], [101, 102, 101], [103, 104, 105, 103]])
    result = simplicial_set_quotient(request)
    assert result.quotient_map.maps[0] == (0, 0)

    malformed = SimplicialSetQuotientRequest.model_construct()
    with pytest.raises(OperationDomainValidationError):
        simplicial_set_quotient(malformed)


def test_class_labels_use_the_lossless_json_transport_envelope() -> None:
    boundary_ids = [
        [0, 0],
        [0, 1, 0],
        [0, 1, MAX_CLASS_ID, 0],
    ]
    native = _request(boundary_ids)
    source_json = standard_simplex(1, 2).model_dump(mode="json")
    wire = parse_operation_input(
        SimplicialSetQuotientRequest,
        {"simplicial_set": source_json, "degree_class_ids": boundary_ids},
    )
    assert wire == native
    assert simplicial_set_quotient(wire).quotient_map.maps[0] == (0, 0)

    # Above the envelope neither native validation nor the JSON transport
    # admits the label, and a forged native construct shares owner admission.
    with pytest.raises(ValidationError):
        _request(
            [
                [0, 0],
                [0, 1, 0],
                [0, 1, MAX_CLASS_ID + 1, 0],
            ]
        )
    with pytest.raises(CanonicalizationError):
        parse_operation_input(
            SimplicialSetQuotientRequest,
            {
                "simplicial_set": source_json,
                "degree_class_ids": [
                    [0, 0],
                    [0, 1, 0],
                    [0, 1, MAX_CLASS_ID + 1, 0],
                ],
            },
        )
    forged = SimplicialSetQuotientRequest.model_construct(
        simplicial_set=standard_simplex(1, 2),
        degree_class_ids=(
            (0, 0),
            (0, 1, 0),
            (0, 1, MAX_CLASS_ID + 1, 0),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_set_quotient(forged)
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.quotient.class_axis_invalid"
    )


def test_malformed_native_source_is_structurally_rejected() -> None:
    oversized_label = "x" * (MAX_SIMPLEX_LABEL_LENGTH + 1)
    cases = (
        (
            FiniteTruncatedSimplicialSet.model_construct(
                max_degree=1,
                sets=((oversized_label, "b"), ("ab", "ba", "bb")),
                face_maps=(((0, 1, 1), (0, 0, 1)),),
                degeneracy_maps=(((0, 2),),),
                total_simplices=5,
                checked_identities=0,
            ),
            "simplicial_set.degree_set_invalid",
        ),
        (
            FiniteTruncatedSimplicialSet.model_construct(
                max_degree=1,
                sets=(("a", "b"), ("ab", "ba", "bb")),
                face_maps=(((0, 1, 1), (0, 0, 1, 0)),),
                degeneracy_maps=(((0, 2),),),
                total_simplices=5,
                checked_identities=0,
            ),
            "simplicial_set.face_map_axis_invalid",
        ),
    )
    for source, expected_code in cases:
        request = SimplicialSetQuotientRequest.model_construct(
            simplicial_set=source,
            degree_class_ids=((0, 0), (0, 0, 0)),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            simplicial_set_quotient(request)
        assert error.value.errors()[0]["type"] == expected_code


def test_published_tool_example_dispatches_and_serializes() -> None:
    operation_id = "topology.simplicial_set.quotient_by_congruence.compute"
    assert any(tool.operation_id == operation_id for tool in ALL_TOOLS)
    tool = QUOTIENT_TOOLS[0]
    assert tool.operation_id == operation_id
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.quotient_map.target.sets == (
        ("q0",),
        ("q0",),
        ("q0",),
    )


def test_equivalent_vertices_must_have_equivalent_degeneracies() -> None:
    request = _request([[0, 0], [0, 1, 0], [0, 1, 2, 3]])
    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_set_quotient(request)
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.quotient.degeneracy_congruence_failed"
    )


def test_equivalent_edges_must_have_equivalent_faces() -> None:
    request = _request([[0, 1], [0, 0, 2], [0, 1, 2, 3]])
    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_set_quotient(request)
    assert error.value.errors()[0]["type"] == (
        "simplicial_set.quotient.face_congruence_failed"
    )
