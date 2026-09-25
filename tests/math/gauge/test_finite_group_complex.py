import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.gauge import (
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeComplexRequest,
    FiniteGroupGaugeFace,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    construct_finite_group_gauge_complex,
)
from jacobian.math.groups._table_models import FiniteGroupTable


def _group():
    return FiniteGroupTable(multiplication=((0,),), identity=0, inverse=(0,))


def _square():
    lattice = GaugeLattice(
        vertices=("a", "b", "c", "d"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="cd", tail="c", head="d"),
            GaugeEdge(edge_id="da", tail="d", head="a"),
        ),
    )
    oriented = ("ab", "bc", "cd", "da")
    forward = OrientedGaugePath(
        steps=tuple(GaugePathStep(edge_id=edge, forward=True) for edge in oriented),
        basepoint="a",
    )
    reversed_path = OrientedGaugePath(
        steps=tuple(
            GaugePathStep(edge_id=edge, forward=False) for edge in reversed(oriented)
        ),
        basepoint="a",
    )
    return lattice, forward, reversed_path


def test_oriented_square_and_reversed_face_round_trip_with_same_parents():
    lattice, forward, reversed_path = _square()
    request = FiniteGroupGaugeComplexRequest(
        lattice=lattice,
        group=_group(),
        faces=(
            FiniteGroupGaugeFace(face_id="forward", boundary=forward),
            FiniteGroupGaugeFace(face_id="reverse", boundary=reversed_path),
        ),
    )
    result = construct_finite_group_gauge_complex(request)

    assert result.lattice == lattice
    assert result.group == request.group
    assert result.faces[0].boundary.steps == forward.steps
    assert result.faces[1].boundary.steps == reversed_path.steps
    decoded = FiniteGroupGaugeComplex.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert decoded == result


def test_empty_face_tuple_round_trips_as_a_finite_complex():
    lattice, _, _ = _square()
    request = FiniteGroupGaugeComplexRequest(lattice=lattice, group=_group(), faces=())
    result = construct_finite_group_gauge_complex(request)
    assert result.faces == ()
    assert (
        FiniteGroupGaugeComplex.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )


def test_constant_and_backtracking_attaching_maps_are_distinct_degenerate_faces():
    lattice = GaugeLattice(
        vertices=("v",), edges=(GaugeEdge(edge_id="loop", tail="v", head="v"),)
    )
    request = FiniteGroupGaugeComplexRequest(
        lattice=lattice,
        group=_group(),
        faces=(
            FiniteGroupGaugeFace(
                face_id="backtrack",
                boundary=OrientedGaugePath(
                    steps=(
                        GaugePathStep(edge_id="loop", forward=True),
                        GaugePathStep(edge_id="loop", forward=False),
                    )
                ),
            ),
            FiniteGroupGaugeFace(
                face_id="constant",
                boundary=OrientedGaugePath(steps=(), basepoint="v"),
            ),
        ),
    )
    result = construct_finite_group_gauge_complex(request)
    assert len(result.faces[0].boundary.steps) == 2
    assert result.faces[1].boundary.steps == ()
    assert result.faces[1].boundary.basepoint == "v"


def test_open_or_foreign_face_boundary_is_rejected():
    lattice, _, _ = _square()
    request = FiniteGroupGaugeComplexRequest.model_construct(
        lattice=lattice,
        group=_group(),
        faces=(
            FiniteGroupGaugeFace.model_construct(
                face_id="open",
                boundary=OrientedGaugePath.model_construct(
                    steps=(GaugePathStep(edge_id="ab", forward=True),),
                    basepoint=None,
                ),
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="must be closed"):
        construct_finite_group_gauge_complex(request)

    foreign = FiniteGroupGaugeComplex.model_construct(
        lattice=lattice,
        group=_group(),
        faces=(
            FiniteGroupGaugeFace.model_construct(
                face_id="foreign",
                boundary=OrientedGaugePath.model_construct(
                    steps=(
                        GaugePathStep(edge_id="missing", forward=True),
                        GaugePathStep(edge_id="missing", forward=False),
                    ),
                    basepoint=None,
                ),
            ),
        ),
    )
    with pytest.raises(ValidationError, match="source lattice edges"):
        FiniteGroupGaugeComplex.model_validate(foreign.model_dump())


def test_aggregate_face_growth_is_rejected_before_nested_value_parsing():
    lattice = {
        "vertices": ["v"],
        "edges": [{"edge_id": "e", "tail": "v", "head": "v"}],
    }
    boundary = {"steps": [{"edge_id": "e", "forward": True}] * 256}
    raw = {
        "lattice": lattice,
        "group": {"multiplication": [[0]], "identity": 0, "inverse": [0]},
        "faces": [{"face_id": f"f{i:02}", "boundary": boundary} for i in range(17)],
    }
    with pytest.raises(ValidationError, match="complex_boundary_bound"):
        FiniteGroupGaugeComplexRequest.model_validate(raw)


def test_face_orientation_requires_a_strict_boolean():
    raw = {
        "lattice": {
            "vertices": ["v"],
            "edges": [{"edge_id": "e", "tail": "v", "head": "v"}],
        },
        "group": {"multiplication": [[0]], "identity": 0, "inverse": [0]},
        "faces": [
            {
                "face_id": "f",
                "boundary": {"steps": [{"edge_id": "e", "forward": 1}]},
            }
        ],
    }
    with pytest.raises(ValidationError):
        FiniteGroupGaugeComplexRequest.model_validate(raw)


def test_exact_aggregate_face_step_bound_is_accepted():
    edge_id = "e" * 64
    lattice = GaugeLattice(
        vertices=("v",), edges=(GaugeEdge(edge_id=edge_id, tail="v", head="v"),)
    )
    boundary = OrientedGaugePath(
        steps=tuple(GaugePathStep(edge_id=edge_id, forward=True) for _ in range(256))
    )
    request = FiniteGroupGaugeComplexRequest(
        lattice=lattice,
        group=_group(),
        faces=tuple(
            FiniteGroupGaugeFace(face_id=f"f{i:02}", boundary=boundary)
            for i in range(16)
        ),
    )
    assert len(construct_finite_group_gauge_complex(request).faces) == 16


def test_result_size_is_admitted_before_complex_result_construction():
    vertices = tuple(f"v{i:02}" + "x" * 61 for i in range(64))
    edges = tuple(
        GaugeEdge(
            edge_id=f"e{i:03}" + "x" * 60,
            tail=vertices[0],
            head=vertices[0],
        )
        for i in range(128)
    )
    lattice = GaugeLattice(vertices=vertices, edges=edges)
    boundary = OrientedGaugePath(
        steps=tuple(
            GaugePathStep(edge_id=edges[0].edge_id, forward=True) for _ in range(256)
        )
    )
    request = FiniteGroupGaugeComplexRequest(
        lattice=lattice,
        group=_group(),
        faces=tuple(
            FiniteGroupGaugeFace(face_id=f"f{i:02}", boundary=boundary)
            for i in range(16)
        ),
    )
    result = construct_finite_group_gauge_complex(request)
    assert len(result.faces) == 16
