import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.gauge import (
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeFace,
    FiniteGroupGaugeField,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    construct_finite_group_gauge_complex,
    finite_group_gauge_curvature,
)
from jacobian.math.gauge._models import FiniteGroupGaugeComplexRequest
from jacobian.math.gauge.finite_group_complex import _admit_faces, _admit_lattice
from jacobian.math.groups._table_models import FiniteGroupTable, FiniteGroupTableElement


def _group() -> FiniteGroupTable:
    return FiniteGroupTable(multiplication=((0,),), identity=0, inverse=(0,))


def _square() -> tuple[GaugeLattice, OrientedGaugePath, OrientedGaugePath]:
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


def test_oriented_square_and_reversed_face_round_trip_with_same_parents() -> None:
    lattice, forward, reversed_path = _square()
    request = FiniteGroupGaugeComplexRequest(
        lattice=lattice,
        group=_group(),
        faces=(
            FiniteGroupGaugeFace(face_id="forward", boundary=forward),
            FiniteGroupGaugeFace(face_id="reverse", boundary=reversed_path),
        ),
    )
    result = construct_finite_group_gauge_complex(
        request.lattice, request.group, request.faces
    )

    assert result.lattice == lattice
    assert result.group == request.group
    assert result.faces[0].boundary.steps == forward.steps
    assert result.faces[1].boundary.steps == reversed_path.steps
    decoded = FiniteGroupGaugeComplex.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert decoded == result


def test_constant_and_backtracking_attaching_maps_are_distinct_degenerate_faces() -> (
    None
):
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
    result = construct_finite_group_gauge_complex(
        request.lattice, request.group, request.faces
    )
    assert len(result.faces[0].boundary.steps) == 2
    assert result.faces[1].boundary.steps == ()
    assert result.faces[1].boundary.basepoint == "v"


def test_open_or_foreign_face_boundary_is_rejected() -> None:
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
    with pytest.raises(OperationDomainValidationError) as exc_info:
        construct_finite_group_gauge_complex(
            request.lattice, request.group, request.faces
        )
    assert exc_info.value.errors()[0]["type"] == "lattice_gauge.complex.face_closed"

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
    with pytest.raises(ValidationError) as exc_info:
        FiniteGroupGaugeComplex.model_validate(foreign.model_dump())
    assert exc_info.value.errors()[0]["type"] == "lattice_gauge.complex_face_edge"


def test_aggregate_face_growth_is_rejected_before_nested_value_parsing() -> None:
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
    with pytest.raises(ValidationError) as exc_info:
        FiniteGroupGaugeComplexRequest.model_validate(raw)
    assert exc_info.value.errors()[0]["type"] == "lattice_gauge.complex_boundary_bound"


def test_face_orientation_requires_a_strict_boolean() -> None:
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


def test_exact_aggregate_face_step_bound_is_accepted() -> None:
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
    assert (
        len(
            construct_finite_group_gauge_complex(
                request.lattice, request.group, request.faces
            ).faces
        )
        == 16
    )


def test_maximum_length_labels_are_not_rejected_by_a_serialized_size_estimate() -> None:
    """Native admission bounds materialized cells, not encoded characters.

    Every label here is at the MAX_GAUGE_LABEL_LENGTH ceiling and the lattice,
    face count, aggregate steps, work, and allocation are each independently
    bounded, so this complex is admitted. It was previously rejected only
    because a character-width heuristic exceeded the output-unit envelope,
    which made a delivery-oriented width part of the mathematical domain.
    """
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
    complex_ = construct_finite_group_gauge_complex(
        request.lattice, request.group, request.faces
    )
    assert len(complex_.faces) == 16
    assert complex_.lattice == request.lattice
    assert type(complex_).model_validate_json(complex_.model_dump_json()) == complex_


@pytest.mark.parametrize("admission", ("native", "value"))
@pytest.mark.parametrize("explicit_basepoint", (False, True))
@pytest.mark.parametrize("group_order", (8, 24))
def test_maximum_mixed_step_and_constant_faces_are_accepted(
    admission: str, explicit_basepoint: bool, group_order: int
) -> None:
    """Saturated step counts can coexist with 112 constant-face basepoints."""
    vertices = tuple(f"v{i:02}" for i in range(64))
    lattice = GaugeLattice(
        vertices=vertices,
        edges=tuple(
            GaugeEdge(edge_id=f"e{i:03}", tail=vertices[0], head=vertices[0])
            for i in range(128)
        ),
    )
    group = FiniteGroupTable(
        multiplication=tuple(
            tuple((i + j) % group_order for j in range(group_order))
            for i in range(group_order)
        ),
        identity=0,
        inverse=tuple((-i) % group_order for i in range(group_order)),
    )
    boundary = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="e000", forward=True),) * 256,
        basepoint=vertices[0] if explicit_basepoint else None,
    )
    constant = OrientedGaugePath(steps=(), basepoint=vertices[0])
    faces = tuple(
        FiniteGroupGaugeFace(
            face_id=f"f{i:03}", boundary=boundary if i < 16 else constant
        )
        for i in range(128)
    )
    # Each face retains its ID, boundary, every step, and any explicit basepoint.
    _, _, vertex_set, edge_by_id, lattice_units = _admit_lattice(lattice)
    face_units = _admit_faces(faces, vertex_set, edge_by_id)
    assert face_units == 2 * 128 + 4096 + 112 + (16 if explicit_basepoint else 0)
    if group_order == 24:
        assert lattice_units + 24**2 + 24 + face_units == (
            5592 if explicit_basepoint else 5576
        )
    request = FiniteGroupGaugeComplexRequest(lattice=lattice, group=group, faces=faces)
    if admission == "native":
        result = construct_finite_group_gauge_complex(
            request.lattice, request.group, request.faces
        )
    else:
        result = FiniteGroupGaugeComplex.model_validate_json(request.model_dump_json())
    assert result.lattice == lattice
    assert result.group == group
    assert result.faces == faces
    assert sum(len(face.boundary.steps) for face in result.faces) == 4096
    assert sum(not face.boundary.steps for face in result.faces) == 112
    assert (
        FiniteGroupGaugeComplex.model_validate_json(result.model_dump_json()) == result
    )
    # The order-24 carrier exceeds curvature's independent output envelope.
    if group_order == 8:
        decoded = FiniteGroupGaugeComplex.model_validate_json(result.model_dump_json())
        field = FiniteGroupGaugeField(
            lattice=lattice,
            group=group,
            edge_values=tuple(
                FiniteGroupGaugeEdgeLabel(
                    edge_id=edge.edge_id,
                    value=FiniteGroupTableElement(group=group, index=0),
                )
                for edge in lattice.edges
            ),
        )
        curvature = finite_group_gauge_curvature(decoded, field)
        assert curvature.flat
        assert len(curvature.face_values) == 128
        assert all(face.value.index == 0 for face in curvature.face_values)
