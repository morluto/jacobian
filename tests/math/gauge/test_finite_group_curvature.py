"""Exact finite-table face curvature over source-bound oriented 2-complexes."""

from itertools import permutations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge import (
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeComplexRequest,
    FiniteGroupGaugeCurvatureRequest,
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
from jacobian.math.groups._table_models import (
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table


def _s3():
    elements = tuple(permutations(range(3)))

    def compose(first, second):
        return tuple(second[first[i]] for i in range(3))

    index = {element: i for i, element in enumerate(elements)}
    table = tuple(tuple(index[compose(a, b)] for b in elements) for a in elements)
    group = construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=table, identity=index[(0, 1, 2)])
    ).group
    return group, index


def _triangle(group, index):
    lattice = GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )
    field = FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="ab",
                value=FiniteGroupTableElement(group=group, index=index[(1, 2, 0)]),
            ),
            FiniteGroupGaugeEdgeLabel(
                edge_id="bc",
                value=FiniteGroupTableElement(group=group, index=index[(1, 0, 2)]),
            ),
            FiniteGroupGaugeEdgeLabel(
                edge_id="ca",
                value=FiniteGroupTableElement(group=group, index=index[(0, 2, 1)]),
            ),
        ),
    )
    forward = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="ab", forward=True),
            GaugePathStep(edge_id="bc", forward=True),
            GaugePathStep(edge_id="ca", forward=True),
        )
    )
    reverse = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="ca", forward=False),
            GaugePathStep(edge_id="bc", forward=False),
            GaugePathStep(edge_id="ab", forward=False),
        )
    )
    complex_value = construct_finite_group_gauge_complex(
        FiniteGroupGaugeComplexRequest(
            lattice=lattice,
            group=group,
            faces=(
                FiniteGroupGaugeFace(face_id="forward", boundary=forward),
                FiniteGroupGaugeFace(face_id="reverse", boundary=reverse),
            ),
        )
    )
    return lattice, field, complex_value


def test_face_curvature_preserves_orientation_and_table_product():
    group, index = _s3()
    _, field, complex_value = _triangle(group, index)
    result = finite_group_gauge_curvature(
        FiniteGroupGaugeCurvatureRequest(complex=complex_value, field=field)
    )
    identity = group.identity
    to_permutation = {value: key for key, value in index.items()}

    def compose(first, second):
        return tuple(second[first[i]] for i in range(3))

    def inverse(permutation):
        return tuple(permutation.index(i) for i in range(3))

    ab, bc, ca = (to_permutation[entry.value.index] for entry in field.edge_values)
    forward = index[compose(compose(ab, bc), ca)]
    backward = index[compose(compose(inverse(ca), inverse(bc)), inverse(ab))]

    assert tuple(entry.face_id for entry in result.face_values) == (
        "forward",
        "reverse",
    )
    assert tuple(entry.value.index for entry in result.face_values) == (
        forward,
        backward,
    )
    assert result.flat is (forward == identity)


def test_curvature_is_gauge_covariant_for_nonabelian_table_group():
    group, index = _s3()
    _, field, complex_value = _triangle(group, index)
    source = finite_group_gauge_curvature(
        FiniteGroupGaugeCurvatureRequest(complex=complex_value, field=field)
    )
    table = group.multiplication
    frames = {
        "a": index[(1, 2, 0)],
        "b": index[(1, 0, 2)],
        "c": index[(0, 2, 1)],
    }
    labels = {entry.edge_id: entry.value.index for entry in field.edge_values}
    endpoints = {edge.edge_id: (edge.tail, edge.head) for edge in field.lattice.edges}
    transformed = field.model_copy(
        update={
            "edge_values": tuple(
                FiniteGroupGaugeEdgeLabel(
                    edge_id=edge_id,
                    value=FiniteGroupTableElement(
                        group=group,
                        index=table[table[frames[endpoints[edge_id][0]]][value]][
                            group.inverse[frames[endpoints[edge_id][1]]]
                        ],
                    ),
                )
                for edge_id, value in labels.items()
            )
        }
    )
    target = finite_group_gauge_curvature(
        FiniteGroupGaugeCurvatureRequest(complex=complex_value, field=transformed)
    )
    source_by_face = {entry.face_id: entry.value.index for entry in source.face_values}
    target_by_face = {entry.face_id: entry.value.index for entry in target.face_values}
    edge_by_id = {edge.edge_id: edge for edge in field.lattice.edges}
    for face in complex_value.faces:
        first_step = face.boundary.steps[0]
        edge = edge_by_id[first_step.edge_id]
        basepoint = edge.tail if first_step.forward else edge.head
        expected = table[table[frames[basepoint]][source_by_face[face.face_id]]][
            group.inverse[frames[basepoint]]
        ]
        assert target_by_face[face.face_id] == expected
    assert source.flat == target.flat


def test_empty_face_is_identity_and_result_round_trips_through_json():
    group, index = _s3()
    lattice, field, _ = _triangle(group, index)
    complex_value = construct_finite_group_gauge_complex(
        FiniteGroupGaugeComplexRequest(
            lattice=lattice,
            group=group,
            faces=(
                FiniteGroupGaugeFace(
                    face_id="constant",
                    boundary=OrientedGaugePath(steps=(), basepoint="b"),
                ),
            ),
        )
    )
    request = FiniteGroupGaugeCurvatureRequest(complex=complex_value, field=field)
    result = finite_group_gauge_curvature(request)
    assert result.face_values[0].value.index == group.identity
    assert result.flat
    assert type(result).model_validate_json(result.model_dump_json()) == result

    tool = Catalog.open().operation("lattice_gauge.finite_group.curvature.compute")
    assert tool.run(request) == result


def test_parent_table_repetition_is_admitted_before_face_values_are_built():
    order = 24
    table = tuple(tuple((a + b) % order for b in range(order)) for a in range(order))
    group = construct_finite_group_table(
        FiniteGroupTableRequest(
            multiplication=table,
            identity=0,
        )
    ).group
    lattice = GaugeLattice(
        vertices=("v",),
        edges=tuple(
            GaugeEdge(edge_id=f"e{i:03}", tail="v", head="v") for i in range(128)
        ),
    )
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
    complex_value = construct_finite_group_gauge_complex(
        FiniteGroupGaugeComplexRequest(
            lattice=lattice,
            group=group,
            faces=tuple(
                FiniteGroupGaugeFace(
                    face_id=f"f{i:03}",
                    boundary=OrientedGaugePath(steps=(), basepoint="v"),
                )
                for i in range(128)
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_group_gauge_curvature(
            FiniteGroupGaugeCurvatureRequest(complex=complex_value, field=field)
        )
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.curvature_output_bound"
    )


def test_curvature_consumer_rejects_forged_open_face_walk():
    group, index = _s3()
    lattice, field, complex_value = _triangle(group, index)
    forged = type(complex_value).model_construct(
        lattice=lattice,
        group=group,
        faces=(
            FiniteGroupGaugeFace(
                face_id="open",
                boundary=OrientedGaugePath(
                    steps=(GaugePathStep(edge_id="ab", forward=True),)
                ),
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_curvature(
            FiniteGroupGaugeCurvatureRequest.model_construct(
                complex=forged, field=field
            )
        )
    assert error.value.errors()[0]["type"] == "lattice_gauge.complex.face_closed"


def test_curvature_consumer_rejects_missing_forged_complex_fields():
    group, index = _s3()
    lattice, field, _ = _triangle(group, index)
    forged = FiniteGroupGaugeComplex.model_construct(lattice=lattice)

    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_curvature(
            FiniteGroupGaugeCurvatureRequest.model_construct(
                complex=forged, field=field
            )
        )
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.curvature_parent_mismatch"
    )
