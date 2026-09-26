from itertools import permutations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge import (
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyRequest,
    FiniteGroupGaugeTransformRequest,
    FiniteGroupGaugeVertexValue,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    finite_group_gauge_holonomy,
    finite_group_gauge_transform,
)
from jacobian.math.groups._table_models import (
    FiniteGroupTable,
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(right[left[index]] for index in range(len(left)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(permutation.index(index) for index in range(len(permutation)))


def _s3() -> tuple[FiniteGroupTable, dict[tuple[int, ...], int]]:
    elements = tuple(permutations(range(3)))
    indices = {element: index for index, element in enumerate(elements)}
    multiplication = tuple(
        tuple(indices[_compose(left, right)] for right in elements) for left in elements
    )
    group = construct_finite_group_table(
        FiniteGroupTableRequest(
            multiplication=multiplication,
            identity=indices[(0, 1, 2)],
        )
    ).group
    return group, indices


def _field_and_frames() -> tuple[
    FiniteGroupTable,
    dict[tuple[int, ...], int],
    FiniteGroupGaugeField,
    dict[str, tuple[int, ...]],
]:
    group, index = _s3()
    lattice = GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )
    labels = {
        "ab": (1, 0, 2),
        "bc": (1, 2, 0),
        "ca": (0, 1, 2),
    }
    field = FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge_id,
                value=FiniteGroupTableElement(group=group, index=index[label]),
            )
            for edge_id, label in labels.items()
        ),
    )
    frames = {
        "a": (1, 2, 0),
        "b": (0, 2, 1),
        "c": (2, 1, 0),
    }
    return group, index, field, frames


def _request(
    field: FiniteGroupGaugeField,
    index: dict[tuple[int, ...], int],
    frame_values: dict[str, tuple[int, ...]],
) -> FiniteGroupGaugeTransformRequest:
    return FiniteGroupGaugeTransformRequest(
        field=field,
        vertex_values=tuple(
            FiniteGroupGaugeVertexValue(
                vertex=vertex,
                value=FiniteGroupTableElement(
                    group=field.group,
                    index=index[frame_values[vertex]],
                ),
            )
            for vertex in reversed(field.lattice.vertices)
        ),
    )


def test_s3_transform_matches_permutation_oracle_and_endpoint_covariance():
    group, index, field, frames = _field_and_frames()
    result = finite_group_gauge_transform(_request(field, index, frames))
    assert tuple(item.vertex for item in result.vertex_values) == field.lattice.vertices

    source_permutations = {"ab": (1, 0, 2), "bc": (1, 2, 0), "ca": (0, 1, 2)}
    target_permutations = {}
    for edge in field.lattice.edges:
        target_permutations[edge.edge_id] = _compose(
            _compose(frames[edge.tail], source_permutations[edge.edge_id]),
            _inverse(frames[edge.head]),
        )
    assert {
        item.edge_id: item.value.index for item in result.transformed.edge_values
    } == {edge: index[value] for edge, value in target_permutations.items()}

    path = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="ab", forward=True),
            GaugePathStep(edge_id="bc", forward=True),
        )
    )
    source = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=path)
    )
    target = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=result.transformed, path=path)
    )
    source_product = _compose(source_permutations["ab"], source_permutations["bc"])
    expected = _compose(_compose(frames["a"], source_product), _inverse(frames["c"]))
    assert source.holonomy.index == index[source_product]
    assert target.holonomy.index == index[expected]
    assert target.holonomy == FiniteGroupTableElement(
        group=group, index=index[expected]
    )
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert (
        finite_group_gauge_holonomy(
            FiniteGroupGaugeHolonomyRequest(field=decoded.transformed, path=path)
        ).holonomy
        == target.holonomy
    )


def test_identity_action_and_sequential_action_order():
    _, index, field, frames = _field_and_frames()
    identity = dict.fromkeys(field.lattice.vertices, (0, 1, 2))
    unchanged = finite_group_gauge_transform(_request(field, index, identity))
    assert tuple(
        item.value.index for item in unchanged.transformed.edge_values
    ) == tuple(item.value.index for item in field.edge_values)

    first_frames = frames
    second_frames = {
        "a": (0, 2, 1),
        "b": (1, 2, 0),
        "c": (1, 0, 2),
    }
    first = finite_group_gauge_transform(_request(field, index, first_frames))
    sequential = finite_group_gauge_transform(
        _request(first.transformed, index, second_frames)
    )
    composed_frames = {
        vertex: _compose(second_frames[vertex], first_frames[vertex])
        for vertex in field.lattice.vertices
    }
    direct = finite_group_gauge_transform(_request(field, index, composed_frames))
    assert sequential.transformed == direct.transformed
    assert any(
        _compose(second_frames[v], first_frames[v])
        != _compose(first_frames[v], second_frames[v])
        for v in field.lattice.vertices
    )


def test_vertex_frames_must_bind_to_the_field_parent_and_cover_vertices():
    _, index, field, frames = _field_and_frames()
    missing = _request(field, index, frames).model_copy(
        update={"vertex_values": _request(field, index, frames).vertex_values[:-1]}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_transform(missing)
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.transform_vertex_coverage"
    )

    alternate_group = construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=((0, 1), (1, 0)), identity=0)
    ).group
    value = FiniteGroupTableElement(group=alternate_group, index=1)
    mismatched = FiniteGroupGaugeTransformRequest(
        field=field,
        vertex_values=(
            FiniteGroupGaugeVertexValue(vertex="a", value=value),
            *_request(field, index, frames).vertex_values[1:],
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_transform(mismatched)
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.transform_vertex_value"
    )


def test_native_transform_rejects_forged_frame_values():
    _, index, field, frames = _field_and_frames()
    request = _request(field, index, frames)
    forged_values = (
        FiniteGroupGaugeVertexValue.model_construct(vertex="a", value=None),
        FiniteGroupGaugeVertexValue.model_construct(
            value=request.vertex_values[0].value
        ),
    )
    for forged in forged_values:
        malformed = request.model_copy(
            update={"vertex_values": (forged, *request.vertex_values[1:])}
        )
        with pytest.raises(OperationDomainValidationError) as error:
            finite_group_gauge_transform(malformed)
        assert error.value.errors()[0]["type"] == (
            "lattice_gauge.finite_group.transform_vertex_value"
        )


def test_manifest_example_executes():
    tool = Catalog.open().operation(
        "lattice_gauge.finite_group.gauge_transform.compute"
    )
    example = tool.examples[0]
    result = tool.run(tool.request_type.model_validate(example.input))
    assert result.transformed.edge_values[0].value.index == 2


def test_repeated_table_output_admission_has_an_exact_boundary():
    order = 24
    group = construct_finite_group_table(
        FiniteGroupTableRequest(
            multiplication=tuple(
                tuple((left + right) % order for right in range(order))
                for left in range(order)
            ),
            identity=0,
        )
    ).group

    def request_with_edges(count: int) -> FiniteGroupGaugeTransformRequest:
        lattice = GaugeLattice(
            vertices=("v",),
            edges=tuple(
                GaugeEdge(edge_id=f"e{index:03}", tail="v", head="v")
                for index in range(count)
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
        return FiniteGroupGaugeTransformRequest(
            field=field,
            vertex_values=(
                FiniteGroupGaugeVertexValue(
                    vertex="v", value=FiniteGroupTableElement(group=group, index=0)
                ),
            ),
        )

    accepted = finite_group_gauge_transform(request_with_edges(85))
    assert len(accepted.transformed.edge_values) == 85
    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_group_gauge_transform(request_with_edges(86))
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.transform_output_bound"
    )
