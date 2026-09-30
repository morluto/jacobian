from __future__ import annotations

from itertools import permutations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge import (
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeField,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    finite_group_gauge_basepoint_transport,
    finite_group_gauge_holonomy,
)
from jacobian.math.gauge._models import FiniteGroupGaugeBasepointTransportRequest
from jacobian.math.gauge._tools import TOOLS
from jacobian.math.groups._table_models import (
    FiniteGroupTable,
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(second[first[index]] for index in range(len(first)))


def _s3() -> tuple[FiniteGroupTable, dict[tuple[int, ...], int]]:
    elements = tuple(permutations(range(3)))
    index = {element: position for position, element in enumerate(elements)}
    table = tuple(
        tuple(index[_compose(first, second)] for second in elements)
        for first in elements
    )
    identity = index[(0, 1, 2)]
    return (
        construct_finite_group_table(
            FiniteGroupTableRequest(multiplication=table, identity=identity)
        ).group,
        index,
    )


def _nonabelian_field() -> tuple[FiniteGroupGaugeField, dict[tuple[int, ...], int]]:
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
        "ab": (0, 2, 1),
        "bc": (1, 0, 2),
        "ca": (0, 1, 2),
    }
    field = FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge_id,
                value=FiniteGroupTableElement(
                    group=group, index=index[labels[edge_id]]
                ),
            )
            for edge_id in ("ab", "bc", "ca")
        ),
    )
    return field, index


def _loop() -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="ab", forward=True),
            GaugePathStep(edge_id="bc", forward=True),
            GaugePathStep(edge_id="ca", forward=True),
        ),
        basepoint="a",
    )


def test_nonabelian_transport_is_exact_conjugation_and_composes_after_json() -> None:
    field, index = _nonabelian_field()
    connector = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="ab", forward=True),), basepoint="a"
    )
    request = FiniteGroupGaugeBasepointTransportRequest(
        field=field, loop=_loop(), connector=connector
    )

    result = finite_group_gauge_basepoint_transport(
        request.field, request.loop, request.connector
    )

    p = (0, 2, 1)
    q = (1, 0, 2)
    source_holonomy = _compose(_compose(p, q), (0, 1, 2))
    expected_transport = _compose(
        _compose(p, source_holonomy), p
    )  # p is its own inverse
    assert result.source_basepoint == "a"
    assert result.target_basepoint == "b"
    assert result.source_holonomy.index == index[source_holonomy]
    assert result.connector_holonomy.index == index[p]
    assert result.transported_holonomy.index == index[expected_transport]
    assert source_holonomy != expected_transport
    assert result.transported_loop.steps == (
        GaugePathStep(edge_id="ab", forward=False),
        *_loop().steps,
        GaugePathStep(edge_id="ab", forward=True),
    )

    decoded = type(result).model_validate_json(result.model_dump_json())
    consumer_value = finite_group_gauge_holonomy(
        decoded.field, decoded.transported_loop
    )
    assert consumer_value.holonomy == decoded.transported_holonomy


def test_empty_connector_and_empty_loop_keep_identity_cases_exact() -> None:
    field, _ = _nonabelian_field()
    empty_connector = OrientedGaugePath(steps=(), basepoint="a")
    loop_result = finite_group_gauge_basepoint_transport(
        field, _loop(), empty_connector
    )
    assert loop_result.target_basepoint == "a"
    assert loop_result.transported_holonomy == loop_result.source_holonomy
    assert loop_result.transported_loop == _loop()

    empty_loop = OrientedGaugePath(steps=(), basepoint="a")
    connector = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="ab", forward=True),), basepoint="a"
    )
    identity_result = finite_group_gauge_basepoint_transport(
        field, empty_loop, connector
    )
    assert identity_result.source_holonomy.index == field.group.identity
    assert identity_result.transported_holonomy.index == field.group.identity


def test_open_source_path_is_rejected() -> None:
    field, _ = _nonabelian_field()
    open_path = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="ab", forward=True),), basepoint="a"
    )
    connector = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="ab", forward=True),), basepoint="a"
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_basepoint_transport(field, open_path, connector)
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.loop_not_closed"
    )


def test_transported_path_admits_exactly_256_steps_and_rejects_above() -> None:
    group = FiniteGroupTable(multiplication=((0,),), identity=0, inverse=(0,))
    lattice = GaugeLattice(
        vertices=("v",),
        edges=(GaugeEdge(edge_id="loop", tail="v", head="v"),),
    )
    field = FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="loop",
                value=FiniteGroupTableElement(group=group, index=0),
            ),
        ),
    )
    loop = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="loop", forward=True),
            GaugePathStep(edge_id="loop", forward=False),
        ),
        basepoint="v",
    )

    exact = finite_group_gauge_basepoint_transport(
        field,
        loop,
        OrientedGaugePath(
            steps=tuple(
                GaugePathStep(edge_id="loop", forward=True) for _ in range(127)
            ),
            basepoint="v",
        ),
    )
    assert len(exact.transported_loop.steps) == 256

    with pytest.raises(OperationResourceAdmissionError) as error:
        finite_group_gauge_basepoint_transport(
            field,
            loop,
            OrientedGaugePath(
                steps=tuple(
                    GaugePathStep(edge_id="loop", forward=True) for _ in range(128)
                ),
                basepoint="v",
            ),
        )
    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.basepoint_path_bound"
    )


def test_catalog_publishes_the_basepoint_transport_operation() -> None:
    assert any(
        tool.operation_id == "lattice_gauge.finite_group.basepoint_transport.compute"
        for tool in TOOLS
    )


@pytest.mark.parametrize("reverse_loop", (False, True))
@pytest.mark.parametrize("reverse_connector", (False, True))
def test_transport_round_trips_with_non_involutive_reverse_labels(
    reverse_loop: bool, reverse_connector: bool
) -> None:
    field, index = _nonabelian_field()
    labels = {"ab": (1, 0, 2), "bc": (1, 2, 0), "ca": (1, 2, 0)}
    field = FiniteGroupGaugeField(
        lattice=field.lattice,
        group=field.group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge_id,
                value=FiniteGroupTableElement(group=field.group, index=index[label]),
            )
            for edge_id, label in labels.items()
        ),
    )
    loop = _loop()
    if reverse_loop:
        loop = OrientedGaugePath(
            steps=tuple(
                GaugePathStep(edge_id=step.edge_id, forward=False)
                for step in reversed(loop.steps)
            ),
            basepoint="a",
        )
    connector = OrientedGaugePath(
        steps=(
            GaugePathStep(
                edge_id="ca" if reverse_connector else "ab",
                forward=not reverse_connector,
            ),
        ),
        basepoint="a",
    )

    def permutation_product(path: OrientedGaugePath) -> tuple[int, ...]:
        product: tuple[int, ...] = (0, 1, 2)
        for step in path.steps:
            element: tuple[int, ...] = labels[step.edge_id]
            if not step.forward:
                element = tuple(element.index(i) for i in range(3))
            product = _compose(product, element)
        return product

    result = finite_group_gauge_basepoint_transport(field, loop, connector)
    assert result.source_holonomy.index == index[permutation_product(loop)]
    assert result.connector_holonomy.index == index[permutation_product(connector)]
    assert (
        result.transported_holonomy.index
        == index[permutation_product(result.transported_loop)]
    )
    assert type(result).model_validate(result.model_dump()) == result
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert (
        finite_group_gauge_holonomy(decoded.field, decoded.transported_loop).holonomy
        == result.transported_holonomy
    )
