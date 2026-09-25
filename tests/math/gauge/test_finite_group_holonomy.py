from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.gauge import (
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeField,
    FiniteGroupGaugeHolonomyRequest,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    finite_group_gauge_holonomy,
)
from jacobian.math.gauge import finite_group as finite_group_kernel
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
    identity = index[(0, 1, 2)]
    return construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=table, identity=identity)
    ).group, index


def _field(group, index, labels=((1, 2, 0), (1, 0, 2))):
    lattice = GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="e1", tail="a", head="b"),
            GaugeEdge(edge_id="e2", tail="b", head="c"),
        ),
    )
    return FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge,
                value=FiniteGroupTableElement(group=group, index=index[label]),
            )
            for edge, label in zip(("e1", "e2"), labels, strict=True)
        ),
    )


def _s4_group():
    elements = tuple(permutations(range(4)))

    def compose(first, second):
        return tuple(second[first[i]] for i in range(4))

    index = {element: i for i, element in enumerate(elements)}
    table = tuple(tuple(index[compose(a, b)] for b in elements) for a in elements)
    return construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=table, identity=index[(0, 1, 2, 3)])
    ).group


def _loop_field(group, edge_count):
    ids = tuple(f"edge-{i:03}" for i in range(edge_count))
    lattice = GaugeLattice(
        vertices=("v",),
        edges=tuple(GaugeEdge(edge_id=edge_id, tail="v", head="v") for edge_id in ids),
    )
    return FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge_id,
                value=FiniteGroupTableElement(group=group, index=group.identity),
            )
            for edge_id in ids
        ),
    )


def test_noncommutative_path_order_and_serialization():
    group, index = _s3()
    field = _field(group, index)
    path = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="e1", forward=True),
            GaugePathStep(edge_id="e2", forward=True),
        )
    )
    result = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=path)
    )
    expected = group.multiplication[index[(1, 2, 0)]][index[(1, 0, 2)]]
    opposite = group.multiplication[index[(1, 0, 2)]][index[(1, 2, 0)]]
    assert expected != opposite
    assert result.holonomy.index == expected
    decoded = FiniteGroupGaugeHolonomyRequest.model_validate(
        FiniteGroupGaugeHolonomyRequest.model_dump(
            FiniteGroupGaugeHolonomyRequest(field=field, path=path)
        )
    )
    assert finite_group_gauge_holonomy(decoded).holonomy == result.holonomy


def test_backtracking_is_identity_and_reverse_path_inverts_product():
    group, index = _s3()
    field = _field(group, index)
    backtrack = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="e1", forward=True),
            GaugePathStep(edge_id="e1", forward=False),
        )
    )
    identity = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=backtrack)
    )
    assert identity.holonomy.index == group.identity
    path = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="e1", forward=True),
            GaugePathStep(edge_id="e2", forward=True),
        )
    )
    reverse = OrientedGaugePath(
        steps=(
            GaugePathStep(edge_id="e2", forward=False),
            GaugePathStep(edge_id="e1", forward=False),
        )
    )
    forward_value = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=path)
    ).holonomy.index
    reverse_value = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=reverse)
    ).holonomy.index
    assert reverse_value == group.inverse[forward_value]


def test_malformed_constructed_group_is_rejected_structurally():
    group, _ = _s3()
    malformed_group = type(group).model_construct(identity=0, inverse=group.inverse)
    field = _field(group, _s3()[1])
    malformed_field = FiniteGroupGaugeField.model_construct(
        lattice=field.lattice, group=malformed_group, edge_values=field.edge_values
    )
    request = FiniteGroupGaugeHolonomyRequest.model_construct(
        field=malformed_field, path=OrientedGaugePath(steps=(), basepoint="a")
    )
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError):
        finite_group_gauge_holonomy(request)


def test_noncanonical_edge_label_is_rejected():
    group, index = _s3()
    field = _field(group, index)
    malformed_field = FiniteGroupGaugeField.model_construct(
        lattice=field.lattice,
        group=group,
        edge_values=(object(), *field.edge_values[1:]),
    )
    request = FiniteGroupGaugeHolonomyRequest.model_construct(
        field=malformed_field,
        path=OrientedGaugePath(steps=(), basepoint="a"),
    )
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError):
        finite_group_gauge_holonomy(request)


def test_mismatched_nonempty_path_basepoint_is_rejected():
    group, index = _s3()
    field = _field(group, index)
    path = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="e1", forward=True),), basepoint="c"
    )
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError, match="basepoint"):
        finite_group_gauge_holonomy(
            FiniteGroupGaugeHolonomyRequest(field=field, path=path)
        )


def test_edge_parent_substitution_is_rejected():
    group, index = _s3()
    other = construct_finite_group_table(
        FiniteGroupTableRequest(
            multiplication=((0, 1, 2), (1, 2, 0), (2, 0, 1)), identity=0
        )
    ).group
    # Change the table parent while retaining an edge element from the original
    # table; the typed field must reject that mismatched parent.
    values = (
        FiniteGroupGaugeEdgeLabel(
            edge_id="e1",
            value=FiniteGroupTableElement(group=group, index=index[(1, 2, 0)]),
        ),
        FiniteGroupGaugeEdgeLabel(
            edge_id="e2",
            value=FiniteGroupTableElement(group=group, index=index[(1, 0, 2)]),
        ),
    )
    with pytest.raises(ValidationError):
        FiniteGroupGaugeField.model_validate(
            {
                "lattice": _field(group, index).lattice.model_dump(),
                "group": other.model_dump(),
                "edge_values": [v.model_dump() for v in values],
            }
        )


def test_catalog_publishes_finite_group_holonomy():
    from jacobian.catalog.catalog import Catalog

    assert (
        Catalog.open().operation("lattice_gauge.finite_group.holonomy.compute")
        is not None
    )


def test_output_expansion_is_admitted_before_contribution_construction():
    group = _s4_group()
    field = _loop_field(group, 1)
    path = OrientedGaugePath.model_construct(
        steps=tuple(GaugePathStep(edge_id="loop", forward=True) for _ in range(256)),
        basepoint=None,
    )
    with pytest.raises(OperationResourceAdmissionError):
        finite_group_gauge_holonomy(
            FiniteGroupGaugeHolonomyRequest(field=field, path=path)
        )


def test_aggregate_parent_table_output_is_admitted_before_result_construction(
    monkeypatch,
):
    group = _s4_group()
    field = _loop_field(group, 128)
    path = OrientedGaugePath.model_construct(
        steps=tuple(GaugePathStep(edge_id="edge-000", forward=True) for _ in range(50)),
        basepoint=None,
    )

    def output_construction_is_too_late(*args, **kwargs):
        pytest.fail("output admission must precede ledger and result construction")

    monkeypatch.setattr(
        finite_group_kernel,
        "FiniteGroupGaugeContribution",
        output_construction_is_too_late,
    )
    monkeypatch.setattr(
        finite_group_kernel,
        "FiniteGroupGaugeHolonomyResult",
        type(
            "ForbiddenResultConstruction",
            (),
            {"model_construct": staticmethod(output_construction_is_too_late)},
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        finite_group_gauge_holonomy(
            FiniteGroupGaugeHolonomyRequest(field=field, path=path)
        )


def test_smaller_s4_output_boundary_is_accepted():
    group = _s4_group()
    field = _loop_field(group, 1)
    path = OrientedGaugePath.model_construct(
        steps=tuple(GaugePathStep(edge_id="edge-000", forward=True) for _ in range(50)),
        basepoint=None,
    )
    result = finite_group_gauge_holonomy(
        FiniteGroupGaugeHolonomyRequest(field=field, path=path)
    )
    assert result.holonomy.index == group.identity
    assert len(result.contributions) == 50
