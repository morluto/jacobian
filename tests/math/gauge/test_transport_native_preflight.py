"""Transport decoding admits retained carriers before traversing their axes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from jacobian._models import StrictModel
from jacobian.math.gauge import (
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeField,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    finite_group_gauge_basepoint_transport,
)
from jacobian.math.gauge._models import FiniteGroupGaugeBasepointTransportResult
from jacobian.math.groups._table_models import FiniteGroupTable, FiniteGroupTableElement


def _result() -> FiniteGroupGaugeBasepointTransportResult:
    group = FiniteGroupTable(
        multiplication=((0, 1, 2), (1, 2, 0), (2, 0, 1)),
        identity=0,
        inverse=(0, 2, 1),
    )
    field = FiniteGroupGaugeField(
        lattice=GaugeLattice(
            vertices=("a",), edges=(GaugeEdge(edge_id="e", tail="a", head="a"),)
        ),
        group=group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="e", value=FiniteGroupTableElement(group=group, index=1)
            ),
        ),
    )
    path = OrientedGaugePath(
        steps=(GaugePathStep(edge_id="e", forward=False),), basepoint="a"
    )
    return finite_group_gauge_basepoint_transport(field, path, path)


def _at(value: object, path: tuple[str | int, ...]) -> object:
    for part in path:
        if isinstance(part, str):
            assert isinstance(value, StrictModel)
            value = value.__dict__[part]
        else:
            assert isinstance(value, tuple)
            value = value[part]
    return value


def _replace(value: object, path: tuple[str | int, ...], replacement: object) -> object:
    if not path:
        return replacement
    part, *tail = path
    if isinstance(part, str):
        assert isinstance(value, StrictModel)
        return value.model_copy(
            update={part: _replace(value.__dict__[part], tuple(tail), replacement)}
        )
    assert isinstance(value, tuple)
    return tuple(
        _replace(item, tuple(tail), replacement) if index == part else item
        for index, item in enumerate(value)
    )


def _decode(value: object, wrapped: bool) -> FiniteGroupGaugeBasepointTransportResult:
    assert isinstance(value, FiniteGroupGaugeBasepointTransportResult)
    return type(value).model_validate(value if wrapped else dict(value.__dict__))


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "path",
    (
        (),
        ("field",),
        ("field", "lattice"),
        ("field", "lattice", "edges", 0),
        ("field", "group"),
        ("field", "edge_values", 0),
        ("field", "edge_values", 0, "value"),
        ("field", "edge_values", 0, "value", "group"),
        ("source_holonomy",),
        ("source_holonomy", "group"),
        ("connector_holonomy",),
        ("transported_holonomy",),
    ),
)
def test_transport_rejects_each_missing_native_field(
    wrapped: bool, path: tuple[str | int, ...]
) -> None:
    result = _result()
    carrier = _at(result, path)
    assert isinstance(carrier, StrictModel)
    for missing in carrier.__dict__:
        fields = {
            key: value for key, value in carrier.__dict__.items() if key != missing
        }
        forged = _replace(result, path, type(carrier).model_construct(**fields))
        with pytest.raises(ValidationError):
            _decode(forged, wrapped)


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "path",
    (
        ("field", "lattice", "edges", 0),
        ("field", "lattice", "edges", 0, "edge_id"),
        ("field", "lattice", "edges", 0, "tail"),
        ("field", "lattice", "vertices", 0),
        ("field", "edge_values", 0, "edge_id"),
        ("source_holonomy", "index"),
        ("connector_holonomy", "group", "identity"),
        ("source_basepoint",),
    ),
)
def test_transport_rejects_malformed_leaves_before_copying(
    wrapped: bool, path: tuple[str | int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = _replace(_result(), path, ["invalid"])

    def unexpected_dump(*_args: object, **_kwargs: object) -> object:
        pytest.fail("malformed retained leaves must be rejected before serialization")

    monkeypatch.setattr(StrictModel, "model_dump", unexpected_dump)
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


@pytest.mark.parametrize("wrapped", (False, True))
def test_transport_does_not_iterate_noncanonical_lattice_axes(wrapped: bool) -> None:
    class UnvisitedEdges(tuple[GaugeEdge, ...]):
        def __len__(self) -> int:
            raise AssertionError("axis subclass must not be measured")

        def __iter__(self) -> Iterator[GaugeEdge]:
            raise AssertionError("axis subclass must not be traversed")

    forged = _replace(_result(), ("field", "lattice", "edges"), UnvisitedEdges())
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


@pytest.mark.parametrize("wrapped", (False, True))
def test_transport_rejects_an_edge_with_foreign_endpoint(wrapped: bool) -> None:
    forged = _replace(_result(), ("field", "lattice", "edges", 0, "tail"), "foreign")
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    ("path", "length"),
    (
        (("field", "lattice", "vertices"), 65),
        (("field", "lattice", "edges"), 129),
        (("field", "edge_values"), 129),
        (("field", "group", "multiplication"), 25),
        (("connector_holonomy", "group", "inverse"), 25),
        (("transported_loop", "steps"), 257),
    ),
)
def test_transport_rejects_oversized_axes_before_copying(
    wrapped: bool,
    path: tuple[str | int, ...],
    length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    axis = _at(result, path)
    assert isinstance(axis, tuple)
    forged = _replace(result, path, (axis[0],) * length)

    def unexpected_dump(*_args: object, **_kwargs: object) -> object:
        pytest.fail("oversized retained axes must be rejected before serialization")

    monkeypatch.setattr(StrictModel, "model_dump", unexpected_dump)
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


def test_transport_typed_mapping_and_wire_decoding_preserve_reverse_products() -> None:
    result = _result()
    assert result.source_holonomy.index == 2
    assert result.connector_holonomy.index == 2
    assert result.transported_holonomy.index == 2
    assert _decode(result, True) == result
    assert _decode(result, False) == result
    assert type(result).model_validate_json(result.model_dump_json()) == result
