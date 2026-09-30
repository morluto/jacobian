"""Curvature decoding bounds every retained carrier before traversal or copying."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from jacobian._models import StrictModel
from jacobian.math.gauge import (
    FiniteGroupGaugeEdgeLabel,
    FiniteGroupGaugeFace,
    FiniteGroupGaugeField,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    _models,
    construct_finite_group_gauge_complex,
    finite_group_gauge_curvature,
)
from jacobian.math.gauge._models import FiniteGroupGaugeCurvatureResult
from jacobian.math.groups._table_models import FiniteGroupTable, FiniteGroupTableElement


def _result() -> FiniteGroupGaugeCurvatureResult:
    group = FiniteGroupTable(
        multiplication=((0, 1, 2), (1, 2, 0), (2, 0, 1)),
        identity=0,
        inverse=(0, 2, 1),
    )
    lattice = GaugeLattice(
        vertices=("a",), edges=(GaugeEdge(edge_id="e", tail="a", head="a"),)
    )
    field = FiniteGroupGaugeField(
        lattice=lattice,
        group=group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="e", value=FiniteGroupTableElement(group=group, index=1)
            ),
        ),
    )
    complex_value = construct_finite_group_gauge_complex(
        lattice,
        group,
        (
            FiniteGroupGaugeFace(
                face_id="identity", boundary=OrientedGaugePath(steps=(), basepoint="a")
            ),
            FiniteGroupGaugeFace(
                face_id="reverse",
                boundary=OrientedGaugePath(
                    steps=(GaugePathStep(edge_id="e", forward=False),), basepoint="a"
                ),
            ),
        ),
    )
    return finite_group_gauge_curvature(complex_value, field)


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


def _decode(value: object, wrapped: bool) -> FiniteGroupGaugeCurvatureResult:
    assert isinstance(value, FiniteGroupGaugeCurvatureResult)
    return FiniteGroupGaugeCurvatureResult.model_validate(
        value if wrapped else dict(value.__dict__)
    )


_COMPLEX = ("complex",)
_FIELD = ("field",)
_FACE = (*_COMPLEX, "faces", 1)
_PATH = (*_FACE, "boundary")
_EDGE_VALUE = (*_FIELD, "edge_values", 0, "value")
_ROW = ("face_values", 1)
_ROW_VALUE = (*_ROW, "value")
_GROUP_PARENTS = (
    (*_COMPLEX, "group"),
    (*_FIELD, "group"),
    (*_EDGE_VALUE, "group"),
    (*_ROW_VALUE, "group"),
)
_REQUIRED_FIELDS = (
    ((), ("complex", "field", "face_values", "flat")),
    (_COMPLEX, ("lattice", "group", "faces")),
    (_FIELD, ("lattice", "group", "edge_values")),
    ((*_COMPLEX, "lattice"), ("vertices", "edges")),
    ((*_FIELD, "lattice"), ("vertices", "edges")),
    ((*_COMPLEX, "lattice", "edges", 0), ("edge_id", "tail", "head")),
    ((*_FIELD, "lattice", "edges", 0), ("edge_id", "tail", "head")),
    (_FACE, ("face_id", "boundary")),
    (_PATH, ("steps",)),
    ((*_PATH, "steps", 0), ("edge_id", "forward")),
    ((*_FIELD, "edge_values", 0), ("edge_id", "value")),
    (_EDGE_VALUE, ("group", "index")),
    (_ROW, ("face_id", "value")),
    (_ROW_VALUE, ("group", "index")),
    *((parent, ("multiplication", "identity", "inverse")) for parent in _GROUP_PARENTS),
)


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "path,field_name",
    [(path, field) for path, fields in _REQUIRED_FIELDS for field in fields],
)
def test_curvature_rejects_missing_native_fields(
    wrapped: bool, path: tuple[str | int, ...], field_name: str
) -> None:
    result = _result()
    carrier = _at(result, path)
    assert isinstance(carrier, StrictModel)
    fields = dict(carrier.__dict__)
    del fields[field_name]
    forged = _replace(result, path, type(carrier).model_construct(**fields))
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


_MALFORMED_LEAVES = (
    ("flat",),
    (*_COMPLEX, "lattice", "vertices", 0),
    (*_FIELD, "lattice", "edges", 0, "tail"),
    (*_FACE, "face_id"),
    (*_PATH, "basepoint"),
    (*_PATH, "steps", 0, "edge_id"),
    (*_PATH, "steps", 0, "forward"),
    (*_FIELD, "edge_values", 0, "edge_id"),
    (*_EDGE_VALUE, "index"),
    (*_ROW, "face_id"),
    (*_ROW_VALUE, "index"),
    *(
        path
        for parent in _GROUP_PARENTS
        for path in (
            (*parent, "multiplication", 0, 0),
            (*parent, "identity"),
            (*parent, "inverse", 0),
        )
    ),
)


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize("path", _MALFORMED_LEAVES)
def test_curvature_rejects_container_leaves_before_copying(
    wrapped: bool, path: tuple[str | int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = _replace(_result(), path, ["invalid"])
    copies: list[str] = []

    def unexpected_copy(*args: object, **kwargs: object) -> object:
        copies.append("copy")
        raise AssertionError("malformed curvature must be refused before copying")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    monkeypatch.setattr(StrictModel, "model_dump", unexpected_copy)
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)
    assert copies == []


_BOUNDED_AXES = (
    ((*_COMPLEX, "lattice", "vertices"), 65),
    ((*_COMPLEX, "lattice", "edges"), 129),
    ((*_FIELD, "lattice", "vertices"), 65),
    ((*_FIELD, "lattice", "edges"), 129),
    ((*_COMPLEX, "faces"), 129),
    ((*_FIELD, "edge_values"), 129),
    (("face_values",), 129),
    ((*_PATH, "steps"), 257),
    *(((*parent, "multiplication"), 25) for parent in _GROUP_PARENTS),
    *(((*parent, "multiplication", 0), 25) for parent in _GROUP_PARENTS),
    *(((*parent, "inverse"), 25) for parent in _GROUP_PARENTS),
)


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize("path,length", _BOUNDED_AXES)
def test_curvature_bounds_nested_axes_before_copying(
    wrapped: bool,
    path: tuple[str | int, ...],
    length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    axis = _at(result, path)
    assert isinstance(axis, tuple)
    forged = _replace(result, path, (axis[0],) * length)
    copies: list[str] = []

    def unexpected_copy(*args: object, **kwargs: object) -> object:
        copies.append("copy")
        raise AssertionError("oversized curvature must be refused before copying")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    monkeypatch.setattr(StrictModel, "model_dump", unexpected_copy)
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)
    assert copies == []


@pytest.mark.parametrize("wrapped", (False, True))
def test_curvature_does_not_invoke_untrusted_container_hooks(wrapped: bool) -> None:
    visits: list[str] = []

    class UnvisitedRows(tuple[object, ...]):
        def __len__(self) -> int:
            visits.append("length")
            raise AssertionError("noncanonical rows must be rejected before len")

        def __iter__(self) -> Iterator[object]:
            visits.append("iteration")
            raise AssertionError("noncanonical rows must be rejected before iteration")

    forged = _replace(_result(), ("face_values",), UnvisitedRows())
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)
    assert visits == []


@pytest.mark.parametrize("wrapped", (False, True))
def test_curvature_bounds_aggregate_face_steps_before_copying(
    wrapped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result()
    face = result.complex.faces[1]
    face = face.model_copy(
        update={
            "boundary": face.boundary.model_copy(
                update={"steps": face.boundary.steps * 256}
            )
        }
    )
    forged = _replace(result, (*_COMPLEX, "faces"), (face,) * 17)
    copies: list[str] = []

    def unexpected_copy(*args: object, **kwargs: object) -> object:
        copies.append("copy")
        raise AssertionError("aggregate steps must be bounded before copying")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    monkeypatch.setattr(StrictModel, "model_dump", unexpected_copy)
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)
    assert copies == []


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize("path", (_FACE, _PATH, (*_PATH, "steps", 0), _ROW_VALUE))
def test_curvature_rejects_raw_children_inside_native_models(
    wrapped: bool, path: tuple[str | int, ...]
) -> None:
    result = _result()
    child = _at(result, path)
    assert isinstance(child, StrictModel)
    forged = _replace(result, path, child.model_dump())
    with pytest.raises(ValidationError):
        _decode(forged, wrapped)


@pytest.mark.parametrize("encoding", ("native", "mixed", "mapping", "json"))
def test_curvature_preserves_valid_native_and_json_controls(encoding: str) -> None:
    result = _result()
    if encoding == "json":
        restored = FiniteGroupGaugeCurvatureResult.model_validate_json(
            result.model_dump_json()
        )
    elif encoding == "mapping":
        restored = FiniteGroupGaugeCurvatureResult.model_validate(result.model_dump())
    else:
        restored = _decode(result, encoding == "native")
    assert restored == result
    assert tuple(row.value.index for row in restored.face_values) == (0, 2)
    assert restored.flat is False


@pytest.mark.parametrize("wrapped", (False, True))
def test_curvature_decoding_does_not_replay_face_products(wrapped: bool) -> None:
    result = _result()
    changed = _replace(result, (*_ROW_VALUE, "index"), 0)
    changed = _replace(changed, ("flat",), True)
    restored = _decode(changed, wrapped)
    assert tuple(row.value.index for row in restored.face_values) == (0, 0)
    assert restored.flat is True


def test_curvature_preserves_the_full_face_and_aggregate_step_bound() -> None:
    sample = _result()
    boundary = sample.complex.faces[1].boundary
    boundary = boundary.model_copy(update={"steps": boundary.steps * 32})
    complex_value = construct_finite_group_gauge_complex(
        sample.complex.lattice,
        sample.complex.group,
        tuple(
            FiniteGroupGaugeFace(face_id=f"f{index:03d}", boundary=boundary)
            for index in range(128)
        ),
    )
    result = finite_group_gauge_curvature(complex_value, sample.field)
    restored = FiniteGroupGaugeCurvatureResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    assert len(restored.face_values) == 128
    assert sum(len(face.boundary.steps) for face in restored.complex.faces) == 4096
    # Each reverse edge contributes 2 in C3, and 32 * 2 is 1 modulo 3.
    assert tuple(row.value.index for row in restored.face_values) == (1,) * 128
    assert restored.flat is False
