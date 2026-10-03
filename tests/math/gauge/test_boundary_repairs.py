"""Trust-boundary regressions for the finite-group gauge owner.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
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
    finite_group_gauge_basepoint_transport,
    finite_group_gauge_curvature,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_LABEL_LENGTH,
    FiniteGroupGaugeBasepointTransportResult,
    FiniteGroupGaugeCurvatureResult,
)
from jacobian.math.groups._table_models import (
    FiniteGroupTable,
    FiniteGroupTableElement,
    FiniteGroupTableRequest,
)
from jacobian.math.groups._tools import construct_finite_group_table


def _table(order: int = 3) -> FiniteGroupTable:
    """The cyclic group of the given order, as an admitted multiplication table."""
    rows = tuple(
        tuple((row + column) % order for column in range(order)) for row in range(order)
    )
    return construct_finite_group_table(
        FiniteGroupTableRequest(multiplication=rows, identity=0)
    ).group


def _square_lattice() -> GaugeLattice:
    return GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )


def _square_field() -> FiniteGroupGaugeField:
    group = _table()
    return FiniteGroupGaugeField(
        lattice=_square_lattice(),
        group=group,
        edge_values=tuple(
            FiniteGroupGaugeEdgeLabel(
                edge_id=edge_id, value=FiniteGroupTableElement(group=group, index=1)
            )
            for edge_id in ("ab", "bc", "ca")
        ),
    )


def _complex() -> FiniteGroupGaugeComplex:
    lattice = _square_lattice()
    group = _table()
    return construct_finite_group_gauge_complex(
        lattice,
        group,
        (
            FiniteGroupGaugeFace(
                face_id="loop",
                boundary=OrientedGaugePath(
                    steps=(
                        GaugePathStep(edge_id="ab", forward=True),
                        GaugePathStep(edge_id="bc", forward=True),
                        GaugePathStep(edge_id="ca", forward=True),
                    ),
                    basepoint="a",
                ),
            ),
        ),
    )


def _path(*edges: str, basepoint: str | None = None) -> OrientedGaugePath:
    """An oriented walk over the square lattice, based at its first vertex."""
    return OrientedGaugePath(
        steps=tuple(GaugePathStep(edge_id=edge, forward=True) for edge in edges),
        basepoint=basepoint or "a",
    )


def test_basepoint_transport_rejects_a_bypass_constructed_field() -> None:
    """A missing parent must not leak AttributeError from the guard itself."""
    field = _square_field()
    without_group = FiniteGroupGaugeField.model_construct(
        lattice=field.lattice, group=None, edge_values=()
    )

    with pytest.raises(OperationDomainValidationError) as error:
        finite_group_gauge_basepoint_transport(
            without_group, _path("ab", "bc", "ca"), _path()
        )

    assert error.value.errors()[0]["type"] == (
        "lattice_gauge.finite_group.basepoint_request_shape"
    )


def test_curvature_rejects_bypass_constructed_parents() -> None:
    """Every parent read must be guarded, including the lattice comparison."""
    field = _square_field()
    complex_value = _complex()

    without_group = FiniteGroupGaugeComplex.model_construct(
        lattice=complex_value.lattice, group=None, faces=()
    )
    without_lattice = FiniteGroupGaugeField.model_construct(
        lattice=None, group=field.group, edge_values=field.edge_values
    )
    other_lattice = FiniteGroupGaugeField(
        lattice=GaugeLattice(
            vertices=("p", "q"), edges=(GaugeEdge(edge_id="pq", tail="p", head="q"),)
        ),
        group=field.group,
        edge_values=(
            FiniteGroupGaugeEdgeLabel(
                edge_id="pq", value=FiniteGroupTableElement(group=field.group, index=1)
            ),
        ),
    )

    for complex_candidate, field_candidate, code in (
        (
            without_group,
            field,
            "lattice_gauge.finite_group.curvature_request_shape",
        ),
        (
            complex_value,
            without_lattice,
            "lattice_gauge.finite_group.curvature_request_shape",
        ),
        (
            complex_value,
            other_lattice,
            "lattice_gauge.finite_group.curvature_parent_mismatch",
        ),
    ):
        with pytest.raises(OperationDomainValidationError) as error:
            finite_group_gauge_curvature(complex_candidate, field_candidate)
        assert error.value.errors()[0]["type"] == code


def test_curvature_result_binds_rows_to_its_complex_and_flatness() -> None:
    """A decoded curvature result must describe its own retained parents."""
    group = _table()
    complex_value = _complex()
    field = _square_field()
    identity = FiniteGroupTableElement(group=group, index=group.identity)
    base = {
        "complex": complex_value.model_dump(),
        "field": field.model_dump(),
        "flat": True,
    }
    row = {"face_id": "loop", "value": identity.model_dump()}

    FiniteGroupGaugeCurvatureResult.model_validate({**base, "face_values": [row]})

    for mutation, reason in (
        ({"face_id": "not-a-face"}, "curvature_face_binding"),
        (
            {
                "face_id": "loop",
                "value": FiniteGroupTableElement(group=_table(2), index=0).model_dump(),
            },
            "curvature_parent",
        ),
        (
            {
                "face_id": "loop",
                "value": FiniteGroupTableElement(group=group, index=1).model_dump(),
            },
            "curvature_flatness",
        ),
    ):
        with pytest.raises(ValidationError) as error:
            FiniteGroupGaugeCurvatureResult.model_validate(
                {**base, "face_values": [{**row, **mutation}]}
            )
        assert reason in str(error.value), reason

    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeCurvatureResult.model_validate(
            {**base, "face_values": [row, row]}
        )
    assert "curvature_face_binding" in str(error.value)


def test_curvature_result_accepts_a_genuine_computed_result() -> None:
    result = finite_group_gauge_curvature(_complex(), _square_field())

    revived = FiniteGroupGaugeCurvatureResult.model_validate(result.model_dump())

    assert revived == result
    assert revived.flat is True


def test_basepoint_transport_result_binds_the_conjugated_walk() -> None:
    """A decoded transport must retain its parent-bound conjugated walk."""
    field = _square_field()
    loop = _path("ab", "bc", "ca")
    connector = _path("ab")
    genuine = finite_group_gauge_basepoint_transport(field, loop, connector)
    payload = genuine.model_dump()

    FiniteGroupGaugeBasepointTransportResult.model_validate(payload)

    open_loop = _path("ab", "bc")
    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeBasepointTransportResult.model_validate(
            {**payload, "loop": open_loop.model_dump()}
        )
    assert "basepoint_transport_loop" in str(error.value)

    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeBasepointTransportResult.model_validate(
            {
                **payload,
                "transported_loop": _path("bc", "ca", "ab", basepoint="b").model_dump(),
            }
        )
    assert "basepoint_transport_walk" in str(error.value)


def test_basepoint_transport_result_accepts_the_degenerate_identity_case() -> None:
    """An empty loop and empty connector stay exactly the identity."""
    field = _square_field()
    genuine = finite_group_gauge_basepoint_transport(
        field, _path("ab", "bc", "ca"), _path()
    )

    revived = FiniteGroupGaugeBasepointTransportResult.model_validate(
        genuine.model_dump()
    )

    assert revived == genuine


def test_malformed_nested_arrays_are_refused_before_canonicalization() -> None:
    """One malformed label must not carry an unbounded inner array.

    ``canonicalize_json_containers`` copies the whole container tree before
    Pydantic rejects a non-string label, so an outer count alone leaves
    unbounded parsing and allocation.
    """
    request = {
        "lattice": {"vertices": [["x"] * 1_000_000], "edges": []},
        "group": _table().model_dump(),
        "faces": [],
    }

    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeComplex.model_validate(request)
    assert "complex_vertex_shape" in str(error.value)

    for lattice in (
        {
            "vertices": [],
            "edges": [{"edge_id": ["x"] * 1_000_000, "tail": "a", "head": "b"}],
        },
        {
            "vertices": [],
            "edges": [{"edge_id": "ab", "tail": ["x"] * 1_000_000, "head": "b"}],
        },
    ):
        with pytest.raises(ValidationError) as error:
            FiniteGroupGaugeComplex.model_validate(
                {"lattice": lattice, "group": _table().model_dump(), "faces": []}
            )
        assert "complex_edge_shape" in str(error.value)

    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeComplex.model_validate(
            {
                "lattice": _square_lattice().model_dump(),
                "group": _table().model_dump(),
                "faces": [
                    {
                        "face_id": ["x"] * 1_000_000,
                        "boundary": {"steps": [], "basepoint": "a"},
                    }
                ],
            }
        )
    assert "complex_face_shape" in str(error.value)

    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeComplex.model_validate(
            {
                "lattice": _square_lattice().model_dump(),
                "group": _table().model_dump(),
                "faces": [
                    {
                        "face_id": "f",
                        "boundary": {
                            "steps": [{"edge_id": "ab", "forward": ["x"] * 1_000_000}],
                            "basepoint": "a",
                        },
                    }
                ],
            }
        )
    assert "complex_step_shape" in str(error.value)


def test_maximum_length_labels_are_still_accepted() -> None:
    """The nested bounds are per-value domain limits, not a size heuristic."""
    long_id = "e" * MAX_GAUGE_LABEL_LENGTH
    lattice = GaugeLattice(
        vertices=("a", "b"),
        edges=(GaugeEdge(edge_id=long_id, tail="a", head="b"),),
    )

    complex_value = construct_finite_group_gauge_complex(
        lattice,
        _table(),
        (
            FiniteGroupGaugeFace(
                face_id=long_id,
                boundary=OrientedGaugePath(
                    steps=(
                        GaugePathStep(edge_id=long_id, forward=True),
                        GaugePathStep(edge_id=long_id, forward=False),
                    ),
                    basepoint="a",
                ),
            ),
        ),
    )

    assert complex_value.faces[0].face_id == long_id


def test_deferred_gauge_projections_are_native_only() -> None:
    """The three deferred operations must not be published.

    A batch of already-published holonomies, a projection of one holonomy, and
    a transform that the permutation parent already publishes are useful
    projections. They stay native exports, not catalog operations.
    """
    from jacobian.math.gauge._tools import TOOLS

    published = {tool.operation_id for tool in TOOLS}
    for operation_id in (
        "lattice_gauge.loop_family.holonomies.compute",
        "lattice_gauge.holonomy.conjugacy_profile.compute",
        "lattice_gauge.finite_group.gauge_transform.compute",
    ):
        assert operation_id not in published, operation_id
    # The native functions remain available to Python callers.
    from jacobian.math.gauge import (
        finite_group_gauge_transform,
        finite_group_holonomy_conjugacy_profile,
        loop_family_holonomies,
    )

    assert callable(loop_family_holonomies)
    assert callable(finite_group_holonomy_conjugacy_profile)
    assert callable(finite_group_gauge_transform)


@pytest.mark.parametrize("step", (1, None, "ab", [], True))
def test_complex_rejects_non_mapping_steps_before_canonicalization(
    step: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jacobian.math.gauge import _models

    payload = _complex().model_dump()
    payload["faces"][0]["boundary"]["steps"] = [step]

    def unexpected_copy(value: object) -> object:
        raise AssertionError("invalid steps must be refused before copying")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    with pytest.raises(ValidationError) as error:
        FiniteGroupGaugeComplex.model_validate(payload)
    assert error.value.errors()[0]["type"] == "lattice_gauge.complex_step_shape"


@pytest.mark.parametrize("path_name", ("loop", "connector"))
def test_transport_rejects_forged_authored_path_basepoints(path_name: str) -> None:
    result = finite_group_gauge_basepoint_transport(
        _square_field(), _path("ab", "bc", "ca"), _path("ab")
    )
    payload = result.model_dump()
    payload[path_name]["basepoint"] = "b"
    with pytest.raises(ValidationError) as exc_info:
        FiniteGroupGaugeBasepointTransportResult.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"] == "lattice_gauge.basepoint_transport_walk"
    )


def test_empty_transport_cannot_change_its_basepoint() -> None:
    result = finite_group_gauge_basepoint_transport(_square_field(), _path(), _path())
    assert type(result).model_validate_json(result.model_dump_json()) == result
    payload = result.model_dump()
    payload["target_basepoint"] = "b"
    payload["transported_loop"]["basepoint"] = "b"
    with pytest.raises(ValidationError) as exc_info:
        type(result).model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "lattice_gauge.basepoint_transport_connector"
    )


def test_curvature_rejects_a_field_on_a_different_lattice() -> None:
    result = finite_group_gauge_curvature(_complex(), _square_field())
    other_field = FiniteGroupGaugeField(
        lattice=GaugeLattice(
            vertices=("a", "b", "c", "d"), edges=result.field.lattice.edges
        ),
        group=result.field.group,
        edge_values=result.field.edge_values,
    )
    payload = result.model_dump()
    payload["field"] = other_field.model_dump()
    with pytest.raises(ValidationError) as exc_info:
        type(result).model_validate(payload)
    assert exc_info.value.errors()[0]["type"] == "lattice_gauge.curvature_parent"


def test_transport_decode_retains_values_without_replaying_holonomy() -> None:
    result = finite_group_gauge_basepoint_transport(
        _square_field(), _path("ab", "bc", "ca"), _path("ab")
    )
    assert result.transported_holonomy.index == 0
    payload = result.model_dump()
    payload["transported_holonomy"]["index"] = 2
    # Parsing checks the table-bound representation. The producing kernel owns
    # the path product; ordinary result decoding does not authenticate it.
    decoded = type(result).model_validate(payload)
    assert decoded.transported_holonomy.index == 2
    assert type(result).model_validate_json(decoded.model_dump_json()) == decoded


@pytest.mark.parametrize("path_name", ("loop", "connector", "transported_loop"))
@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "defect",
    (
        "wrong_step",
        "missing_edge",
        "missing_forward",
        "nested_edge",
        "missing_steps",
        "list_steps",
        "nested_basepoint",
    ),
)
def test_transport_rejects_forged_native_path_shapes(
    path_name: str, wrapped: bool, defect: str
) -> None:
    result = finite_group_gauge_basepoint_transport(
        _square_field(), _path("ab", "bc", "ca"), _path("ab")
    )
    path = getattr(result, path_name)
    update: dict[str, object]
    if defect == "wrong_step":
        update = {"steps": (1,)}
    elif defect == "missing_edge":
        update = {"steps": (GaugePathStep.model_construct(forward=True),)}
    elif defect == "missing_forward":
        update = {"steps": (GaugePathStep.model_construct(edge_id="ab"),)}
    elif defect == "nested_edge":
        update = {
            "steps": (GaugePathStep.model_construct(edge_id=["ab"], forward=True),)
        }
    elif defect == "missing_steps":
        update = {"steps": None}
    elif defect == "list_steps":
        update = {"steps": list(path.steps)}
    else:
        update = {"basepoint": ["a"]}
    forged = path.model_copy(update=update)
    payload = (
        result.model_copy(update={path_name: forged})
        if wrapped
        else {**result.model_dump(), path_name: forged}
    )
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_native_path_container_hooks_are_not_invoked() -> None:
    class UnvisitedSteps(tuple[GaugePathStep, ...]):
        def __len__(self) -> int:
            raise AssertionError("noncanonical path containers must not be traversed")

    path = OrientedGaugePath.model_construct(steps=UnvisitedSteps(), basepoint="a")
    with pytest.raises(ValidationError) as exc_info:
        OrientedGaugePath.model_validate(path)
    assert exc_info.value.errors()[0]["type"] == "lattice_gauge.path_shape"
