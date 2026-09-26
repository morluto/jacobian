from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugeLoopFamilyHolonomies,
    GaugeLoopFamilyRequest,
    GaugePathStep,
    OrientedGaugePath,
    PermutationLabel,
    loop_family_holonomies,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS,
    MAX_GAUGE_LOOP_FAMILY_SIZE,
    MAX_GAUGE_LOOP_FAMILY_STEPS,
)
from jacobian.math.gauge._tools import TOOLS


def _field() -> GaugeField:
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
        "bc": (0, 2, 1),
        "ca": (0, 1, 2),
    }
    return GaugeField(
        lattice=lattice,
        degree=3,
        edge_labels=tuple(
            GaugeFieldEdgeLabel(
                edge_id=edge_id,
                label=PermutationLabel(degree=3, image=image),
            )
            for edge_id, image in labels.items()
        ),
    )


def _path(*steps: tuple[str, bool], basepoint: str | None = None) -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=tuple(
            GaugePathStep(edge_id=edge, forward=forward) for edge, forward in steps
        ),
        basepoint=basepoint,
    )


def _direct_oracle(field: GaugeField, path: OrientedGaugePath) -> tuple[int, ...]:
    """Apply every edge action pointwise, independently of the kernel product."""
    labels = {entry.edge_id: entry.label.image for entry in field.edge_labels}
    action = list(range(field.degree))
    for step in path.steps:
        edge_action = labels[step.edge_id]
        if not step.forward:
            inverse = [0] * field.degree
            for point, image in enumerate(edge_action):
                inverse[image] = point
            edge_action = tuple(inverse)
        # The traversal convention applies the accumulated action first, then
        # applies this edge action to each current image.
        action = [edge_action[image] for image in action]
    return tuple(action)


def _self_loop_field() -> GaugeField:
    return GaugeField(
        lattice=GaugeLattice(
            vertices=("v",),
            edges=(GaugeEdge(edge_id="loop", tail="v", head="v"),),
        ),
        degree=2,
        edge_labels=(
            GaugeFieldEdgeLabel(
                edge_id="loop", label=PermutationLabel(degree=2, image=(1, 0))
            ),
        ),
    )


def test_noncommuting_loop_family_matches_direct_point_action_and_round_trips() -> None:
    field = _field()
    loops = (
        _path(("ab", True), ("bc", True), ("ca", True)),
        _path(("bc", True), ("ca", True), ("ab", True)),
        _path(("ca", False), ("bc", False), ("ab", False)),
        _path(basepoint="b"),
    )

    result = loop_family_holonomies(field, loops)

    assert result.field == field
    assert tuple(item.basepoint for item in result.loops) == ("a", "b", "a", "b")
    assert tuple(item.path for item in result.loops) == loops
    assert tuple(item.holonomy.image for item in result.loops) == tuple(
        _direct_oracle(field, path) for path in loops
    )
    assert loop_family_holonomies(field, ()).loops == ()
    # Cyclic rotations need not be literally equal for noncommuting links.
    assert result.loops[0].holonomy.image != result.loops[1].holonomy.image

    decoded = GaugeLoopFamilyHolonomies.model_validate_json(result.model_dump_json())
    assert decoded == result
    decoded_data = decoded.model_dump()
    assert "field" in decoded_data
    assert all("field" not in item for item in decoded_data["loops"])


def test_operation_is_published_and_example_runs() -> None:
    assert {tool.operation_id for tool in TOOLS} >= {
        "lattice_gauge.loop_family.holonomies.compute"
    }
    example = next(
        tool.examples[0].input
        for tool in TOOLS
        if tool.operation_id == "lattice_gauge.loop_family.holonomies.compute"
    )
    request = GaugeLoopFamilyRequest.model_validate(example)
    assert loop_family_holonomies(request.field, request.loops).loops


@pytest.mark.parametrize(
    "path",
    (
        _path(("ab", True)),
        _path(("ab", True), ("ca", True)),
        _path(("missing", True), ("missing", False)),
        _path(("ab", True), ("ab", False), basepoint="c"),
    ),
)
def test_open_foreign_or_wrong_basepoint_paths_are_rejected(
    path: OrientedGaugePath,
) -> None:
    with pytest.raises(OperationDomainValidationError):
        loop_family_holonomies(_field(), (path,))


def test_loop_count_and_aggregate_step_bounds_are_admitted_before_products() -> None:
    field = _self_loop_field()
    empty = _path(basepoint="v")
    accepted = loop_family_holonomies(
        field, tuple(empty for _ in range(MAX_GAUGE_LOOP_FAMILY_SIZE))
    )
    assert len(accepted.loops) == MAX_GAUGE_LOOP_FAMILY_SIZE
    with pytest.raises(OperationResourceAdmissionError) as count_error:
        loop_family_holonomies(
            field, tuple(empty for _ in range(MAX_GAUGE_LOOP_FAMILY_SIZE + 1))
        )
    assert count_error.value.errors()[0]["type"] == (
        "lattice_gauge.loop_family.count_over_envelope"
    )

    full_loop = _path(*(("loop", True),) * 256)
    exact_step_family = tuple(
        full_loop for _ in range(MAX_GAUGE_LOOP_FAMILY_STEPS // 256)
    )
    accepted_steps = loop_family_holonomies(field, exact_step_family)
    assert len(accepted_steps.loops) == 16
    above_step_family = (*exact_step_family, _path(("loop", True)))
    with pytest.raises(OperationResourceAdmissionError) as step_error:
        loop_family_holonomies(field, above_step_family)
    assert step_error.value.errors()[0]["type"] == (
        "lattice_gauge.loop_family.steps_over_envelope"
    )


def test_raw_catalog_request_preflights_family_and_aggregate_lengths() -> None:
    field = _self_loop_field().model_dump(mode="json")
    empty_path = {"steps": [], "basepoint": "v"}
    with pytest.raises(ValueError, match="loop_family_count"):
        GaugeLoopFamilyRequest.model_validate(
            {
                "field": field,
                "loops": [empty_path] * (MAX_GAUGE_LOOP_FAMILY_SIZE + 1),
            }
        )

    long_loop = {"steps": [{"edge_id": "loop", "forward": True}] * 256}
    over = [long_loop] * 16 + [{"steps": [{"edge_id": "loop", "forward": True}]}]
    assert len(over[0]["steps"]) * 16 + len(over[-1]["steps"]) == (
        MAX_GAUGE_LOOP_FAMILY_STEPS + 1
    )
    with pytest.raises(ValueError, match="loop_family_steps"):
        GaugeLoopFamilyRequest.model_validate({"field": field, "loops": over})

    forged_long_path = OrientedGaugePath.model_construct(
        steps=tuple(GaugePathStep(edge_id="loop", forward=True) for _ in range(257)),
        basepoint="v",
    )
    with pytest.raises(ValueError, match="loop_family_path_length"):
        GaugeLoopFamilyRequest.model_validate(
            {"field": _self_loop_field(), "loops": (forged_long_path,)}
        )


def test_output_envelope_is_checked_before_any_permutation_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.gauge import operations

    field = _field()
    path = _path(("ab", True), ("bc", True), ("ca", True))
    monkeypatch.setattr(operations, "MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS", 1)

    def unexpected_product(*_args: object) -> tuple[int, ...]:
        raise AssertionError("the product kernel ran before output admission")

    monkeypatch.setattr(operations, "_compose", unexpected_product)
    with pytest.raises(OperationResourceAdmissionError) as raised:
        loop_family_holonomies(field, (path,))
    assert raised.value.errors()[0]["type"] == (
        "lattice_gauge.loop_family.output_over_envelope"
    )
    assert MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS > 1


def test_work_envelope_is_checked_before_path_replay_or_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.gauge import operations

    monkeypatch.setattr(operations, "MAX_GAUGE_LOOP_FAMILY_WORK", 1)

    def unexpected_path(*_args: object) -> str:
        raise AssertionError("path replay ran before work admission")

    monkeypatch.setattr(operations, "_admit_closed_loop", unexpected_path)
    with pytest.raises(OperationResourceAdmissionError) as raised:
        loop_family_holonomies(
            _field(), (_path(("ab", True), ("bc", True), ("ca", True)),)
        )
    assert raised.value.errors()[0]["type"] == (
        "lattice_gauge.loop_family.work_over_envelope"
    )
