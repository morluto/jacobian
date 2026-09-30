"""Decoded loop families retain their authored lattice basepoints."""

import pytest
from pydantic import ValidationError

from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
    PermutationLabel,
    loop_family_holonomies,
    path_holonomy,
)


def _field() -> GaugeField:
    return GaugeField(
        lattice=GaugeLattice(
            vertices=("a", "b"),
            edges=(GaugeEdge(edge_id="ab", tail="a", head="b"),),
        ),
        degree=1,
        edge_labels=(
            GaugeFieldEdgeLabel(
                edge_id="ab", label=PermutationLabel(degree=1, image=(0,))
            ),
        ),
    )


def _loop(empty: bool) -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=()
        if empty
        else (
            GaugePathStep(edge_id="ab", forward=True),
            GaugePathStep(edge_id="ab", forward=False),
        ),
        basepoint="a",
    )


@pytest.mark.parametrize(
    "empty,path_basepoint,entry_basepoint",
    (
        (False, "b", "a"),
        (True, "foreign", "foreign"),
        (True, "a", "b"),
        (True, "b", "a"),
    ),
)
def test_loop_family_rejects_inconsistent_or_foreign_basepoints(
    empty: bool, path_basepoint: str, entry_basepoint: str
) -> None:
    result = loop_family_holonomies(_field(), (_loop(empty),))
    payload = result.model_dump()
    payload["loops"][0]["path"]["basepoint"] = path_basepoint
    payload["loops"][0]["basepoint"] = entry_basepoint
    with pytest.raises(ValidationError, match="loop_family_path"):
        type(result).model_validate(payload)


@pytest.mark.parametrize("empty", (False, True))
def test_loop_family_round_trip_preserves_based_identity_paths(empty: bool) -> None:
    result = loop_family_holonomies(_field(), (_loop(empty),))
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    entry = restored.loops[0]
    assert entry.basepoint == entry.path.basepoint == "a"
    assert path_holonomy(restored.field, entry.path).holonomy == entry.holonomy


def test_empty_loop_family_round_trips() -> None:
    result = loop_family_holonomies(_field(), ())
    assert type(result).model_validate_json(result.model_dump_json()) == result
