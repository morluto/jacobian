"""Loop-family native carriers are bounded before traversal or recursive copying."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugeLoopFamilyHolonomies,
    GaugeLoopHolonomy,
    GaugePathStep,
    OrientedGaugePath,
    PermutationLabel,
    loop_family_holonomies,
    path_holonomy,
)


def _family() -> GaugeLoopFamilyHolonomies:
    field = GaugeField(
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
    return loop_family_holonomies(field, (OrientedGaugePath(steps=(), basepoint="a"),))


def _payload(
    family: GaugeLoopFamilyHolonomies, wrapped: bool, **updates: object
) -> object:
    if wrapped:
        return family.model_copy(update=updates)
    return {"field": family.field, "loops": family.loops, **updates}


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "defect",
    ("wrong_entry", "missing_path", "none_path", "missing_steps", "none_steps"),
)
def test_loop_family_rejects_malformed_entries_before_step_counting(
    wrapped: bool, defect: str
) -> None:
    family = _family()
    entry = family.loops[0]
    if defect == "wrong_entry":
        forged: object = 1
    elif defect == "missing_path":
        forged = GaugeLoopHolonomy.model_construct(
            basepoint=entry.basepoint, holonomy=entry.holonomy
        )
    elif defect == "none_path":
        forged = entry.model_copy(update={"path": None})
    elif defect == "missing_steps":
        forged = entry.model_copy(
            update={"path": OrientedGaugePath.model_construct(basepoint="a")}
        )
    else:
        forged = entry.model_copy(
            update={"path": entry.path.model_copy(update={"steps": None})}
        )
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(
            _payload(family, wrapped, loops=(forged,))
        )


@pytest.mark.parametrize("wrapped", (False, True))
def test_loop_family_does_not_invoke_noncanonical_step_container_hooks(
    wrapped: bool,
) -> None:
    visits: list[str] = []

    class UnvisitedSteps(tuple[GaugePathStep, ...]):
        def __len__(self) -> int:
            visits.append("length")
            raise AssertionError("noncanonical steps must be rejected before len")

    family = _family()
    forged = family.loops[0].model_copy(
        update={
            "path": OrientedGaugePath.model_construct(
                steps=UnvisitedSteps(), basepoint="a"
            )
        }
    )
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(
            _payload(family, wrapped, loops=(forged,))
        )
    assert visits == []


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize("defect", ("missing_label", "wrong_label"))
def test_loop_family_rejects_malformed_source_entries(
    wrapped: bool, defect: str
) -> None:
    family = _family()
    bad_label = (
        GaugeFieldEdgeLabel.model_construct(edge_id="ab")
        if defect == "missing_label"
        else None
    )
    field = family.field.model_copy(update={"edge_labels": (bad_label,)})
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(_payload(family, wrapped, field=field))


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "defect", ("vertices", "edges", "edge_labels", "nested_image", "long_image")
)
def test_loop_family_bounds_source_before_recursive_dump(
    wrapped: bool, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dump sentinel proves malformed native contents are refused before copying."""
    family = _family()
    field = family.field
    if defect == "vertices":
        lattice = field.lattice.model_copy(
            update={"vertices": tuple(f"v{i}" for i in range(65))}
        )
        field = field.model_copy(update={"lattice": lattice})
    elif defect == "edges":
        lattice = field.lattice.model_copy(update={"edges": field.lattice.edges * 129})
        field = field.model_copy(update={"lattice": lattice})
    elif defect == "edge_labels":
        field = field.model_copy(update={"edge_labels": field.edge_labels * 129})
    else:
        label = field.edge_labels[0]
        permutation = label.label.model_copy(
            update={"image": ([0],) if defect == "nested_image" else (0,) * 9}
        )
        field = field.model_copy(
            update={"edge_labels": (label.model_copy(update={"label": permutation}),)}
        )

    copies: list[str] = []

    def unexpected_dump(*args: object, **kwargs: object) -> object:
        copies.append("source")
        raise AssertionError("invalid source must be rejected before recursive dump")

    monkeypatch.setattr(GaugeField, "model_dump", unexpected_dump)
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(_payload(family, wrapped, field=field))
    assert copies == []


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "defect", ("steps", "nested_image", "long_image", "basepoint", "path_basepoint")
)
def test_loop_family_bounds_entries_before_recursive_dump(
    wrapped: bool, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dump sentinel proves each bounded entry is admitted before copying."""
    family = _family()
    entry = family.loops[0]
    if defect == "steps":
        entry = entry.model_copy(
            update={
                "path": entry.path.model_copy(
                    update={"steps": (GaugePathStep(edge_id="ab", forward=True),) * 257}
                )
            }
        )
    elif defect in ("nested_image", "long_image"):
        entry = entry.model_copy(
            update={
                "holonomy": entry.holonomy.model_copy(
                    update={"image": ([0],) if defect == "nested_image" else (0,) * 9}
                )
            }
        )
    elif defect == "basepoint":
        entry = entry.model_copy(update={"basepoint": ["a"]})
    else:
        entry = entry.model_copy(
            update={"path": entry.path.model_copy(update={"basepoint": ["a"]})}
        )

    copies: list[str] = []

    def unexpected_dump(*args: object, **kwargs: object) -> object:
        copies.append("entry")
        raise AssertionError("invalid entry must be rejected before recursive dump")

    monkeypatch.setattr(GaugeLoopHolonomy, "model_dump", unexpected_dump)
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(
            _payload(family, wrapped, loops=(entry,))
        )
    assert copies == []


@pytest.mark.parametrize("wrapped", (False, True))
def test_loop_family_bounds_native_family_count(wrapped: bool) -> None:
    family = _family()
    with pytest.raises(ValidationError):
        GaugeLoopFamilyHolonomies.model_validate(
            _payload(family, wrapped, loops=family.loops * 129)
        )


@pytest.mark.parametrize("kind", ("empty_family", "empty_path", "backtrack"))
@pytest.mark.parametrize("encoding", ("native", "mapping", "json"))
def test_loop_family_preserves_valid_native_and_json_carriers(
    kind: str, encoding: str
) -> None:
    family = _family()
    if kind == "empty_family":
        family = loop_family_holonomies(family.field, ())
    elif kind == "backtrack":
        path = OrientedGaugePath(
            steps=(
                GaugePathStep(edge_id="ab", forward=True),
                GaugePathStep(edge_id="ab", forward=False),
            ),
            basepoint="a",
        )
        family = loop_family_holonomies(family.field, (path,))
    if encoding == "json":
        restored = GaugeLoopFamilyHolonomies.model_validate_json(
            family.model_dump_json()
        )
    else:
        value = family if encoding == "native" else family.model_dump()
        restored = GaugeLoopFamilyHolonomies.model_validate(value)
    assert restored == family
    for entry in restored.loops:
        assert entry.basepoint == entry.path.basepoint == "a"
        assert path_holonomy(restored.field, entry.path).holonomy == entry.holonomy
