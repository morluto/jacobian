"""Exact nonabelian gauge transformations and plaquettes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugePathStep,
    GaugeTransformResult,
    GaugeVertexValue,
    OrientedGaugePath,
    PermutationLabel,
    gauge_transform,
    path_holonomy,
    plaquette_curvature,
)


def _field() -> GaugeField:
    lattice = GaugeLattice(
        vertices=("a", "b", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )
    return GaugeField(
        lattice=lattice,
        degree=3,
        edge_labels=tuple(
            GaugeFieldEdgeLabel(
                edge_id=edge, label=PermutationLabel(degree=3, image=image)
            )
            for edge, image in (("ab", (1, 2, 0)), ("bc", (2, 1, 0)), ("ca", (0, 2, 1)))
        ),
    )


def _path() -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=tuple(
            GaugePathStep(edge_id=edge, forward=True) for edge in ("ab", "bc", "ca")
        )
    )


def test_plaquette_orientation_reverses_curvature() -> None:
    field = _field()
    forward = plaquette_curvature(field, _path())
    reverse = path_holonomy(
        field,
        OrientedGaugePath(
            steps=tuple(
                GaugePathStep(edge_id=step.edge_id, forward=False)
                for step in reversed(_path().steps)
            )
        ),
    )
    composed = tuple(
        reverse.holonomy.image[forward.curvature.image[i]] for i in range(3)
    )
    assert composed == (0, 1, 2)


def test_gauge_transform_conjugates_closed_holonomy() -> None:
    field = _field()
    frames = tuple(
        GaugeVertexValue(vertex=vertex, value=PermutationLabel(degree=3, image=image))
        for vertex, image in (("a", (1, 2, 0)), ("b", (0, 1, 2)), ("c", (2, 0, 1)))
    )
    transformed = gauge_transform(field, frames).transformed
    old = path_holonomy(field, _path()).holonomy
    new = path_holonomy(transformed, _path()).holonomy
    h = frames[0].value.image
    inv = [0, 0, 0]
    for i, image in enumerate(h):
        inv[image] = i
    expected = tuple(h[old.image[inv[i]]] for i in range(3))
    assert tuple(new.image) == expected


def test_forged_transformed_field_parent_is_rejected() -> None:
    source = _field()
    foreign_lattice = GaugeLattice(
        vertices=("x", "y", "z"),
        edges=(
            GaugeEdge(edge_id="ab", tail="x", head="y"),
            GaugeEdge(edge_id="bc", tail="y", head="z"),
            GaugeEdge(edge_id="ca", tail="z", head="x"),
        ),
    )
    foreign = GaugeField.model_construct(
        lattice=foreign_lattice,
        degree=source.degree,
        edge_labels=source.edge_labels,
    )
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in source.lattice.vertices
    )
    with pytest.raises(ValidationError):
        GaugeTransformResult.model_validate(
            {
                "source": source.model_dump(),
                "transformed": foreign.model_dump(),
                "vertex_values": [frame.model_dump() for frame in frames],
            }
        )


@pytest.mark.parametrize("variant", ("empty_labels", "none_label", "unhashable_vertex"))
def test_native_transform_rejects_forged_structure_without_raw_exceptions(
    variant: str,
) -> None:
    source = _field()
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in source.lattice.vertices
    )
    if variant == "empty_labels":
        forged_field = GaugeField.model_construct(
            lattice=source.lattice, degree=source.degree, edge_labels=()
        )
        forged_frames = frames
    elif variant == "none_label":
        forged_field = GaugeField.model_construct(
            lattice=source.lattice,
            degree=source.degree,
            edge_labels=tuple(
                GaugeFieldEdgeLabel.model_construct(edge_id="ab", label=None)
                if edge_id == "ab"
                else entry
                for edge_id, entry in (
                    (entry.edge_id, entry) for entry in source.edge_labels
                )
            ),
        )
        forged_frames = frames
    else:
        forged_field = source
        forged_frames = (
            GaugeVertexValue.model_construct(vertex=["a"], value=frames[0].value),
            *frames[1:],
        )
    with pytest.raises(OperationDomainValidationError):
        gauge_transform(forged_field, forged_frames)


def test_transformed_field_retains_lattice_parent() -> None:
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in ("a", "b", "c")
    )
    result = gauge_transform(_field(), frames)
    assert result.transformed.lattice == result.source.lattice
