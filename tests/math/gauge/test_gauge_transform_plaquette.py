"""Exact nonabelian gauge transformations and plaquettes."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge import (
    GaugeEdge,
    GaugeField,
    GaugeFieldEdgeLabel,
    GaugeLattice,
    GaugePathStep,
    GaugeVertexValue,
    OrientedGaugePath,
    PermutationLabel,
    gauge_transform,
    path_holonomy,
    plaquette_curvature,
)
from jacobian.math.gauge._models import GaugeTransformResult, PlaquetteResult


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
    # A transposition at the start frame does not commute with this field
    # holonomy, so the assertion distinguishes h^-1 H h from h H h^-1.
    frames = tuple(
        GaugeVertexValue(vertex=vertex, value=PermutationLabel(degree=3, image=image))
        for vertex, image in (("a", (1, 0, 2)), ("b", (0, 1, 2)), ("c", (0, 1, 2)))
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


def test_forged_transform_result_rejects_a_foreign_lattice() -> None:
    field = _field()
    foreign_lattice = GaugeLattice(
        vertices=("b", "a", "c"),
        edges=(
            GaugeEdge(edge_id="ab", tail="a", head="b"),
            GaugeEdge(edge_id="bc", tail="b", head="c"),
            GaugeEdge(edge_id="ca", tail="c", head="a"),
        ),
    )
    foreign = GaugeField(
        lattice=foreign_lattice,
        degree=3,
        edge_labels=field.edge_labels,
    )
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in ("a", "b", "c")
    )
    with pytest.raises(ValueError, match="transform_parent"):
        GaugeTransformResult.model_validate(
            {"source": field, "transformed": foreign, "vertex_values": frames}
        )


def test_forged_plaquette_result_rejects_unbound_curvature_degree() -> None:
    field = _field()
    path = _path()
    with pytest.raises(ValueError, match="plaquette_degree"):
        PlaquetteResult.model_validate(
            {
                "field": field,
                "path": path,
                "curvature": PermutationLabel(degree=2, image=(0, 1)),
                "start": "a",
            }
        )


def test_native_transform_rejects_forged_nested_values() -> None:
    field = _field()
    forged = GaugeVertexValue.model_construct(vertex="a", value=None)
    with pytest.raises(OperationDomainValidationError):
        gauge_transform(field, (forged,))

    forged_label = PermutationLabel.model_construct(degree=3, image=(0, 0, 1))
    forged_field = GaugeField.model_construct(
        lattice=field.lattice,
        degree=field.degree,
        edge_labels=(
            GaugeFieldEdgeLabel.model_construct(edge_id="ab", label=forged_label),
            *field.edge_labels[1:],
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        path_holonomy(forged_field, _path())

    incomplete = GaugeField.model_construct(
        lattice=field.lattice, degree=field.degree, edge_labels=()
    )
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in ("a", "b", "c")
    )
    with pytest.raises(OperationDomainValidationError):
        gauge_transform(incomplete, frames)


def test_native_holonomy_rejects_forged_path_steps_with_owner_error() -> None:
    field = _field()
    for step in (
        None,
        GaugePathStep.model_construct(edge_id="ab", forward=None),
    ):
        forged = OrientedGaugePath.model_construct(steps=(step,))
        with pytest.raises(OperationDomainValidationError):
            path_holonomy(field, forged)
        with pytest.raises(OperationDomainValidationError):
            plaquette_curvature(field, forged)


def test_transformed_field_retains_lattice_parent() -> None:
    frames = tuple(
        GaugeVertexValue(
            vertex=vertex, value=PermutationLabel(degree=3, image=(0, 1, 2))
        )
        for vertex in ("a", "b", "c")
    )
    result = gauge_transform(_field(), frames)
    assert result.transformed.lattice == result.source.lattice
