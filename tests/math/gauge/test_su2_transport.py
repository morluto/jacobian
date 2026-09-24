"""Exact rational SU(2) transport checked against independent group arithmetic."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge._models import (
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
)
from jacobian.math.gauge._su2_models import (
    SU2GaugeEdgeValue,
    SU2GaugeField,
    SU2GaugeVertexValue,
    SU2HolonomyResult,
)
from jacobian.math.gauge._tools import TOOLS
from jacobian.math.gauge.su2 import (
    su2_gauge_transform,
    su2_path_holonomy,
    su2_wilson_trace,
)
from jacobian.math.quaternions import RationalUnitQuaternion

OPERATION_IDS = {tool.operation_id for tool in TOOLS}


def q(*coordinates: Fraction) -> RationalUnitQuaternion:
    return RationalUnitQuaternion(
        coordinates=tuple(
            CanonicalRational.from_fraction(value) for value in coordinates
        )
    )


def exact_product(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """An independent Hamilton product over exact rationals."""

    a, b, c, d = left
    e, f, g, h = right
    return (
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    )


def exact_inverse(image: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    a, b, c, d = image
    return (a, -b, -c, -d)


def coordinates_of(value: RationalUnitQuaternion) -> tuple[Fraction, ...]:
    return tuple(coordinate.as_fraction() for coordinate in value.coordinates)


LINKS: tuple[tuple[Fraction, ...], ...] = (
    (Fraction(3, 5), Fraction(4, 5), Fraction(0), Fraction(0)),
    (Fraction(5, 13), Fraction(0), Fraction(12, 13), Fraction(0)),
    (Fraction(8, 17), Fraction(0), Fraction(0), Fraction(15, 17)),
    (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
)


def _lattice() -> GaugeLattice:
    return GaugeLattice(
        vertices=("v0", "v1", "v2", "v3"),
        edges=(
            GaugeEdge(edge_id="e0", tail="v0", head="v1"),
            GaugeEdge(edge_id="e1", tail="v1", head="v2"),
            GaugeEdge(edge_id="e2", tail="v2", head="v3"),
            GaugeEdge(edge_id="e3", tail="v3", head="v0"),
        ),
    )


def _field() -> SU2GaugeField:
    return SU2GaugeField(
        lattice=_lattice(),
        edge_values=tuple(
            SU2GaugeEdgeValue(edge_id=f"e{index}", value=q(*link))
            for index, link in enumerate(LINKS)
        ),
    )


def _path(*steps: tuple[int, bool]) -> OrientedGaugePath:
    return OrientedGaugePath(
        steps=tuple(
            GaugePathStep(edge_id=f"e{index}", forward=forward)
            for index, forward in steps
        )
    )


def _forward_path() -> OrientedGaugePath:
    return _path((0, True), (1, True), (2, True), (3, True))


def expected_loop_product(
    links: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    product = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    for link in links:
        product = exact_product(product, link)
    return product


IDENTITY = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))


def test_plaquette_holonomy_matches_exact_rational_composition() -> None:
    result = su2_path_holonomy(_field(), _forward_path())
    assert coordinates_of(result.holonomy) == expected_loop_product(LINKS)
    assert (result.start, result.end) == ("v0", "v0")


def test_backward_traversal_composes_to_the_group_inverse() -> None:
    field = _field()
    forward = su2_path_holonomy(field, _forward_path())
    reverse_path = _path((3, False), (2, False), (1, False), (0, False))
    backward = su2_path_holonomy(field, reverse_path)
    assert coordinates_of(backward.holonomy) == exact_inverse(
        coordinates_of(forward.holonomy)
    )
    assert (backward.start, backward.end) == ("v0", "v0")
    composed = exact_product(
        coordinates_of(forward.holonomy), coordinates_of(backward.holonomy)
    )
    assert composed == IDENTITY


def test_wilson_trace_rejects_a_forged_holonomy_with_valid_endpoints() -> None:
    field = _field()
    path = _forward_path()
    forged = SU2HolonomyResult(
        field=field,
        path=path,
        holonomy=q(Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        start="v0",
        end="v0",
    )

    with pytest.raises(OperationDomainValidationError) as raised:
        su2_wilson_trace(forged)

    assert raised.value.errors()[0]["type"] == (
        "lattice_gauge.su2.holonomy_composition_mismatch"
    )


def test_wilson_trace_rejects_a_native_forged_quaternion_carrier() -> None:
    field = _field()
    forged = SU2HolonomyResult.model_construct(
        field=field,
        path=_forward_path(),
        holonomy=RationalUnitQuaternion.model_construct(coordinates=None),
        start="v0",
        end="v0",
    )
    with pytest.raises(OperationDomainValidationError) as raised:
        su2_wilson_trace(forged)
    assert raised.value.errors()[0]["type"] == "lattice_gauge.su2.wilson_parent"


def test_wilson_trace_of_source_bound_holonomy_is_twice_the_real_part() -> None:
    holonomy = su2_path_holonomy(_field(), _forward_path())
    result = su2_wilson_trace(holonomy)
    expected = 2 * expected_loop_product(LINKS)[0]
    assert result.trace.as_fraction() == expected
    assert result.holonomy == holonomy


def test_wilson_trace_is_invariant_under_an_exact_gauge_transform() -> None:
    field = _field()
    path = _forward_path()
    frames = (
        SU2GaugeVertexValue(vertex="v0", value=q(*LINKS[0])),
        SU2GaugeVertexValue(vertex="v1", value=q(*IDENTITY)),
        SU2GaugeVertexValue(vertex="v2", value=q(*LINKS[1])),
        SU2GaugeVertexValue(vertex="v3", value=q(*IDENTITY)),
    )
    transformed = su2_gauge_transform(field, frames)
    frame_by_vertex = {entry.vertex: coordinates_of(entry.value) for entry in frames}
    for index, edge in enumerate(_lattice().edges):
        expected_edge = exact_product(
            exact_product(
                frame_by_vertex[edge.tail],
                coordinates_of(field.edge_values[index].value),
            ),
            exact_inverse(frame_by_vertex[edge.head]),
        )
        assert (
            coordinates_of(transformed.transformed.edge_values[index].value)
            == expected_edge
        )
    source_loop = su2_path_holonomy(field, path)
    moved_loop = su2_path_holonomy(transformed.transformed, path)
    g0 = coordinates_of(frames[0].value)
    expected = exact_product(
        exact_product(g0, expected_loop_product(LINKS)),
        exact_inverse(g0),
    )
    assert coordinates_of(moved_loop.holonomy) == expected
    assert su2_wilson_trace(moved_loop).trace == su2_wilson_trace(source_loop).trace


def test_wilson_traces_are_native_helpers_not_catalog_operations() -> None:
    assert "lattice_gauge.su2.wilson_trace.compute" not in OPERATION_IDS
    assert "lattice_gauge.permutation.wilson_trace.compute" not in OPERATION_IDS


def test_holonomy_operations_carry_the_wilson_discovery_vocabulary() -> None:
    su2_holonomy = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "lattice_gauge.su2.holonomy.compute"
    )
    holonomy = next(
        tool for tool in TOOLS if tool.operation_id == "lattice_gauge.holonomy.compute"
    )
    assert any("wilson" in term.casefold() for term in su2_holonomy.discovery_terms)
    assert any("wilson" in term.casefold() for term in holonomy.discovery_terms)
