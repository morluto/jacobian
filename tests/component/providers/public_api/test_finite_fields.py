from __future__ import annotations

import pytest
from flint import nmod_mat

from jacobian.domains.finite_fields import build_finite_field_bundle
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    DirectionRankLedger,
    FiniteDimensionalSubspace,
    FiniteFieldElement,
    ProjectiveLine,
    direction_rank_ledger,
    element,
    finite_field,
    orbit_distribution,
    projective_line,
    projective_point,
    restrict_scalars,
)
from jacobian.math.prime_field_linear_algebra import rank

pytestmark = pytest.mark.requires_provider("flint")


def _slice_a_values() -> tuple[
    FiniteDimensionalSubspace,
    ProjectiveLine,
]:
    # The exact presentation, actions, and direction order are Theorem 1.2 of
    # https://arxiv.org/abs/2607.23857. Coefficients are low-degree first.
    presentation = finite_field(2, (1, 1, 0, 1))
    rows = Axis(name="b", labels=("b1", "b2"))
    columns = Axis(name="B^T b", labels=("y1", "y2"))
    basis_axis = Axis(
        name="matrix subspace",
        labels=("B1", "B2", "B3", "B4"),
    )

    def e(coordinates: tuple[int, int, int]) -> FiniteFieldElement:
        return element(presentation, coordinates)

    zero, one, a, a2 = (
        e((0, 0, 0)),
        e((1, 0, 0)),
        e((0, 1, 0)),
        e((0, 0, 1)),
    )
    one_plus_a2 = e((1, 0, 1))
    basis = (
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=rows,
            column_axis=columns,
            entries=((one, zero), (zero, zero)),
        ),
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=rows,
            column_axis=columns,
            entries=((zero, zero), (zero, one)),
        ),
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=rows,
            column_axis=columns,
            entries=((one_plus_a2, one), (a, a2)),
        ),
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=rows,
            column_axis=columns,
            entries=((one_plus_a2, a2), (one_plus_a2, a2)),
        ),
    )
    subspace = FiniteDimensionalSubspace(
        presentation=presentation,
        basis_axis=basis_axis,
        basis=basis,
    )
    paper_affine_order = (
        zero,
        one,
        a,
        e((1, 1, 0)),
        e((1, 1, 1)),
        a2,
        one_plus_a2,
        e((0, 1, 1)),
    )
    directions = ProjectiveLine(
        presentation=presentation,
        axis=rows,
        points=(
            projective_point(presentation, rows, (zero, one)),
            *(
                projective_point(presentation, rows, (one, value))
                for value in paper_affine_order
            ),
        ),
    )
    return subspace, directions


def test_exact_flint_presentation_and_element_coordinates_round_trip() -> None:
    presentation = finite_field(2, (1, 1, 0, 1))

    assert presentation.modulus_coefficients == (1, 1, 0, 1)
    assert presentation.ordered_basis == ("1", "a", "a^2")
    assert tuple(
        tuple(coordinate.coordinates for coordinate in point.coordinates)
        for point in projective_line(
            presentation,
            Axis(name="b", labels=("b1", "b2")),
        ).points[1:]
    ) == tuple(
        ((1, 0, 0), coordinates)
        for coordinates in (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
    )


def test_slice_a_restricts_to_f2_4_to_f2_6_and_matches_paper_ranks() -> None:
    subspace, directions = _slice_a_values()

    maps = tuple(
        restrict_scalars(subspace, direction) for direction in directions.points
    )
    assert all(linear_map.matrix.prime == 2 for linear_map in maps)
    assert all(len(linear_map.matrix.entries) == 6 for linear_map in maps)
    assert all(linear_map.matrix.columns == 4 for linear_map in maps)
    assert tuple(rank(linear_map.matrix) for linear_map in maps) == (
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
    )
    assert tuple(
        nmod_mat([list(row) for row in linear_map.matrix.entries], 2).rank()
        for linear_map in maps
    ) == tuple(rank(linear_map.matrix) for linear_map in maps)


def test_slice_a_keeps_directions_bound_through_orbit_aggregation() -> None:
    subspace, directions = _slice_a_values()

    ledger = direction_rank_ledger(subspace, directions)
    distribution = orbit_distribution(ledger)

    assert tuple(entry.direction for entry in ledger.entries) == directions.points
    assert ledger.subspace is subspace
    assert tuple(entry.rank for entry in ledger.entries) == (3, 3, 3, 3, 3, 3, 4, 4, 4)
    assert distribution.counts == ((1, 9), (8, 48), (16, 12))
    assert distribution.ledger is ledger
    assert (
        type(distribution).model_validate(distribution.model_dump(mode="json"))
        == distribution
    )


def test_orbit_distribution_rejects_a_forged_in_range_rank() -> None:
    subspace, directions = _slice_a_values()
    ledger_payload = direction_rank_ledger(subspace, directions).model_dump(mode="json")
    ledger_payload["entries"][0]["rank"] = 0
    forged = DirectionRankLedger.model_validate(ledger_payload)

    with pytest.raises(ValueError, match="rank does not match"):
        orbit_distribution(forged)


def test_slice_a_ports_compose_restriction_into_rank_without_wire_conversion() -> None:
    subspace, directions = _slice_a_values()
    direction = directions.points[0]
    _, restrict_operation, rank_operation, *_ = build_finite_field_bundle().capabilities

    restrict_payload: dict[str, object] = {}
    for port, value in zip(
        restrict_operation.input_ports,
        (subspace, direction),
        strict=True,
    ):
        restrict_payload = port.bind_to_request(restrict_payload, value)
    restrict_request = restrict_operation.spec.request_type.model_validate(
        restrict_payload
    )
    linear_map = restrict_operation.spec.execute(restrict_request)
    carried_map = restrict_operation.output_ports[0].extract_from_result(linear_map)

    rank_payload: dict[str, object] = {}
    for port, value in zip(
        rank_operation.input_ports,
        (direction, carried_map),
        strict=True,
    ):
        rank_payload = port.bind_to_request(rank_payload, value)
    rank_request = rank_operation.spec.request_type.model_validate(rank_payload)
    result = rank_operation.spec.execute(rank_request)

    assert result.rank == 3
    assert result.direction is direction
    assert result.linear_map is carried_map


def test_slice_a_ports_compose_projective_line_into_orbit_distribution() -> None:
    subspace, _ = _slice_a_values()
    projective, _, _, ledger_operation, orbit_operation, *_ = (
        build_finite_field_bundle().capabilities
    )

    projective_payload: dict[str, object] = {}
    for port, value in zip(
        projective.input_ports,
        (subspace.presentation, subspace.row_axis),
        strict=True,
    ):
        projective_payload = port.bind_to_request(projective_payload, value)
    line = projective.spec.execute(
        projective.spec.request_type.model_validate(projective_payload)
    )

    ledger_payload: dict[str, object] = {}
    for port, value in zip(
        ledger_operation.input_ports,
        (subspace, line),
        strict=True,
    ):
        ledger_payload = port.bind_to_request(ledger_payload, value)
    ledger = ledger_operation.spec.execute(
        ledger_operation.spec.request_type.model_validate(ledger_payload)
    )

    orbit_payload = orbit_operation.input_ports[0].bind_to_request({}, ledger)
    distribution = orbit_operation.spec.execute(
        orbit_operation.spec.request_type.model_validate(orbit_payload)
    )

    assert len(line.points) == 9
    assert tuple(entry.rank for entry in ledger.entries).count(3) == 6
    assert tuple(entry.rank for entry in ledger.entries).count(4) == 3
    assert distribution.counts == ((1, 9), (8, 48), (16, 12))


def test_slice_a_rejects_wrong_presentation_and_axis() -> None:
    subspace, directions = _slice_a_values()
    other_presentation = finite_field(2, (1, 0, 1, 1), generator="z")
    wrong_parent_direction = projective_point(
        other_presentation,
        subspace.row_axis,
        (
            element(other_presentation, (1, 0, 0)),
            element(other_presentation, (0, 0, 0)),
        ),
    )
    wrong_axis = Axis(name="other b", labels=subspace.row_axis.labels)
    wrong_axis_direction = projective_point(
        subspace.presentation,
        wrong_axis,
        directions.points[0].coordinates,
    )

    with pytest.raises(ValueError, match="presentation"):
        restrict_scalars(subspace, wrong_parent_direction)
    with pytest.raises(ValueError, match="axis"):
        restrict_scalars(subspace, wrong_axis_direction)
