"""Contract tests for exact periodic rational fan validation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.periodic_fans._models import (
    PeriodicFanPresentation,
    PeriodicFanValidationRequest,
    PeriodicFanValidationResult,
    PeriodicOverlapCandidate,
)
from jacobian.math.geometry.periodic_fans._tools import (
    PERIODIC_FAN_VALIDATE_OPERATION,
    TOOLS,
)
from jacobian.math.geometry.periodic_fans.operations import validate_periodic_fan


def q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def overlap(
    first: int, second: int, translation: tuple[int, ...]
) -> PeriodicOverlapCandidate:
    return PeriodicOverlapCandidate(
        first_cell=first, second_cell=second, translation=translation
    )


def _translations(rank: int, span: int) -> tuple[tuple[int, ...], ...]:
    if rank == 0:
        return ((),)
    return tuple(
        (head, *tail)
        for head in range(-span, span + 1)
        for tail in _translations(rank - 1, span)
    )


def all_translates(
    cells: int, rank: int, span: int = 1
) -> tuple[PeriodicOverlapCandidate, ...]:
    return tuple(
        overlap(first, second, translation)
        for first in range(cells)
        for second in range(first, cells)
        for translation in _translations(rank, span)
    )


def unit_square_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(1), q(0)), (q(0), q(1))),
        vertices=((0, 0), (1, 0), (0, 1), (1, 1)),
        cells=((0, 1, 3), (0, 2, 3)),
        unimodular_cells=(0, 1),
        overlap_candidates=all_translates(2, 2),
    )


def one_dimensional_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=1,
        period_basis=((q(1),),),
        vertices=((0,), (1,)),
        cells=((0, 1),),
        unimodular_cells=(0,),
        overlap_candidates=all_translates(1, 1),
    )


def square_polygon_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(1), q(0)), (q(0), q(1))),
        vertices=((0, 0), (1, 0), (0, 1), (1, 1)),
        # The vertex sequence is a counterclockwise boundary cycle.
        cells=((0, 1, 3, 2),),
        overlap_candidates=all_translates(1, 2, 1),
    )


def hexagon_and_triangles_fan() -> PeriodicFanPresentation:
    # Opposite corners of the square fundamental domain are cut off so that
    # boundary faces match under period translations.
    return PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(2), q(0)), (q(0), q(2))),
        vertices=((0, 0), (1, 0), (2, 1), (2, 2), (1, 2), (0, 1), (2, 0), (0, 2)),
        cells=((0, 1, 2, 3, 4, 5), (1, 2, 6), (4, 5, 7)),
        overlap_candidates=all_translates(3, 2, 2),
    )


def rank_deficient_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=1,
        period_basis=((q(0),),),
        vertices=((0,), (1,)),
        cells=((0, 1),),
        overlap_candidates=all_translates(1, 1),
    )


def non_integral_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=1,
        period_basis=((q(1, 2),),),
        vertices=((0,), (1,)),
        cells=((0, 1),),
        overlap_candidates=all_translates(1, 1),
    )


def non_unimodular_claim_fan() -> PeriodicFanPresentation:
    return PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(2), q(0)), (q(0), q(1))),
        vertices=((0, 0), (2, 0), (0, 1)),
        cells=((0, 1, 2),),
        unimodular_cells=(0,),
    )


def non_face_to_face_fan() -> PeriodicFanPresentation:
    # Two triangles in the [0,2]x[0,1] fundamental parallelotope of
    # B = ((2,0),(0,1)) cross in a smaller triangle rather than a common face.
    return PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(2), q(0)), (q(0), q(1))),
        vertices=((0, 0), (2, 0), (0, 1), (1, 1)),
        cells=((0, 1, 2), (0, 2, 3)),
        overlap_candidates=(overlap(0, 1, (0, 0)),),
    )


def test_catalog_contains_the_periodic_fan_validator() -> None:
    assert {tool.operation_id for tool in TOOLS} == {"periodic_fan.quotient.validate"}


def test_unit_square_quotient_known_answer() -> None:
    result = validate_periodic_fan(unit_square_fan())
    assert result.status == "VALID"
    assert result.period_index == 1
    assert result.locally_finite is True
    assert result.face_to_face is True
    assert result.covers_fundamental_domain is True
    by_dimension: dict[int, int] = {}
    for cell in result.quotient_cells:
        by_dimension[cell.dimension] = by_dimension.get(cell.dimension, 0) + 1
    assert by_dimension == {0: 1, 1: 3, 2: 2}
    assert all(cell.stabilizer_rank == 0 for cell in result.quotient_cells)
    assert all(cell.stabilizer_index is None for cell in result.quotient_cells)


def test_one_dimensional_periodic_fan() -> None:
    result = validate_periodic_fan(one_dimensional_fan())
    assert result.status == "VALID"
    assert result.period_index == 1
    assert [cell.dimension for cell in result.quotient_cells] == [0, 1]
    assert {(r.tau_cell_id, r.sigma_cell_id) for r in result.face_relations} == {
        (0, 0),
        (0, 1),
        (1, 1),
    }


def test_periodic_quadrilateral_returns_its_polygon_face_lattice() -> None:
    result = validate_periodic_fan(square_polygon_fan())
    assert result.status == "VALID"
    assert result.period_index == 1
    by_dimension: dict[int, int] = {}
    for cell in result.quotient_cells:
        by_dimension[cell.dimension] = by_dimension.get(cell.dimension, 0) + 1
    # A square tiling of the plane has one vertex orbit, two edge orbits,
    # and one polygon orbit. Its diagonal is not a face.
    assert by_dimension == {0: 1, 1: 2, 2: 1}
    assert len(result.face_orbit_rows) == 9
    ids_by_dimension = {
        dimension: {
            cell.cell_id
            for cell in result.quotient_cells
            if cell.dimension == dimension
        }
        for dimension in range(3)
    }
    relations = {(row.tau_cell_id, row.sigma_cell_id) for row in result.face_relations}
    expected_relations = {
        (cell_id, cell_id) for ids in ids_by_dimension.values() for cell_id in ids
    }
    polygon_id = next(iter(ids_by_dimension[2]))
    expected_relations.update(
        (vertex_id, edge_id)
        for vertex_id in ids_by_dimension[0]
        for edge_id in ids_by_dimension[1]
    )
    expected_relations.update(
        (face_id, polygon_id)
        for face_id in (*ids_by_dimension[0], *ids_by_dimension[1])
    )
    assert relations == expected_relations


def test_periodic_hexagon_and_triangles_cover_fundamental_domain() -> None:
    result = validate_periodic_fan(hexagon_and_triangles_fan())
    assert result.status == "VALID"
    assert result.covers_fundamental_domain is True
    assert sum(cell.dimension == 2 for cell in result.quotient_cells) == 3


def test_periodic_quadrilateral_rejects_nonconvex_vertex_cycle() -> None:
    fan = PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(2), q(0)), (q(0), q(2))),
        vertices=((0, 0), (2, 0), (0, 2), (2, 2)),
        cells=((0, 1, 2, 3),),
    )
    result = validate_periodic_fan(fan)
    assert result.status == "INVALID"
    assert result.obstruction_code == (
        "geometry.periodic_fan.polygon_not_strictly_convex"
    )


def test_unimodularity_claim_is_rejected_for_a_polygon_cell() -> None:
    payload = square_polygon_fan().model_dump()
    payload["unimodular_cells"] = (0,)
    with pytest.raises(ValidationError, match="unimodularity claims"):
        PeriodicFanPresentation.model_validate(payload)


def test_period_lattice_rank_deficient_is_rejected() -> None:
    result = validate_periodic_fan(rank_deficient_fan())
    assert result.status == "INVALID"
    assert result.obstruction_code == "geometry.periodic_fan.period_rank_deficient"


def test_period_lattice_non_integral_is_rejected() -> None:
    result = validate_periodic_fan(non_integral_fan())
    assert result.status == "INVALID"
    assert result.obstruction_code == "geometry.periodic_fan.period_not_integral"


def test_non_face_to_face_overlap_names_the_offending_pair() -> None:
    result = validate_periodic_fan(non_face_to_face_fan())
    assert result.status == "INVALID"
    assert result.obstruction_code == "geometry.periodic_fan.overlap_not_face_to_face"
    assert "cells 0 and 1" in (result.obstruction_message or "")


def test_claimed_non_unimodular_cell_is_rejected() -> None:
    result = validate_periodic_fan(non_unimodular_claim_fan())
    assert result.status == "INVALID"
    assert result.obstruction_code == (
        "geometry.periodic_fan.claimed_cell_not_unimodular"
    )


def test_undeclared_overlap_is_rejected() -> None:
    fan = unit_square_fan()
    partial = tuple(
        candidate
        for candidate in fan.overlap_candidates
        if (candidate.first_cell, candidate.second_cell, candidate.translation)
        != (0, 1, (0, 0))
    )
    incomplete = PeriodicFanPresentation(
        lattice_rank=fan.lattice_rank,
        period_basis=fan.period_basis,
        vertices=fan.vertices,
        cells=fan.cells,
        unimodular_cells=fan.unimodular_cells,
        overlap_candidates=partial,
    )
    result = validate_periodic_fan(incomplete)
    assert result.status == "INVALID"
    assert result.obstruction_code == "geometry.periodic_fan.undeclared_overlap"


def test_outside_fundamental_domain_is_rejected() -> None:
    fan = PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(1), q(0)), (q(0), q(1))),
        vertices=((0, 0), (2, 0), (0, 1)),
        cells=((0, 1, 2),),
    )
    result = validate_periodic_fan(fan)
    assert result.obstruction_code == (
        "geometry.periodic_fan.cell_outside_fundamental_domain"
    )


def test_degenerate_cell_is_rejected() -> None:
    fan = PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(3), q(0)), (q(0), q(1))),
        vertices=((0, 0), (1, 0), (2, 0)),
        cells=((0, 1, 2),),
    )
    result = validate_periodic_fan(fan)
    assert result.obstruction_code == "geometry.periodic_fan.cell_degenerate"


def test_validation_result_round_trips_through_strict_json() -> None:
    valid = validate_periodic_fan(unit_square_fan())
    replayed = PeriodicFanValidationResult.model_validate_json(valid.model_dump_json())
    assert replayed == valid
    invalid = validate_periodic_fan(non_face_to_face_fan())
    replayed_invalid = PeriodicFanValidationResult.model_validate_json(
        invalid.model_dump_json()
    )
    assert replayed_invalid == invalid
    assert replayed_invalid.fan == invalid.fan

    polygon = validate_periodic_fan(square_polygon_fan())
    replayed_polygon = PeriodicFanValidationResult.model_validate_json(
        polygon.model_dump_json()
    )
    assert replayed_polygon == polygon


def test_catalog_and_native_paths_share_one_recognition() -> None:
    payload = PERIODIC_FAN_VALIDATE_OPERATION.examples[0].input
    request = PeriodicFanValidationRequest.model_validate_json(
        encode_strict_json(payload), strict=True
    )
    catalog_result = PERIODIC_FAN_VALIDATE_OPERATION.run(request)
    native_result = validate_periodic_fan(unit_square_fan())
    assert catalog_result == native_result


def test_truthful_index_is_computed_for_a_non_trivial_period() -> None:
    fan = PeriodicFanPresentation(
        lattice_rank=1,
        period_basis=((q(3),),),
        vertices=((0,), (1,), (2,), (3,)),
        cells=((0, 1), (1, 2), (2, 3)),
        overlap_candidates=all_translates(3, 1, span=3),
    )
    result = validate_periodic_fan(fan)
    assert result.status == "VALID"
    assert result.period_index == 3


def test_structural_wire_rejections_are_owner_coded() -> None:
    with pytest.raises(ValidationError) as unsorted_cells:
        PeriodicFanPresentation(
            lattice_rank=2,
            period_basis=((q(1), q(0)), (q(0), q(1))),
            vertices=((0, 0), (1, 0), (0, 1), (1, 1)),
            cells=((0, 2, 3), (0, 1, 3)),
        )
    assert unsorted_cells.value.errors()[0]["type"].startswith("geometry.periodic_fan.")
    with pytest.raises(ValidationError) as wrong_size:
        PeriodicFanPresentation(
            lattice_rank=2,
            period_basis=((q(1), q(0)), (q(0), q(1))),
            vertices=((0, 0), (1, 0), (0, 1)),
            cells=((0, 1),),
        )
    assert wrong_size.value.errors()[0]["type"].startswith("geometry.periodic_fan.")
    with pytest.raises(ValidationError) as non_square:
        PeriodicFanPresentation(
            lattice_rank=2,
            period_basis=((q(1), q(0)),),
            vertices=((0, 0), (1, 0), (0, 1)),
            cells=((0, 1, 2),),
        )
    assert non_square.value.errors()[0]["type"].startswith("geometry.periodic_fan.")


def test_envelope_rejects_high_rank_from_a_native_caller() -> None:
    oversized = PeriodicFanPresentation.model_construct(
        lattice_rank=5,
        period_basis=(),
        vertices=(),
        cells=(),
        unimodular_cells=(),
        overlap_candidates=(),
    )
    with pytest.raises(OperationResourceAdmissionError) as rejection:
        validate_periodic_fan(oversized)
    assert rejection.value.errors()[0]["type"] == (
        "geometry.periodic_fan.resource_budget_exceeded"
    )


def test_envelope_rejects_oversized_coordinates_from_a_native_caller() -> None:
    oversized = PeriodicFanPresentation.model_construct(
        lattice_rank=2,
        period_basis=((q(1), q(0)), (q(0), q(1))),
        vertices=((10**9, 0), (0, 1), (1, 1)),
        cells=((0, 1, 2),),
        unimodular_cells=(),
        overlap_candidates=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        validate_periodic_fan(oversized)


def test_envelope_rejects_unbounded_translation_enumeration() -> None:
    huge = PeriodicFanPresentation(
        lattice_rank=2,
        period_basis=((q(1000), q(0)), (q(0), q(1000))),
        vertices=((0, 0), (1000, 0), (0, 1000)),
        cells=((0, 1, 2),),
        unimodular_cells=(0,),
        overlap_candidates=(),
    )
    with pytest.raises(OperationResourceAdmissionError) as rejection:
        validate_periodic_fan(huge)
    assert rejection.value.errors()[0]["type"] == (
        "geometry.periodic_fan.resource_budget_exceeded"
    )


def test_example_payload_carries_expected_candidate_count() -> None:
    candidates = PERIODIC_FAN_VALIDATE_OPERATION.examples[0].input["fan"][
        "overlap_candidates"
    ]
    assert len(candidates) == 27
