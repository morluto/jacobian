"""Native exact validation of periodic rational fan presentations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.periodic_fans._kernel import (
    MAX_PERIODIC_TRANSLATION_ENUMERATION,
    _translation_count,
    _translations_between,
    recognize_periodic_fan,
)
from jacobian.math.geometry.periodic_fans._models import (
    MAX_PERIODIC_CELLS,
    MAX_PERIODIC_COORDINATE_DIGITS,
    MAX_PERIODIC_INDEX_DIGITS,
    MAX_PERIODIC_LATTICE_RANK,
    MAX_PERIODIC_OVERLAP_CANDIDATES,
    MAX_PERIODIC_POLYGON_VERTICES,
    MAX_PERIODIC_VERTICES,
    PeriodicFanPresentation,
    PeriodicFanValidationResult,
)
from jacobian.math.matrices._flint import rational_determinant


def _reject_envelope(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("fan",),
        code="geometry.periodic_fan.resource_budget_exceeded",
        message=message,
    )


def _reject_domain(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _digit_weight(value: int | Fraction) -> int:
    return len(str(abs(value)))


def _admit_periodic_fan(fan: PeriodicFanPresentation) -> None:  # noqa: C901
    """Enforce the published envelope and preflight overlap enumeration.

    Catalog requests are already bounded by the presentation model, but native
    callers can bypass wire validation, so the same envelope and the derived
    translation-enumeration count are checked here before any exact arithmetic
    starts.
    """

    rank = fan.lattice_rank
    if not 1 <= rank <= MAX_PERIODIC_LATTICE_RANK:
        _reject_envelope(f"lattice rank is limited to {MAX_PERIODIC_LATTICE_RANK}")
    if len(fan.cells) > MAX_PERIODIC_CELLS:
        _reject_envelope(f"periodic fans are limited to {MAX_PERIODIC_CELLS} cells")
    if len(fan.vertices) > MAX_PERIODIC_VERTICES:
        _reject_envelope(
            f"periodic fans are limited to {MAX_PERIODIC_VERTICES} vertices"
        )
    if len(fan.overlap_candidates) > MAX_PERIODIC_OVERLAP_CANDIDATES:
        _reject_envelope(
            "periodic fans are limited to "
            f"{MAX_PERIODIC_OVERLAP_CANDIDATES} overlap candidates"
        )
    if len(fan.period_basis) != rank or any(
        len(row) != rank for row in fan.period_basis
    ):
        _reject_domain(
            ("fan", "period_basis"),
            "geometry.periodic_fan.period_basis_shape",
            "period_basis must be a square lattice_rank x lattice_rank matrix",
        )
    if any(
        canonical_rational_component_digits(value) > MAX_PERIODIC_COORDINATE_DIGITS
        for row in fan.period_basis
        for value in row
    ):
        _reject_envelope(
            "period basis coordinates are limited to "
            f"{MAX_PERIODIC_COORDINATE_DIGITS} decimal digits"
        )
    coordinate_limit = 10**MAX_PERIODIC_COORDINATE_DIGITS
    for vertex in fan.vertices:
        if len(vertex) != rank:
            _reject_domain(
                ("fan", "vertices"),
                "geometry.periodic_fan.vertex_length_matches_lattice_rank",
                "every vertex must have exactly lattice_rank coordinates",
            )
        if any(abs(value) >= coordinate_limit for value in vertex):
            _reject_envelope(
                "vertex coordinates are limited to "
                f"{MAX_PERIODIC_COORDINATE_DIGITS} decimal digits"
            )
    for cell in fan.cells:
        max_cell_vertices = MAX_PERIODIC_POLYGON_VERTICES if rank == 2 else rank + 1
        if len(cell) < rank + 1 or len(cell) > max_cell_vertices:
            _reject_domain(
                ("fan", "cells"),
                "geometry.periodic_fan.cell_dimension_matches_lattice_rank",
                "maximal cells must be simplices, except for bounded rank-two polygons",
            )
        if any(index < 0 or index >= len(fan.vertices) for index in cell):
            _reject_domain(
                ("fan", "cells"),
                "geometry.periodic_fan.cell_vertex_index_out_of_range",
                "cell vertex indices must address declared vertices",
            )
        if len(set(cell)) != len(cell):
            _reject_domain(
                ("fan", "cells"),
                "geometry.periodic_fan.cell_vertex_indices_distinct",
                "cell vertex indices must be distinct",
            )
        if len(cell) == rank + 1 and tuple(sorted(cell)) != cell:
            _reject_domain(
                ("fan", "cells"),
                "geometry.periodic_fan.cell_vertex_indices_strictly_increasing",
                "simplex vertex indices must be strictly increasing",
            )
        if rank == 2 and len(cell) > rank + 1 and cell[0] != min(cell):
            _reject_domain(
                ("fan", "cells"),
                "geometry.periodic_fan.polygon_vertex_order",
                "polygon order must begin at its smallest vertex index",
            )
    if tuple(sorted(fan.cells)) != fan.cells or len(set(fan.cells)) != len(fan.cells):
        _reject_domain(
            ("fan", "cells"),
            "geometry.periodic_fan.cells_sorted_and_distinct",
            "cells must be sorted lexicographically and distinct",
        )
    if any(len(fan.cells[index]) != rank + 1 for index in fan.unimodular_cells):
        _reject_domain(
            ("fan", "unimodular_cells"),
            "geometry.periodic_fan.unimodular_cell_not_simplex",
            "unimodularity claims are defined only for simplex cells",
        )
    if any(
        candidate.first_cell < 0
        or candidate.first_cell >= len(fan.cells)
        or candidate.second_cell < 0
        or candidate.second_cell >= len(fan.cells)
        for candidate in fan.overlap_candidates
    ):
        _reject_domain(
            ("fan", "overlap_candidates"),
            "geometry.periodic_fan.overlap_candidate_index_out_of_range",
            "overlap candidate cell indices must address declared cells",
        )
    period_determinant = rational_determinant(
        tuple(tuple(value.as_fraction() for value in row) for row in fan.period_basis)
    )
    if _digit_weight(period_determinant) > MAX_PERIODIC_INDEX_DIGITS:
        _reject_envelope(
            f"the period index is limited to {MAX_PERIODIC_INDEX_DIGITS} decimal digits"
        )
    enumeration = 0
    cell_coordinates = [
        tuple(fan.vertices[index] for index in cell) for cell in fan.cells
    ]
    for first in range(len(fan.cells)):
        for second in range(first, len(fan.cells)):
            enumeration += _translation_count(
                _translations_between(cell_coordinates[first], cell_coordinates[second])
            )
            if enumeration > MAX_PERIODIC_TRANSLATION_ENUMERATION:
                _reject_envelope(
                    "periodic overlap candidate enumeration exceeds "
                    f"{MAX_PERIODIC_TRANSLATION_ENUMERATION} translations"
                )


def validate_periodic_fan(
    fan: PeriodicFanPresentation,
) -> PeriodicFanValidationResult:
    """Recognize a periodic fan and retain the first exact obstruction."""

    _admit_periodic_fan(fan)
    recognized = recognize_periodic_fan(
        lattice_rank=fan.lattice_rank,
        period_basis=tuple(
            tuple(value.as_fraction() for value in row) for row in fan.period_basis
        ),
        vertices=fan.vertices,
        cells=fan.cells,
        unimodular_cells=fan.unimodular_cells,
        overlap_candidates=tuple(
            (
                candidate.first_cell,
                candidate.second_cell,
                candidate.translation,
            )
            for candidate in fan.overlap_candidates
        ),
    )
    return PeriodicFanValidationResult._from_recognition(fan, recognized=recognized)


__all__ = ["_admit_periodic_fan", "validate_periodic_fan"]
