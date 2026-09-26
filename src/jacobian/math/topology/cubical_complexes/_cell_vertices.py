"""Exact vertex enumeration for one elementary cubical cell."""

from __future__ import annotations

from itertools import product

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cubical_complexes._models import MAX_DIM, CubicalCell

MAX_CUBICAL_CELL_VERTEX_COORDINATE_DIGITS = 64
MAX_CUBICAL_CELL_VERTEX_COUNT = 1 << 10
MAX_CUBICAL_CELL_VERTICES_RESULT_BYTES = 512 * 1024


def _coordinate_digit_count(coordinate: int) -> int:
    """Return the magnitude decimal digit width without a raw string error.

    The bit-length preflight rejects astronomically large native values before
    any decimal formatting, so the canonical digit-width helper never has to
    materialize a decimal string beyond Python's configured conversion limit.
    """

    if abs(coordinate).bit_length() > 4 * MAX_CUBICAL_CELL_VERTEX_COORDINATE_DIGITS:
        return MAX_CUBICAL_CELL_VERTEX_COORDINATE_DIGITS + 1
    return decimal_digit_width(coordinate)


class CubicalCellVerticesResult(StrictModel):
    """The source cell and its complete, canonically ordered point cells."""

    cell: CubicalCell
    vertices: tuple[CubicalCell, ...] = Field(
        min_length=1, max_length=MAX_CUBICAL_CELL_VERTEX_COUNT
    )


def _result_size_bound(cell: CubicalCell, vertex_count: int) -> int:
    """Bound compact JSON size before constructing the vertex family."""
    source_bytes = len(cell.model_dump_json().encode("utf-8"))
    # A point repeats each endpoint twice in its interval pair. This bound
    # includes pair separators, the interval list, and a generous object/key
    # allowance; it is independent of the number of coordinate combinations.
    one_vertex_bytes = 32 + sum(
        2 * max(_coordinate_digit_count(lower), _coordinate_digit_count(upper)) + 4
        for lower, upper in cell.intervals
    )
    return source_bytes + vertex_count * one_vertex_bytes + 128


def cell_vertices(cell: CubicalCell) -> CubicalCellVerticesResult:
    """Return all vertices of an elementary integer-lattice cube exactly."""
    if type(cell) is not CubicalCell:
        raise OperationDomainValidationError(
            location=("cell",),
            code="cubical_complex.cell_vertices_invalid_cell",
            message="cell vertices require a canonical cubical cell",
        )
    try:
        cell = CubicalCell.model_validate(cell.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("cell",),
            code="cubical_complex.cell_vertices_invalid_cell",
            message="cell vertices require a canonical cubical cell",
        ) from exc
    if len(cell.intervals) > MAX_DIM:
        raise OperationDomainValidationError(
            location=("cell",),
            code="cubical_complex.cell_vertices_ambient_dimension",
            message=f"cell ambient dimension exceeds the {MAX_DIM}-axis bound",
        )
    if any(
        _coordinate_digit_count(coordinate) > MAX_CUBICAL_CELL_VERTEX_COORDINATE_DIGITS
        for interval in cell.intervals
        for coordinate in interval
    ):
        raise OperationResourceAdmissionError(
            location=("cell",),
            code="cubical_complex.cell_vertices_coordinate_digits",
            message=(
                "cell vertex coordinates exceed the "
                f"{MAX_CUBICAL_CELL_VERTEX_COORDINATE_DIGITS}-digit bound"
            ),
        )

    vertex_count = 1 << cell.dimension
    if vertex_count > MAX_CUBICAL_CELL_VERTEX_COUNT:
        raise OperationResourceAdmissionError(
            location=("cell",),
            code="cubical_complex.cell_vertices_count",
            message="cell vertex count exceeds the output bound",
        )
    if _result_size_bound(cell, vertex_count) > MAX_CUBICAL_CELL_VERTICES_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cell",),
            code="cubical_complex.cell_vertices_result_bytes",
            message=(
                "cell vertex result exceeds the "
                f"{MAX_CUBICAL_CELL_VERTICES_RESULT_BYTES}-byte bound"
            ),
        )

    coordinate_choices = tuple(
        (lower,) if lower == upper else (lower, upper)
        for lower, upper in cell.intervals
    )
    vertices = tuple(
        CubicalCell(intervals=tuple((coordinate, coordinate) for coordinate in point))
        for point in product(*coordinate_choices)
    )
    return CubicalCellVerticesResult(cell=cell, vertices=vertices)


__all__ = [
    "CubicalCellVerticesResult",
    "cell_vertices",
]
