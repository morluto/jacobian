"""Bounded exact lattice-basis reduction."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.lattices import reduce_basis
from jacobian.math.lattices._models import (
    _MAX_LATTICE_INPUT_SCALAR_DIGITS,
    LatticeReductionRequest,
    LatticeReductionResult,
    _require_lattice_matrix_dimensions,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    require_matrix_scalar_digits,
)


def _wire(matrix: Any) -> IntegerMatrix:
    entries = tuple(
        tuple(
            format_canonical_integer(int(matrix[row, column]))
            for column in range(matrix.ncols())
        )
        for row in range(matrix.nrows())
    )
    if any(
        len(value.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
        for row in entries
        for value in row
    ):
        raise ValueError("The exact LLL result exceeded its bounded output contract.")
    return IntegerMatrix(entries=entries)


def reduce_lattice_basis(
    request: LatticeReductionRequest,
) -> LatticeReductionResult:
    require_matrix_scalar_digits(
        request.basis.entries,
        maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
        label="basis input",
    )
    _require_lattice_matrix_dimensions(request.basis, label="basis")
    entries = [
        [parse_canonical_integer(value) for value in row]
        for row in request.basis.entries
    ]
    reduced, transformation, rank = reduce_basis(entries)
    return LatticeReductionResult(
        reduced_basis=_wire(reduced),
        transformation=_wire(transformation),
        rank=rank,
    )


LATTICE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lattice.basis.reduce",
        title="Reduce an exact integer lattice basis",
        description=(
            "Reduce a bounded exact integer row basis and return its exact left "
            "transformation."
        ),
        request_type=LatticeReductionRequest,
        result_type=LatticeReductionResult,
        run=reduce_lattice_basis,
        tags=("lattice", "lll", "exact-integer", "bounded"),
        examples=(
            example(
                "unit_basis",
                "Reduce the one-dimensional unit basis.",
                {"basis": {"entries": [["1"]]}},
            ),
        ),
    ),
)
