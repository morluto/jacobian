"""Bounded exact lattice-basis reduction."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.lattices import (
    LatticeReductionRequest,
    LatticeReductionResult,
)
from jacobian.contracts.matrices import MAX_MATRIX_SCALAR_DIGITS, IntegerMatrix
from jacobian.domains._examples import example
from jacobian.math.lattices import reduce_basis
from jacobian.math_tools import MathTool


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
        version="3",
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
