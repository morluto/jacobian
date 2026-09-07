"""Bounded exact lattice-basis reduction."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lattices import reduce_basis
from jacobian.math.lattices._models import (
    LatticeReductionRequest,
    LatticeReductionResult,
    _require_lattice_matrix_envelope,
    _run_admission,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
)


def _wire(matrix: Any) -> IntegerMatrix:
    entries = tuple(
        tuple(int(matrix[row, column]) for column in range(matrix.ncols()))
        for row in range(matrix.nrows())
    )
    if any(
        len(format_canonical_integer(abs(value))) > MAX_MATRIX_SCALAR_DIGITS
        for row in entries
        for value in row
    ):
        raise ValueError("The exact LLL result exceeded its bounded output contract.")
    return IntegerMatrix(entries=entries)


def reduce_lattice_basis(
    request: LatticeReductionRequest,
) -> LatticeReductionResult:
    _run_admission(
        lambda: _require_lattice_matrix_envelope(request.basis, label="basis input"),
        location=("basis",),
    )
    entries = [list(row) for row in request.basis.entries]
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
            "Reduce a bounded exact integer row basis and return the retained "
            "source basis with its exact left transformation."
        ),
        request_type=LatticeReductionRequest,
        result_type=LatticeReductionResult,
        run=reduce_lattice_basis,
        tags=("lattice", "lll", "exact-integer", "bounded"),
        examples=(
            OperationExample(
                name="unit_basis",
                description="Reduce the one-dimensional unit basis.",
                input={"basis": {"entries": [["1"]]}},
            ),
        ),
    ),
)
