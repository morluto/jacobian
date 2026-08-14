"""Bounded exact lattice-basis reduction."""

from __future__ import annotations

from typing import Any, Never

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.matrices import MAX_MATRIX_SCALAR_DIGITS, IntegerMatrix
from jacobian.contracts.matrix_operations import (
    LatticeReductionRequest,
    LatticeReductionResult,
)
from jacobian.contracts.operations import (
    OperationDiagnostic,
    ProviderAvailability,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._examples import example
from jacobian.operation_declarations import InlineOperation
from jacobian.operations import (
    OperationAbortError,
)
from jacobian.providers.flint_runtime import python_flint_lll_provider_runtime

LATTICE_RUNTIME = python_flint_lll_provider_runtime()


def _failure(
    status: ExecutionStatus,
    code: str,
    message: str,
) -> Never:
    raise OperationAbortError(
        status,
        OperationDiagnostic(
            code=code,
            stage="lattice_reduction",
            message=message,
            hint=(
                "Install the pinned Python-FLINT LLL provider or reduce the "
                "matrix size or scalar size."
            ),
        ),
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
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_OUTPUT_LIMIT_EXCEEDED",
            "The exact LLL result exceeded its bounded output contract.",
        )
    return IntegerMatrix(entries=entries)


def reduce_lattice_basis(
    request: LatticeReductionRequest,
) -> LatticeReductionResult:
    if (
        LATTICE_RUNTIME.availability is not ProviderAvailability.AVAILABLE
        or python_flint_lll_provider_runtime(refresh=True) != LATTICE_RUNTIME
    ):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_PROVIDER_UNAVAILABLE",
            "The pinned Python-FLINT exact-gram LLL provider is unavailable.",
        )
    entries = [
        [parse_canonical_integer(value) for value in row]
        for row in request.basis.entries
    ]
    try:
        import flint

        source = flint.fmpz_mat(entries)
        if source.nrows() == 1:
            # Every one-row integer basis is already LLL-reduced.  FLINT rejects
            # this mathematically valid boundary case, so preserve it exactly
            # with the unique one-dimensional identity transformation.
            reduced = source
            transformation = flint.fmpz_mat([[1]])
        else:
            reduced, transformation = source.lll(
                True,
                0.99,
                0.51,
                "zbasis",
                "exact",
            )
        if transformation * source != reduced:
            return _failure(
                ExecutionStatus.ERROR,
                "FLINT_LLL_RELATION_INVALID",
                "The LLL left transformation does not bind the source basis.",
            )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_EXECUTION_FAILED",
            "The pinned Python-FLINT LLL computation did not complete successfully.",
        )
    if python_flint_lll_provider_runtime(refresh=True) != LATTICE_RUNTIME:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_PROVIDER_CHANGED",
            "The Python-FLINT runtime changed during LLL execution.",
        )
    return LatticeReductionResult(
        reduced_basis=_wire(reduced),
        transformation=_wire(transformation),
        rank=int(reduced.rank()),
    )


LATTICE_OPERATIONS: tuple[InlineOperation[Any, Any], ...] = (
    InlineOperation(
        operation_id="lattice.basis.reduce",
        version="3",
        title="Reduce an exact integer lattice basis",
        description=(
            "Run bounded Python-FLINT exact-gram LLL and return the reduced row "
            "basis with its exact left transformation."
        ),
        request_type=LatticeReductionRequest,
        result_type=LatticeReductionResult,
        run=reduce_lattice_basis,
        tags=("lattice", "lll", "exact-integer", "bounded", "python-flint"),
        examples=(
            example(
                "unit_basis",
                "Reduce the one-dimensional unit basis.",
                {"basis": {"entries": [["1"]]}},
            ),
        ),
    ),
)
