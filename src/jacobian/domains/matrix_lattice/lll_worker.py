"""Isolated Python-FLINT worker for exact integer LLL reduction."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.contracts.matrices import MAX_MATRIX_SCALAR_DIGITS, IntegerMatrix
from jacobian.contracts.matrix_operations import LatticeReductionResult
from jacobian.domains.matrix_lattice.lll_protocol import (
    PROTOCOL,
    LllWorkerErrorCode,
    LllWorkerFailure,
    LllWorkerRequest,
    LllWorkerResponse,
    parse_lll_worker_request,
)

INPUT_LIMIT = 1_000_000


class WorkerError(RuntimeError):
    def __init__(self, code: LllWorkerErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def _read() -> LllWorkerRequest:
    encoded = sys.stdin.buffer.read(INPUT_LIMIT + 1)
    if len(encoded) > INPUT_LIMIT:
        raise WorkerError(LllWorkerErrorCode.INPUT_LIMIT_EXCEEDED)
    try:
        payload = loads_strict_json(encoded)
    except CanonicalizationError as exc:
        raise WorkerError(LllWorkerErrorCode.INVALID_REQUEST) from exc
    try:
        return parse_lll_worker_request(payload)
    except ValueError as exc:
        raise WorkerError(LllWorkerErrorCode.MATRIX_INVALID) from exc


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
        raise WorkerError(LllWorkerErrorCode.OUTPUT_LIMIT_EXCEEDED)
    return IntegerMatrix(entries=entries)


def _run(worker_request: LllWorkerRequest) -> LllWorkerResponse:
    try:
        flint: Any = importlib.import_module("flint")
        if (
            getattr(flint, "__version__", None) != "0.9.0"
            or getattr(flint, "__FLINT_VERSION__", None) != "3.6.0"
        ):
            raise WorkerError(LllWorkerErrorCode.VERSION_MISMATCH)
        entries = [
            [parse_canonical_integer(value) for value in row]
            for row in worker_request.request.basis.entries
        ]
        source = flint.fmpz_mat(entries)
        reduced, transformation = source.lll(
            transform=True,
            delta=0.99,
            eta=0.51,
            rep="zbasis",
            gram="exact",
        )
        if transformation * source != reduced:
            raise WorkerError(LllWorkerErrorCode.RELATION_INVALID)
    except WorkerError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise WorkerError(LllWorkerErrorCode.EXECUTION_FAILED) from None
    return LllWorkerResponse(
        protocol=PROTOCOL,
        result=LatticeReductionResult(
            reduced_basis=_wire(reduced),
            transformation=_wire(transformation),
            rank=int(reduced.rank()),
        ),
    )


def main() -> int:
    try:
        output: LllWorkerResponse | LllWorkerFailure = _run(_read())
    except WorkerError as exc:
        output = LllWorkerFailure(protocol=PROTOCOL, error_code=exc.code)
        code = 2
    else:
        code = 0
    sys.stdout.buffer.write(canonicalize_json(output.model_dump(mode="json")) + b"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
