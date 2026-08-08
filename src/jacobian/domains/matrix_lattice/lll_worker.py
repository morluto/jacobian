"""Isolated Python-FLINT worker for exact integer LLL reduction."""

from __future__ import annotations

import importlib
import re
import sys
from typing import Any

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.matrices import MAX_MATRIX_SCALAR_DIGITS
from jacobian.contracts.matrix_operations import (
    MAX_INPUT_SCALAR_DIGITS,
)

PROTOCOL = "jacobian.flint-lll-worker/v1"
INPUT_LIMIT = 1_000_000
MAX_DIMENSION = 32
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


class WorkerError(RuntimeError):
    pass


def _integer(value: object) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > MAX_INPUT_SCALAR_DIGITS
    ):
        raise WorkerError("FLINT_LLL_MATRIX_INVALID")
    return int(value)


def _read() -> list[list[int]]:
    encoded = sys.stdin.buffer.read(INPUT_LIMIT + 1)
    if len(encoded) > INPUT_LIMIT:
        raise WorkerError("FLINT_LLL_INPUT_LIMIT_EXCEEDED")
    try:
        payload = loads_strict_json(encoded)
    except CanonicalizationError as exc:
        raise WorkerError("INVALID_REQUEST") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"protocol", "basis"}
        or payload["protocol"] != PROTOCOL
    ):
        raise WorkerError("FLINT_LLL_INPUT_INVALID")
    basis = payload["basis"]
    if not isinstance(basis, dict) or set(basis) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise WorkerError("FLINT_LLL_MATRIX_INVALID")
    entries = basis["entries"]
    if (
        basis["matrix_schema_version"] != "1"
        or basis["domain"] != "ZZ"
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_DIMENSION
        or not isinstance(entries[0], list)
        or not 1 <= len(entries[0]) <= MAX_DIMENSION
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise WorkerError("FLINT_LLL_MATRIX_INVALID")
    return [[_integer(value) for value in row] for row in entries]


def _wire(matrix: Any) -> list[list[str]]:
    entries = [
        [str(matrix[row, column]) for column in range(matrix.ncols())]
        for row in range(matrix.nrows())
    ]
    if any(
        _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > MAX_MATRIX_SCALAR_DIGITS
        for row in entries
        for value in row
    ):
        raise WorkerError("FLINT_LLL_OUTPUT_LIMIT_EXCEEDED")
    return entries


def _run(entries: list[list[int]]) -> dict[str, object]:
    try:
        flint: Any = importlib.import_module("flint")
        if (
            getattr(flint, "__version__", None) != "0.9.0"
            or getattr(flint, "__FLINT_VERSION__", None) != "3.6.0"
        ):
            raise WorkerError("FLINT_LLL_VERSION_MISMATCH")
        source = flint.fmpz_mat(entries)
        reduced, transformation = source.lll(
            transform=True,
            delta=0.99,
            eta=0.51,
            rep="zbasis",
            gram="exact",
        )
        if transformation * source != reduced:
            raise WorkerError("FLINT_LLL_RELATION_INVALID")
    except WorkerError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise WorkerError("FLINT_LLL_EXECUTION_FAILED") from None
    return {
        "protocol": PROTOCOL,
        "reduced_basis": _wire(reduced),
        "transformation": _wire(transformation),
        "rank": int(reduced.rank()),
    }


def main() -> int:
    try:
        output = _run(_read())
    except WorkerError as exc:
        output = {"protocol": PROTOCOL, "error_code": str(exc)}
        code = 2
    else:
        code = 0
    sys.stdout.buffer.write(canonicalize_json(output) + b"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
