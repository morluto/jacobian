"""Isolated Python-FLINT worker for row Hermite normal form."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from jacobian.canonical import canonicalize_json, loads_strict_json

PROTOCOL = "jacobian.matrix-lattice-hnf-worker/v1"


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.buffer.write(
        canonicalize_json({"protocol": PROTOCOL, **payload}) + b"\n"
    )


def _integer(value: object) -> int:
    if not isinstance(value, str) or (value.startswith("0") and value != "0"):
        raise ValueError("invalid integer")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("invalid integer")
    return parsed


def _run() -> dict[str, object]:
    payload = loads_strict_json(sys.stdin.buffer.read())
    if not isinstance(payload, dict) or set(payload) != {"protocol", "matrix"}:
        raise ValueError("invalid worker request")
    matrix = payload["matrix"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("invalid matrix")
    if matrix["matrix_schema_version"] != "1" or matrix["domain"] != "ZZ":
        raise ValueError("invalid matrix")
    entries = matrix["entries"]
    if (
        not isinstance(entries, list)
        or not entries
        or not isinstance(entries[0], list)
        or not entries[0]
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise ValueError("invalid matrix")
    integer_entries = [[_integer(value) for value in row] for row in entries]
    flint: Any = importlib.import_module("flint")
    if getattr(flint, "__version__", None) != "0.9.0":
        raise ValueError("unsupported Python-FLINT version")
    if getattr(flint, "__FLINT_VERSION__", None) != "3.6.0":
        raise ValueError("unsupported FLINT version")
    normal_form, transformation = flint.fmpz_mat(integer_entries).hnf(transform=True)
    return {
        "status": "NORMAL_FORM_PRODUCED",
        "backend_version": "0.9.0",
        "flint_library_version": "3.6.0",
        "normal_form": [
            [str(normal_form[row, column]) for column in range(normal_form.ncols())]
            for row in range(normal_form.nrows())
        ],
        "transformation": [
            [
                str(transformation[row, column])
                for column in range(transformation.ncols())
            ]
            for row in range(transformation.nrows())
        ],
    }


def main() -> int:
    try:
        _emit(_run())
    except Exception as exc:  # pragma: no cover - process boundary
        _emit({"status": "ERROR", "error": type(exc).__name__})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
