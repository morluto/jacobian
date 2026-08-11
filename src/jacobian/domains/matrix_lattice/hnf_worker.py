"""Isolated Python-FLINT worker for row Hermite normal form."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.contracts.matrices import IntegerMatrix
from jacobian.contracts.matrix_lattice import HermiteNormalFormResult
from jacobian.domains.matrix_lattice.hnf_protocol import (
    PROTOCOL,
    HermiteNormalFormWorkerFailure,
    HermiteNormalFormWorkerRequest,
    HermiteNormalFormWorkerResponse,
    parse_hnf_worker_request,
)


def _matrix(value: Any) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(int(value[row, column]))
                for column in range(value.ncols())
            )
            for row in range(value.nrows())
        )
    )


def _run(
    worker_request: HermiteNormalFormWorkerRequest,
) -> HermiteNormalFormWorkerResponse:
    integer_entries = [
        [parse_canonical_integer(value) for value in row]
        for row in worker_request.request.matrix.entries
    ]
    flint: Any = importlib.import_module("flint")
    if getattr(flint, "__version__", None) != "0.9.0":
        raise ValueError("unsupported Python-FLINT version")
    if getattr(flint, "__FLINT_VERSION__", None) != "3.6.0":
        raise ValueError("unsupported FLINT version")
    normal_form, transformation = flint.fmpz_mat(integer_entries).hnf(transform=True)
    return HermiteNormalFormWorkerResponse(
        protocol=PROTOCOL,
        status="NORMAL_FORM_PRODUCED",
        result=HermiteNormalFormResult(
            normal_form=_matrix(normal_form),
            transformation=_matrix(transformation),
        ),
    )


def main() -> int:
    worker_request: HermiteNormalFormWorkerRequest | None = None
    response: HermiteNormalFormWorkerResponse | HermiteNormalFormWorkerFailure
    try:
        worker_request = parse_hnf_worker_request(
            loads_strict_json(sys.stdin.buffer.read())
        )
        response = _run(worker_request)
        code = 0
    except (CanonicalizationError, ValidationError, ValueError):
        response = HermiteNormalFormWorkerFailure(
            protocol=PROTOCOL,
            status="ERROR",
            error_code="INVALID_REQUEST",
        )
        code = 2
    except Exception:  # pragma: no cover - process boundary
        response = HermiteNormalFormWorkerFailure(
            protocol=PROTOCOL,
            status="ERROR",
            error_code=(
                "INVALID_REQUEST" if worker_request is None else "EXECUTION_FAILED"
            ),
        )
        code = 2
    sys.stdout.buffer.write(canonicalize_json(response.model_dump(mode="json")) + b"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
