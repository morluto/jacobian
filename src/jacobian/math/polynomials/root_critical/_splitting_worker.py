"""Killable splitting-field kernel worker.

This module runs as a standalone child process: it reads one canonical JSON
request on stdin, computes either the exact splitting-field value or the
splitting-field-bound distance profile, and writes canonical JSON on stdout.
The parent can kill the child at the request deadline instead of waiting for an
unbounded SymPy call to return.
"""

from __future__ import annotations

import sys
from typing import Any, cast

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import (
    CanonicalLimits,
    loads_strict_json,
)
from jacobian.math.polynomials.values import RationalPolynomial

_MAX_BYTES = 64 * 1024 * 1024


def _run(payload: dict[str, object]) -> dict[str, object]:
    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )
    from jacobian.math.polynomials.root_critical._splitting import (
        compute_bound_profile,
        compute_splitting_field,
    )

    polynomial = RationalPolynomial.model_validate_json(
        cast(str, payload["polynomial"])
    )
    mode = payload["mode"]
    embedding_index = int(cast(Any, payload["embedding_index"]))
    try:
        if mode == "field":
            data = compute_splitting_field(polynomial, embedding_index)
            return {"ok": True, "field": _jsonable(data)}
        if mode == "bind":
            data = compute_bound_profile(
                polynomial,
                embedding_index=embedding_index,
                max_pair_rows=payload["max_pair_rows"],
            )
            profile = data.pop("profile")
            return {
                "ok": True,
                "field": _jsonable(data),
                "profile": cast(Any, profile).model_dump(mode="json"),
            }
    except OperationResourceAdmissionError as error:
        return _error_payload(error, "resource")
    except OperationDomainValidationError as error:
        return _error_payload(error, "domain")
    return {
        "ok": False,
        "kind": "domain",
        "code": "unknown_mode",
        "message": "unknown splitting-field worker mode",
    }


def _jsonable(value: object) -> object:
    """Flatten nested models and drop internal keys from a field-data mapping."""

    if isinstance(value, dict):
        return {
            key: _jsonable(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _error_payload(error: object, kind: str) -> dict[str, object]:
    diagnostic = next(iter(error.errors()))  # type: ignore[attr-defined]
    return {
        "ok": False,
        "kind": kind,
        "location": list(diagnostic["loc"]),
        "code": diagnostic["type"],
        "message": diagnostic["msg"],
    }


def main() -> int:
    raw = sys.stdin.buffer.read()
    payload = loads_strict_json(
        raw,
        limits=CanonicalLimits(max_input_bytes=_MAX_BYTES, max_output_bytes=_MAX_BYTES),
    )
    if not isinstance(payload, dict):
        raise SystemExit("malformed splitting-field kernel request")
    result = _run(payload)
    sys.stdout.buffer.write(encode_worker_result_frame(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
