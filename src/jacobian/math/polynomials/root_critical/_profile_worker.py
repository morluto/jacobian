"""Killable root--critical distance-profile kernel worker.

This module runs as a standalone child process: it reads one canonical JSON
request on stdin, computes the exact root--critical distance profile with the
SymPy kernel, and writes the canonical profile JSON on stdout. The parent can
kill the child at the request deadline instead of waiting for an unbounded
SymPy ``factor_list``, ``all_roots``, or ``minpoly`` call to return.

The worker computes with the request-free in-process entry point; the parent
re-validates the returned profile, so a killed or malformed worker never
establishes a mathematical result.
"""

from __future__ import annotations

import sys

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.math.polynomials.values import RationalPolynomial

_MAX_BYTES = 64 * 1024 * 1024


def _run(payload: dict[str, object]) -> dict[str, object]:
    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )
    from jacobian.math.polynomials.root_critical.operations import (
        _compute_profile,
    )

    polynomial = RationalPolynomial.model_validate_json(payload["polynomial"])  # type: ignore[arg-type]
    try:
        profile = _compute_profile(
            polynomial,
            max_pair_rows=payload["max_pair_rows"],
        )
    except OperationResourceAdmissionError as error:
        return _error_payload(error, "resource")
    except OperationDomainValidationError as error:
        return _error_payload(error, "domain")
    return {"ok": True, "profile": profile.model_dump_json()}


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
        raise SystemExit("malformed root-critical kernel request")
    result = _run(payload)
    sys.stdout.buffer.write(encode_strict_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
