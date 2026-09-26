"""Standalone child for cancellation-prone exact action normalization."""

from __future__ import annotations

import sys

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import CanonicalLimits, loads_strict_json
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.proper_hypergeometric_actions.operations import (
    _apply_in_process,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)

_MAX_BYTES = 64 * 1024 * 1024


def _error_payload(error: object, kind: str) -> dict[str, object]:
    diagnostic = next(iter(error.errors()))  # type: ignore[attr-defined]
    return {
        "ok": False,
        "kind": kind,
        "location": list(diagnostic["loc"]),
        "code": diagnostic["type"],
        "message": diagnostic["msg"],
    }


def _run(payload: dict[str, object]) -> dict[str, object]:
    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )

    operator = ShiftOreOperator.model_validate_json(payload["operator"])  # type: ignore[arg-type]
    term = ProperHypergeometricTerm.model_validate_json(payload["term"])  # type: ignore[arg-type]
    try:
        multiplier = _apply_in_process(operator, term)
    except OperationResourceAdmissionError as error:
        return _error_payload(error, "resource")
    except OperationDomainValidationError as error:
        return _error_payload(error, "domain")
    return {"ok": True, "relative_multiplier": multiplier.model_dump_json()}


def main() -> int:
    raw = sys.stdin.buffer.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise SystemExit("hypergeometric action request exceeds its byte bound")
    payload = loads_strict_json(
        raw,
        limits=CanonicalLimits(max_input_bytes=_MAX_BYTES, max_output_bytes=_MAX_BYTES),
    )
    if not isinstance(payload, dict):
        raise SystemExit("malformed hypergeometric action request")
    sys.stdout.buffer.write(encode_worker_result_frame(_run(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
