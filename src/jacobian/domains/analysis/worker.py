"""Isolated Arb worker for validated real-function enclosures."""

from __future__ import annotations

import sys

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.contracts.validated_analysis import ExactDyadic
from jacobian.domains.analysis.protocol import (
    PROTOCOL,
    ArbEnclosedWorkerResponse,
    ArbNonfiniteWorkerResponse,
    ArbPointEnclosureWorkerRequest,
    ArbPointEnclosureWorkerResponse,
    parse_arb_worker_request,
)


def _point_enclosure(
    worker_request: ArbPointEnclosureWorkerRequest,
) -> ArbPointEnclosureWorkerResponse:
    from flint import arb, ctx, fmpq

    request = worker_request.request
    numerator, denominator = request.argument.as_integer_ratio()
    with ctx.workprec(request.precision_bits):
        value = arb(fmpq(numerator, denominator))
        result = getattr(value, request.function.value.lower())()
        if not result.is_finite():
            return ArbNonfiniteWorkerResponse(
                protocol=PROTOCOL,
                status="NONFINITE",
            )
        lower_mantissa, lower_exponent = result.lower().man_exp()
        upper_mantissa, upper_exponent = result.upper().man_exp()
        exact = bool(result.is_exact())
        return ArbEnclosedWorkerResponse(
            protocol=PROTOCOL,
            status="ENCLOSED",
            lower=ExactDyadic(
                mantissa=format_canonical_integer(int(lower_mantissa)),
                exponent=int(lower_exponent),
            ),
            upper=ExactDyadic(
                mantissa=format_canonical_integer(int(upper_mantissa)),
                exponent=int(upper_exponent),
            ),
            relative_accuracy_bits=(None if exact else int(result.rel_accuracy_bits())),
            exact=exact,
        )


def main() -> int:
    try:
        worker_request = parse_arb_worker_request(
            loads_strict_json(sys.stdin.buffer.read())
        )
        result = _point_enclosure(worker_request)
    except (CanonicalizationError, TypeError, ValueError, ValidationError):
        sys.stderr.write("validated analysis worker request or execution failed\n")
        return 2
    sys.stdout.buffer.write(canonicalize_json(result.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
