"""One-shot number-field worker isolated from the MCP request process."""

from __future__ import annotations

import hashlib
import sys

from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.number_theory.number_fields._integral_basis import (
    recognized_integral_basis,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    request = NumberFieldRequest.model_validate_json(
        input_bytes,
        strict=True,
    )
    integral_basis = recognized_integral_basis(request.field)
    if integral_basis is None:
        response: dict[str, object] = {"kind": "invalid"}
    else:
        _ring, field_discriminant, _alpha, _leading = integral_basis
        response = {
            "kind": "complete",
            "discriminant": format_canonical_integer(int(field_discriminant)),
        }
    response["request_digest"] = hashlib.sha256(input_bytes).hexdigest()
    sys.stdout.buffer.write(encode_strict_json(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
