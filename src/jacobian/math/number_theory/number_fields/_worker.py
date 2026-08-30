"""One-shot number-field worker isolated from the MCP request process."""

from __future__ import annotations

import json
import sys

from jacobian.math.number_theory.number_fields._integral_basis import (
    recognized_integral_basis,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest


def main() -> int:
    request = NumberFieldRequest.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    integral_basis = recognized_integral_basis(request.field)
    if integral_basis is None:
        response: dict[str, object] = {"kind": "invalid"}
    else:
        _ring, field_discriminant, _alpha, _leading = integral_basis
        response = {
            "kind": "complete",
            "discriminant": str(field_discriminant),
        }
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
