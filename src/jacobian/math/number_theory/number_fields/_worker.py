"""One-shot number-field worker isolated from the MCP request process."""

from __future__ import annotations

import json
import sys

from jacobian.math.number_theory.number_fields import discriminant
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest


def main() -> int:
    request = NumberFieldRequest.model_validate(json.load(sys.stdin))
    import sympy

    variable = sympy.Symbol(request.variable)
    polynomial = sympy.Poly.from_list(
        [int(value) for value in request.coefficients_descending],
        gens=variable,
        domain=sympy.ZZ,
    )
    if not polynomial.is_irreducible:
        response: dict[str, object] = {"kind": "invalid"}
    else:
        response = {
            "kind": "complete",
            "discriminant": discriminant(
                request.coefficients_descending, request.variable
            ),
        }
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
