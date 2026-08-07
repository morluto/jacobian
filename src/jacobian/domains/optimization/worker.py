"""Isolated SymPy worker for rational linear optimization."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.validated_analysis import RationalLinearProgramRequest


def _rational(value: CanonicalRational) -> Any:
    import sympy

    return sympy.Rational(value.as_fraction())


def _wire(value: Any) -> dict[str, str]:
    import sympy

    rational = sympy.Rational(value)
    return {
        "num": format_canonical_integer(int(rational.p)),
        "den": format_canonical_integer(int(rational.q)),
    }


def _linear_program(request: RationalLinearProgramRequest) -> dict[str, Any]:
    import sympy
    from sympy.solvers.simplex import (
        InfeasibleLPError,
        UnboundedLPError,
        linprog,
        lpmax,
    )

    program = request.program
    objective = sympy.Matrix([[_rational(value) for value in program.objective]])
    coefficients = sympy.Matrix(
        [[_rational(value) for value in row] for row in program.coefficients]
    )
    rhs = sympy.Matrix([_rational(value) for value in program.rhs])
    try:
        primal_value, primal_values = linprog(
            objective,
            A=coefficients.col_join(-coefficients),
            b=rhs.col_join(-rhs),
        )
    except (InfeasibleLPError, UnboundedLPError) as exc:
        return {
            "status": "NO_CERTIFICATE",
            "detail": (
                "SymPy produced no primal candidate; solver status is not a "
                f"certificate: {type(exc).__name__}."
            ),
        }

    if len(primal_values) != objective.cols:
        raise RuntimeError("SymPy returned a primal candidate with the wrong dimension")
    primal = sympy.Matrix(primal_values)
    residuals = coefficients * primal - rhs
    primal_payload = {
        "primal_candidate": [_wire(value) for value in primal],
        "primal_objective": _wire(primal_value),
        "primal_residuals": [_wire(value) for value in residuals],
    }
    try:
        dual_symbols = sympy.symbols(f"_dual0:{rhs.rows}")
        dual_column = sympy.Matrix(dual_symbols)
        dual_value, dual_solution = lpmax(
            (rhs.transpose() * dual_column)[0],
            [
                sympy.Le(left, right, evaluate=False)
                for left, right in zip(
                    coefficients.transpose() * dual_column,
                    objective.transpose(),
                    strict=True,
                )
            ],
        )
    except (InfeasibleLPError, UnboundedLPError) as exc:
        return {
            "status": "PRIMAL_ONLY",
            **primal_payload,
            "detail": (
                "SymPy produced a primal candidate but no dual candidate; "
                f"solver status is not a certificate: {type(exc).__name__}."
            ),
        }

    dual_values = [dual_solution.get(symbol, sympy.S.Zero) for symbol in dual_symbols]
    if len(dual_values) != rhs.rows:
        raise RuntimeError("SymPy returned a dual candidate with the wrong dimension")
    dual = sympy.Matrix(dual_values)
    slacks = objective.transpose() - coefficients.transpose() * dual
    if (
        any(value < 0 for value in primal)
        or any(value != 0 for value in residuals)
        or any(value < 0 for value in slacks)
        or primal_value != dual_value
    ):
        return {
            "status": "PRIMAL_ONLY",
            **primal_payload,
            "detail": (
                "The maintained solver candidates failed the exact "
                "producer-side primal/dual consistency checks."
            ),
        }
    return {
        "status": "CERTIFICATE_PRODUCED",
        **primal_payload,
        "dual_candidate": [_wire(value) for value in dual],
        "dual_objective": _wire(dual_value),
        "dual_slacks": [_wire(value) for value in slacks],
        "certificate_available": True,
        "detail": (
            "SymPy produced exact primal and dual candidates with equal "
            "objective values; independent replay remains required."
        ),
    }


def main() -> int:
    try:
        payload = loads_strict_json(sys.stdin.buffer.read())
        request = RationalLinearProgramRequest.model_validate(payload)
    except (CanonicalizationError, ValidationError, ValueError):
        sys.stderr.write("invalid rational optimization worker request\n")
        return 2
    sys.stdout.buffer.write(
        canonicalize_json(
            _linear_program(request),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
