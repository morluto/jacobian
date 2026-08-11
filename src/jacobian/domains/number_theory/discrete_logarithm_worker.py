"""Isolated SymPy worker for bounded modular discrete logarithms."""

from __future__ import annotations

import sys

from pydantic import ValidationError
from sympy.ntheory import discrete_log

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.number_theory import DiscreteLogarithmResult
from jacobian.domains.number_theory.discrete_logarithm_protocol import (
    DiscreteLogarithmWorkerRequest,
    DiscreteLogarithmWorkerResult,
)


def main() -> int:
    try:
        request = DiscreteLogarithmWorkerRequest.model_validate(
            loads_strict_json(sys.stdin.buffer.read())
        ).request
        try:
            exponent = int(
                discrete_log(
                    request.modulus,
                    request.target,
                    request.base,
                )
            )
            result = DiscreteLogarithmResult(
                status="SOLVED",
                base=request.base,
                target=request.target,
                modulus=request.modulus,
                discrete_log=exponent,
            )
        except ValueError as exc:
            if "Log does not exist" not in str(exc):
                raise
            result = DiscreteLogarithmResult(
                status="UNSOLVABLE",
                base=request.base,
                target=request.target,
                modulus=request.modulus,
            )
        sys.stdout.buffer.write(
            canonicalize_json(
                DiscreteLogarithmWorkerResult(
                    protocol="jacobian.number-theory.discrete-logarithm.sympy.v1",
                    result=result,
                ).model_dump(mode="json")
            )
        )
        return 0
    except (TypeError, ValueError, ValidationError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
