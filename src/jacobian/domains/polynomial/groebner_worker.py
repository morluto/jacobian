"""Isolated SymPy worker for one bounded Gröbner-basis computation."""

from __future__ import annotations

import sys

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.domains.polynomial.groebner_protocol import (
    GroebnerWorkerRequest,
    GroebnerWorkerResultLimitError,
    GroebnerWorkerResultLimitExceeded,
    GroebnerWorkerSuccess,
)
from jacobian.domains.polynomial.operations import (
    PolynomialOutputBudgetError,
    polynomial_groebner_basis,
)


def main() -> int:
    try:
        request = GroebnerWorkerRequest.model_validate(
            loads_strict_json(sys.stdin.buffer.read())
        ).request
        try:
            result = polynomial_groebner_basis(request)
        except PolynomialOutputBudgetError as error:
            sys.stdout.buffer.write(
                canonicalize_json(
                    GroebnerWorkerResultLimitExceeded(
                        protocol="jacobian.polynomial.groebner.sympy.v1",
                        error=GroebnerWorkerResultLimitError(
                            code="POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED",
                            message=str(error),
                        ),
                    ).model_dump(mode="json")
                )
            )
            return 0
        sys.stdout.buffer.write(
            canonicalize_json(
                GroebnerWorkerSuccess(
                    protocol="jacobian.polynomial.groebner.sympy.v1",
                    result=result,
                ).model_dump(mode="json")
            )
        )
        return 0
    except (TypeError, ValueError, ValidationError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
