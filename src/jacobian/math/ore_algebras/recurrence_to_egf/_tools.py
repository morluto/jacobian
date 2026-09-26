"""Operation declaration for recurrence-to-EGF conversion."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.recurrence_to_egf._models import (
    RecurrenceEGFEquation,
    RecurrenceEGFEquationRequest,
)
from jacobian.math.ore_algebras.recurrence_to_egf.operations import (
    polynomial_recurrence_to_egf_equation,
)


def _coefficient(terms: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["n"],
        "numerator": {
            "terms": [
                {
                    "coefficient": {"num": str(value), "den": "1"},
                    "exponents": [degree],
                }
                for degree, value in terms
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="holonomic.shift_recurrence.to_egf_differential_equation.compute",
        title="Convert a polynomial recurrence to an exponential generating function equation",
        description=(
            "Return the exact homogeneous polynomial differential equation "
            "sum_i q_i(xD)D^i E=0 induced by sum_i q_i(n)a_(n+i)=0 for n>=0, "
            "where E=sum_n a_n*x^n/n!. Differentiating the EGF shifts its "
            "coefficient index directly, so no initial-term forcing appears. "
            "This transforms a relation and does not assert sequence satisfaction "
            "or analytic convergence."
        ),
        request_type=RecurrenceEGFEquationRequest,
        result_type=RecurrenceEGFEquation,
        run=lambda request: polynomial_recurrence_to_egf_equation(request.recurrence),
        tags=("holonomic", "p-recursive", "exponential-generating-function", "exact"),
        discovery_terms=(
            "convert polynomial recurrence to EGF differential equation",
            "exponential generating function equation from recurrence",
            "P-recursive sequence exponential generating function operator",
        ),
        examples=(
            OperationExample(
                name="fibonacci_egf_equation",
                description=(
                    "Convert a_(n+2)-a_(n+1)-a_n=0 to E''-E'-E=0 for "
                    "E(x)=sum a_n*x^n/n!; the recurrence coefficients must "
                    "be polynomials in QQ[n]."
                ),
                input={
                    "recurrence": {
                        "variable": "n",
                        "terms": [
                            {"exponent": 0, "coefficient": _coefficient([(0, -1)])},
                            {"exponent": 1, "coefficient": _coefficient([(0, -1)])},
                            {"exponent": 2, "coefficient": _coefficient([(0, 1)])},
                        ],
                    }
                },
            ),
        ),
    ),
)
