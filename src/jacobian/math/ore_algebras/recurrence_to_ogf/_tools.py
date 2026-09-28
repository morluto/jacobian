"""Operation declaration for recurrence-to-OGF conversion."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.recurrence_to_ogf._models import (
    RecurrenceOGFEquation,
    RecurrenceOGFEquationRequest,
)
from jacobian.math.ore_algebras.recurrence_to_ogf.operations import (
    polynomial_recurrence_to_ogf_equation,
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
        operation_id="holonomic.shift_recurrence.to_ogf_differential_equation.compute",
        title="Convert a polynomial recurrence to an ordinary generating function equation",
        description=(
            "Return the exact cleared differential equation L(F)=B induced by "
            "sum_i q_i(n)a_(n+i)=0 for n>=0, where F=sum_n a_n*x^n. "
            "The first r coefficients are supplied explicitly for all boundary "
            "terms, with r the largest shift. This transforms the relation and "
            "does not assert that a sequence satisfies it or that F converges."
        ),
        request_type=RecurrenceOGFEquationRequest,
        result_type=RecurrenceOGFEquation,
        run=lambda request: polynomial_recurrence_to_ogf_equation(
            request.recurrence, request.initial_coefficients
        ),
        tags=("holonomic", "p-recursive", "generating-function", "exact"),
        discovery_terms=(
            "convert polynomial recurrence to OGF differential equation",
            "ordinary generating function equation with initial terms",
            "P-recursive recurrence to differential operator",
        ),
        examples=(
            OperationExample(
                name="fibonacci_ogf_equation",
                description=(
                    "Convert a_(n+2)-a_(n+1)-a_n=0 with a_0=a_1=1 to (1-x-x^2)F(x)=1."
                ),
                input={
                    "recurrence": {
                        "variable": "n",
                        "terms": [
                            {"exponent": 0, "coefficient": _coefficient([(0, -1)])},
                            {"exponent": 1, "coefficient": _coefficient([(0, -1)])},
                            {"exponent": 2, "coefficient": _coefficient([(0, 1)])},
                        ],
                    },
                    "initial_coefficients": {"values": ["1", "1"]},
                },
            ),
        ),
    ),
)
