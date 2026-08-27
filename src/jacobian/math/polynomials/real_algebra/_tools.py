"""Real algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.real_algebra._models import (
    RootCountRequest,
    RootCountResult,
    SturmChainRequest,
    SturmChainResult,
)
from jacobian.math.polynomials.real_algebra._operations import (
    compute_root_count,
    compute_strict_sublevel_measure,
    compute_sturm_chain,
)
from jacobian.math.polynomials.real_algebra._strict_sublevel_models import (
    StrictSublevelMeasureRequest,
    StrictSublevelMeasureResult,
)


def ra_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


REAL_ALGEBRA_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    ra_operation(
        "polynomial.sturm_chain.compute",
        "Compute an ordinary exact Sturm sequence",
        "Compute SymPy's ordinary Euclidean-remainder Sturm sequence for a "
        "non-constant univariate polynomial with integer coefficients encoded "
        "as canonical rationals with denominator one. The current envelope is "
        "degree at most 32 and coefficients of at most 16 decimal digits.",
        SturmChainRequest,
        SturmChainResult,
        compute_sturm_chain,
        "polynomial",
        "sturm-chain",
        "exact",
        examples=(
            example(
                "cubic",
                "Sturm chain of x^3 - 2x^2 + x - 3.",
                {
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponent": 3},
                            {"coefficient": {"num": "-2", "den": "1"}, "exponent": 2},
                            {"coefficient": {"num": "1", "den": "1"}, "exponent": 1},
                            {"coefficient": {"num": "-3", "den": "1"}, "exponent": 0},
                        ],
                    },
                },
            ),
        ),
    ),
    ra_operation(
        "polynomial.root_count.compute",
        "Count real roots in an interval via Sturm's theorem",
        "Count distinct real roots of a bounded univariate polynomial with "
        "integer coefficients in the closed interval [lower, upper] using SymPy's "
        "ordinary exact Sturm sequence. The current envelope is degree at most "
        "32 and coefficients of at most 16 decimal digits, encoded as canonical "
        "rationals with denominator one.",
        RootCountRequest,
        RootCountResult,
        compute_root_count,
        "polynomial",
        "root-count",
        "exact",
        examples=(
            example(
                "cubic",
                "Count roots of x^3 - 2x^2 + x - 3 in [-10, 10].",
                {
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponent": 3},
                            {"coefficient": {"num": "-2", "den": "1"}, "exponent": 2},
                            {"coefficient": {"num": "1", "den": "1"}, "exponent": 1},
                            {"coefficient": {"num": "-3", "den": "1"}, "exponent": 0},
                        ],
                    },
                    "lower": {"num": "-10", "den": "1"},
                    "upper": {"num": "10", "den": "1"},
                },
            ),
        ),
    ),
    ra_operation(
        "polynomial.real.strict_sublevel_measure.compute",
        "Compute an exact strict polynomial sublevel measure",
        "Return the complete component decomposition and source-bound exact "
        "real-algebraic measure of {x in [lower, upper] : |f(x)| < threshold} "
        "for a canonical univariate polynomial over QQ.",
        StrictSublevelMeasureRequest,
        StrictSublevelMeasureResult,
        compute_strict_sublevel_measure,
        "polynomial",
        "real-algebra",
        "sublevel-set",
        "measure",
        "exact",
        examples=(
            example(
                "quadratic_irrational_length",
                "Measure |x^2| < 2 on [-2, 2], with endpoints at ±sqrt(2).",
                {
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                }
                            ]
                        },
                    },
                    "threshold": {"num": "2", "den": "1"},
                    "lower": {"num": "-2", "den": "1"},
                    "upper": {"num": "2", "den": "1"},
                },
            ),
        ),
    ),
)

TOOLS = REAL_ALGEBRA_OPERATIONS

__all__ = ["TOOLS"]
