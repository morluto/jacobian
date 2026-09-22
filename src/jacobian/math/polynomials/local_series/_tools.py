"""Public declarations for exact local Laurent-series profiles and arithmetic."""

from collections.abc import Callable
from typing import Any

from jacobian.catalog.models import MathTool, MathTools, OperationExample

from . import arithmetic as native
from ._models import ValuationProfileRequest, ValuationProfileResult
from .arithmetic_models import (
    LaurentBinaryRequest,
    LaurentDeramifyResult,
    LaurentIntegralResult,
    LaurentMultiplyRequest,
    LaurentPowerRequest,
    LaurentPowerResult,
    LaurentPrincipalPartResult,
    LaurentRamifyRequest,
    LaurentResidueResult,
    LaurentScaleRequest,
    LaurentShiftRequest,
    LaurentTruncateRequest,
    LaurentUnaryRequest,
)
from .operations import laurent_valuation_profile
from .values import TruncatedLaurentWindow


def _binary(req: LaurentBinaryRequest) -> TruncatedLaurentWindow:
    return native.add(req.left, req.right)


def _subtract(req: LaurentBinaryRequest) -> TruncatedLaurentWindow:
    return native.subtract(req.left, req.right)


def _multiply(req: LaurentMultiplyRequest) -> TruncatedLaurentWindow:
    return native.multiply(req.left, req.right, req.output_precision)


def _unary(
    req: LaurentUnaryRequest,
    fn: Callable[[TruncatedLaurentWindow], Any],
) -> Any:
    return fn(req.series)


def _example_series() -> dict[str, object]:
    return {
        "variable": "t",
        "center": {"num": "0", "den": "1"},
        "valuation_lower": -1,
        "precision": 2,
        "coefficients": [
            {"num": "1", "den": "1"},
            {"num": "2", "den": "1"},
            {"num": "3", "den": "1"},
        ],
    }


TOOLS: MathTools = (
    MathTool(
        operation_id="local_series.laurent.valuation_profile.compute",
        title="Compute a Laurent valuation profile",
        description="Return the exact retained-prefix valuation without promoting ZERO_AT_PRECISION to a global zero.",
        request_type=ValuationProfileRequest,
        result_type=ValuationProfileResult,
        run=lambda r: laurent_valuation_profile(r.series),
        tags=("local-series", "laurent", "exact"),
        examples=(
            OperationExample(
                name="simple_pole_window",
                description="Profile a three-term Laurent window.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.add.compute",
        title="Add Laurent windows",
        description="Add two exact windows with identical local parent and known exponent range.",
        request_type=LaurentBinaryRequest,
        result_type=TruncatedLaurentWindow,
        run=_binary,
        tags=("local-series", "laurent", "arithmetic"),
        examples=(
            OperationExample(
                name="add_windows",
                description="Add two windows.",
                input={"left": _example_series(), "right": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.subtract.compute",
        title="Subtract Laurent windows",
        description="Subtract two exact windows with identical local parent and known exponent range.",
        request_type=LaurentBinaryRequest,
        result_type=TruncatedLaurentWindow,
        run=_subtract,
        tags=("local-series", "laurent", "arithmetic"),
        examples=(
            OperationExample(
                name="subtract_windows",
                description="Subtract two windows.",
                input={"left": _example_series(), "right": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.multiply.compute",
        title="Multiply Laurent windows",
        description="Compute the exact Cauchy product through an information-theoretically safe precision.",
        request_type=LaurentMultiplyRequest,
        result_type=TruncatedLaurentWindow,
        run=_multiply,
        tags=("local-series", "laurent", "arithmetic"),
        examples=(
            OperationExample(
                name="multiply_windows",
                description="Multiply through exponent zero.",
                input={
                    "left": _example_series(),
                    "right": _example_series(),
                    "output_precision": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.inverse.compute",
        title="Invert a Laurent window",
        description="Invert a nonzero Laurent prefix and retain its exact exponent lattice.",
        request_type=LaurentUnaryRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.inverse(r.series),
        tags=("local-series", "laurent", "inverse"),
        examples=(
            OperationExample(
                name="invert_window",
                description="Invert a nonzero window.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.divide.compute",
        title="Divide Laurent windows",
        description="Divide exact Laurent prefixes after proving the denominator has a nonzero leading coefficient.",
        request_type=LaurentBinaryRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.divide(r.left, r.right),
        tags=("local-series", "laurent", "division"),
        examples=(
            OperationExample(
                name="divide_windows",
                description="Divide two windows.",
                input={"left": _example_series(), "right": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.power.compute",
        title="Power of a Laurent window",
        description="Raise a Laurent prefix to a bounded integer power.",
        request_type=LaurentPowerRequest,
        result_type=LaurentPowerResult,
        run=lambda r: LaurentPowerResult(
            source=r.series,
            exponent=r.exponent,
            result=native.power(r.series, r.exponent, r.output_precision),
        ),
        tags=("local-series", "laurent", "power"),
        examples=(
            OperationExample(
                name="power_window",
                description="Square a window.",
                input={"series": _example_series(), "exponent": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.derivative.compute",
        title="Differentiate a Laurent window",
        description="Apply the exact formal derivative term by term.",
        request_type=LaurentUnaryRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.derivative(r.series),
        tags=("local-series", "laurent", "derivative"),
        examples=(
            OperationExample(
                name="differentiate_window",
                description="Differentiate a window.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.integral.compute",
        title="Integrate a Laurent window formally",
        description="Return the Laurent primitive and explicit residue coefficient of log(t).",
        request_type=LaurentUnaryRequest,
        result_type=LaurentIntegralResult,
        run=lambda r: native.integral(r.series),
        tags=("local-series", "laurent", "integral"),
        examples=(
            OperationExample(
                name="integrate_window",
                description="Integrate a window formally.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.residue.compute",
        title="Extract a Laurent residue",
        description="Return the exact t^-1 coefficient, including zero when that exponent is absent.",
        request_type=LaurentUnaryRequest,
        result_type=LaurentResidueResult,
        run=lambda r: native.residue(r.series),
        tags=("local-series", "laurent", "residue"),
        examples=(
            OperationExample(
                name="residue_window",
                description="Extract a residue.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.principal_part.compute",
        title="Split a Laurent principal part",
        description="Return negative and nonnegative exponent windows with source precision retained.",
        request_type=LaurentUnaryRequest,
        result_type=LaurentPrincipalPartResult,
        run=lambda r: native.principal_part(r.series),
        tags=("local-series", "laurent", "principal-part"),
        examples=(
            OperationExample(
                name="principal_window",
                description="Split a principal part.",
                input={"series": _example_series()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.truncate.compute",
        title="Truncate a Laurent prefix",
        description="Restrict to a subwindow without inventing coefficients.",
        request_type=LaurentTruncateRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.truncate(r.series, r.valuation_lower, r.precision),
        tags=("local-series", "precision"),
        examples=(
            OperationExample(
                name="truncate_window",
                description="Truncate a window.",
                input={
                    "series": _example_series(),
                    "valuation_lower": -1,
                    "precision": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.shift.compute",
        title="Shift Laurent exponents",
        description="Multiply by a local monomial and transport the exact window.",
        request_type=LaurentShiftRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.shift(r.series, r.shift),
        tags=("local-series", "laurent"),
        examples=(
            OperationExample(
                name="shift_window",
                description="Shift a window.",
                input={"series": _example_series(), "shift": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.change_scale.compute",
        title="Change Laurent parameter scale",
        description="Apply t to c times the exact local parameter.",
        request_type=LaurentScaleRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.change_scale(r.series, r.scale),
        tags=("local-series", "transport"),
        examples=(
            OperationExample(
                name="scale_window",
                description="Scale a window.",
                input={"series": _example_series(), "scale": {"num": "2", "den": "1"}},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.deramify.compute",
        title="Deramify a Laurent prefix",
        description="Return the exact preimage when every nonzero exponent lies in the requested image lattice.",
        request_type=LaurentRamifyRequest,
        result_type=LaurentDeramifyResult,
        run=lambda r: native.deramify(r.series, r.ramification),
        tags=("local-series", "ramification"),
        examples=(
            OperationExample(
                name="deramify_window",
                description="Attempt a ramification preimage.",
                input={"series": _example_series(), "ramification": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.laurent.ramify.compute",
        title="Ramify a Laurent window",
        description="Apply t=u^e and transport exact exponents.",
        request_type=LaurentRamifyRequest,
        result_type=TruncatedLaurentWindow,
        run=lambda r: native.ramify(r.series, r.ramification),
        tags=("local-series", "ramification"),
        examples=(
            OperationExample(
                name="ramify_window",
                description="Ramify a window.",
                input={"series": _example_series(), "ramification": 2},
            ),
        ),
    ),
)
__all__ = ["TOOLS", "compute_valuation_profile"]


def compute_valuation_profile(
    request: ValuationProfileRequest,
) -> ValuationProfileResult:
    return laurent_valuation_profile(request.series)
