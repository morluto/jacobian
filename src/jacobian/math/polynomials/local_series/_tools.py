"""Public declarations for exact local Laurent-series profiles and arithmetic."""

from collections.abc import Callable
from typing import Any

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.series._models import TruncatedSeries

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
    LaurentToPowerSeriesResult,
    LaurentTruncateRequest,
    LaurentUnaryRequest,
    PuiseuxBinaryRequest,
    PuiseuxResidueResult,
    PuiseuxUnaryRequest,
    RationalFunctionExpansionRequest,
    RationalFunctionExpansionResult,
    RationalFunctionInfinityExpansionRequest,
)
from .newton_polygon import (
    LocalPolynomialInSeries,
    LocalPolynomialNewtonPolygonResult,
    NewtonEdgeCharacteristicRequest,
    NewtonEdgeCharacteristicResult,
    NewtonEdgeCharacteristicRootsResult,
    local_polynomial_newton_polygon,
    newton_edge_characteristic_polynomial,
    newton_edge_characteristic_roots,
)
from .operations import (
    add_puiseux,
    differentiate_puiseux,
    from_power_series,
    inverse_puiseux,
    laurent_valuation_profile,
    multiply_puiseux,
    rational_function_at_infinity,
    rational_function_at_point,
    residue_puiseux,
    subtract_puiseux,
    to_power_series,
)
from .puiseux_values import PuiseuxTerm, TruncatedPuiseuxWindow
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


def _example_puiseux(coefficient: int = 1) -> dict[str, object]:
    return TruncatedPuiseuxWindow(
        variable="t",
        valuation_lower={"num": 0, "den": 1},
        precision={"num": 2, "den": 1},
        ramification_index=2,
        terms=(
            PuiseuxTerm(
                exponent={"num": 1, "den": 2},
                coefficient={"num": coefficient, "den": 1},
            ),
        ),
    ).model_dump(mode="json")


def _puiseux_add(req: PuiseuxBinaryRequest) -> TruncatedPuiseuxWindow:
    return add_puiseux(req.left, req.right)


def _puiseux_subtract(req: PuiseuxBinaryRequest) -> TruncatedPuiseuxWindow:
    return subtract_puiseux(req.left, req.right)


def _puiseux_multiply(req: PuiseuxBinaryRequest) -> TruncatedPuiseuxWindow:
    return multiply_puiseux(req.left, req.right)


def _puiseux_derivative(req: PuiseuxUnaryRequest) -> TruncatedPuiseuxWindow:
    return differentiate_puiseux(req.series)


def _puiseux_inverse(req: PuiseuxUnaryRequest) -> TruncatedPuiseuxWindow:
    return inverse_puiseux(req.series)


def _rational_function_at_point(
    req: RationalFunctionExpansionRequest,
) -> RationalFunctionExpansionResult:
    return rational_function_at_point(req.function, req.center, req.precision)


def _rational_function_at_infinity(
    req: RationalFunctionInfinityExpansionRequest,
) -> RationalFunctionExpansionResult:
    return rational_function_at_infinity(req.function, req.precision)


TOOLS: MathTools = (
    MathTool(
        operation_id="local_series.polynomial.newton_edge_characteristic_roots.compute",
        title="Solve a quadratic Newton edge characteristic equation",
        description=(
            "Compute every distinct exact rational or algebraic root, with its "
            "multiplicity, for a selected Newton edge characteristic polynomial "
            "of degree at most two over QQ. Higher degrees are rejected before "
            "root computation. Roots are possible leading coefficients only; "
            "this operation does not lift a Puiseux branch."
        ),
        request_type=NewtonEdgeCharacteristicRequest,
        result_type=NewtonEdgeCharacteristicRootsResult,
        run=newton_edge_characteristic_roots,
        tags=("local-series", "polynomial", "newton-polygon", "algebraic-roots", "exact"),
        examples=(
            OperationExample(
                name="quadratic_edge_roots",
                description=(
                    "For y^2 - 2t, the edge characteristic polynomial is "
                    "c^2 - 2 and its two exact roots are ±sqrt(2)."
                ),
                input={
                    "polynomial": {
                        "variable": "t",
                        "place": "FINITE",
                        "center": {"num": "0", "den": "1"},
                        "coefficients": [
                            {
                                "y_degree": 0,
                                "series": {
                                    "variable": "t",
                                    "place": "FINITE",
                                    "center": {"num": "0", "den": "1"},
                                    "valuation_lower": 1,
                                    "precision": 2,
                                    "coefficients": [{"num": "-2", "den": "1"}],
                                },
                            },
                            {
                                "y_degree": 2,
                                "series": {
                                    "variable": "t",
                                    "place": "FINITE",
                                    "center": {"num": "0", "den": "1"},
                                    "valuation_lower": 0,
                                    "precision": 1,
                                    "coefficients": [{"num": "1", "den": "1"}],
                                },
                            },
                        ],
                    },
                    "edge_index": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.polynomial.newton_edge_characteristic.compute",
        title="Compute a Newton edge characteristic polynomial",
        description=(
            "For one exact lower Newton edge, transport each on-edge local "
            "series leading coefficient into the characteristic polynomial "
            "over QQ. The result retains its source and edge; it does not "
            "select roots or claim a Puiseux branch."
        ),
        request_type=NewtonEdgeCharacteristicRequest,
        result_type=NewtonEdgeCharacteristicResult,
        run=newton_edge_characteristic_polynomial,
        tags=("local-series", "polynomial", "newton-polygon", "puiseux", "exact"),
        examples=(
            OperationExample(
                name="leading_coefficient_equation",
                description="Extract the characteristic polynomial of the first edge.",
                input={
                    "polynomial": {
                        "variable": "t",
                        "place": "FINITE",
                        "center": {"num": "0", "den": "1"},
                        "coefficients": [
                            {
                                "y_degree": 0,
                                "series": {
                                    "variable": "t",
                                    "place": "FINITE",
                                    "center": {"num": "0", "den": "1"},
                                    "valuation_lower": 2,
                                    "precision": 3,
                                    "coefficients": [{"num": "-1", "den": "1"}],
                                },
                            },
                            {
                                "y_degree": 2,
                                "series": {
                                    "variable": "t",
                                    "place": "FINITE",
                                    "center": {"num": "0", "den": "1"},
                                    "valuation_lower": 0,
                                    "precision": 1,
                                    "coefficients": [{"num": "1", "den": "1"}],
                                },
                            },
                        ],
                    },
                    "edge_index": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.polynomial.newton_polygon.compute",
        title="Compute a local polynomial Newton polygon",
        description=(
            "Compute the exact lower hull of (y degree, t valuation) for a "
            "sparse polynomial over one rational local-series parent. Null rows "
            "are exact zero coefficients; nonzero rows need a retained nonzero "
            "series term so each reported valuation is exact."
        ),
        request_type=LocalPolynomialInSeries,
        result_type=LocalPolynomialNewtonPolygonResult,
        run=local_polynomial_newton_polygon,
        tags=("local-series", "polynomial", "newton-polygon", "exact"),
        examples=(
            OperationExample(
                name="two_edge_local_newton_polygon",
                description="Compute the lower hull for y^2 + t*y + t^3.",
                input={
                    "variable": "t",
                    "place": "FINITE",
                    "center": {"num": "0", "den": "1"},
                    "coefficients": [
                        {
                            "y_degree": 0,
                            "series": {
                                "variable": "t",
                                "place": "FINITE",
                                "center": {"num": "0", "den": "1"},
                                "valuation_lower": 3,
                                "precision": 4,
                                "coefficients": [{"num": "1", "den": "1"}],
                            },
                        },
                        {
                            "y_degree": 1,
                            "series": {
                                "variable": "t",
                                "place": "FINITE",
                                "center": {"num": "0", "den": "1"},
                                "valuation_lower": 1,
                                "precision": 2,
                                "coefficients": [{"num": "1", "den": "1"}],
                            },
                        },
                        {
                            "y_degree": 2,
                            "series": {
                                "variable": "t",
                                "place": "FINITE",
                                "center": {"num": "0", "den": "1"},
                                "valuation_lower": 0,
                                "precision": 1,
                                "coefficients": [{"num": "1", "den": "1"}],
                            },
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.from_power_series.compute",
        title="Embed a truncated power series as a Laurent prefix",
        description=(
            "Embed an exact QQ[[x]]/(x^N) value at center zero into the integral "
            "Laurent carrier, preserving its variable and precision. Leading zero "
            "coefficients normalize into the valuation lower bound."
        ),
        request_type=TruncatedSeries,
        result_type=TruncatedLaurentWindow,
        run=from_power_series,
        tags=("local-series", "formal-series", "conversion", "exact"),
        examples=(
            OperationExample(
                name="power_series_to_laurent",
                description="Embed 2x + 3x^2 + O(x^4) as a Laurent prefix.",
                input={
                    "variable": "x",
                    "truncation_order": 4,
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.to_power_series.compute",
        title="Project a Laurent prefix to a power series",
        description=(
            "Convert an integral Laurent prefix at finite center zero when its "
            "known coefficients contain no negative exponents. Return a structural "
            "HAS_NEGATIVE_EXPONENTS result for a pole; do not discard its principal part."
        ),
        request_type=LaurentUnaryRequest,
        result_type=LaurentToPowerSeriesResult,
        run=lambda request: to_power_series(request.series),
        tags=("local-series", "formal-series", "conversion", "exact"),
        examples=(
            OperationExample(
                name="regular_laurent_to_power_series",
                description="Convert 2x + 3x^2 + O(x^4) to the formal-series carrier.",
                input={
                    "series": {
                        "variable": "x",
                        "place": "FINITE",
                        "center": {"num": "0", "den": "1"},
                        "valuation_lower": 1,
                        "precision": 4,
                        "coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.at_infinity.compute",
        title="Expand a rational function at infinity",
        description=(
            "Return the exact Laurent prefix in the reciprocal parameter t = 1/x "
            "through the exclusive exponent cutoff. The result records its "
            "INFINITY expansion place and retains the omitted tail as unknown. "
            "Requires a canonical univariate QQ rational function and admits at "
            "most 4096 output coefficients."
        ),
        request_type=RationalFunctionInfinityExpansionRequest,
        result_type=RationalFunctionExpansionResult,
        run=_rational_function_at_infinity,
        tags=("local-series", "laurent", "rational-function", "infinity", "exact"),
        examples=(
            OperationExample(
                name="polynomial_at_infinity",
                description="Expand x^2 + 1 in t = 1/x through exponent 3.",
                input={
                    "function": {
                        "variables": ["x"],
                        "numerator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                        "denominator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                }
                            ]
                        },
                    },
                    "precision": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.from_rational_function_at_point.compute",
        title="Expand a rational function at a finite point",
        description=(
            "Return the exact Laurent prefix in t = x - center through the "
            "exclusive exponent cutoff. Requires a canonical univariate QQ "
            "rational function and admits at most 4096 output coefficients. "
            "The omitted tail remains unknown."
        ),
        request_type=RationalFunctionExpansionRequest,
        result_type=RationalFunctionExpansionResult,
        run=_rational_function_at_point,
        tags=("local-series", "laurent", "rational-function", "exact"),
        examples=(
            OperationExample(
                name="simple_pole_at_one",
                description="Expand 1/(x - 1) around its pole through exponent 2.",
                input={
                    "function": {
                        "variables": ["x"],
                        "numerator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                }
                            ]
                        },
                        "denominator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    },
                    "center": {"num": "1", "den": "1"},
                    "precision": 2,
                },
            ),
        ),
    ),
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
    MathTool(
        operation_id="local_series.puiseux.add.compute",
        title="Add Puiseux windows",
        description="Add exact rational-exponent prefixes through their common known cutoff.",
        request_type=PuiseuxBinaryRequest,
        result_type=TruncatedPuiseuxWindow,
        run=_puiseux_add,
        tags=("local-series", "puiseux", "arithmetic"),
        examples=(
            OperationExample(
                name="add_fractional_terms",
                description="Add two windows on the half-integer lattice.",
                input={"left": _example_puiseux(1), "right": _example_puiseux(2)},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.puiseux.subtract.compute",
        title="Subtract Puiseux windows",
        description="Subtract exact rational-exponent prefixes through their common known cutoff.",
        request_type=PuiseuxBinaryRequest,
        result_type=TruncatedPuiseuxWindow,
        run=_puiseux_subtract,
        tags=("local-series", "puiseux", "arithmetic"),
        examples=(
            OperationExample(
                name="cancel_fractional_term",
                description="Subtract equal fractional terms and retain the zero prefix.",
                input={"left": _example_puiseux(), "right": _example_puiseux()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.puiseux.multiply.compute",
        title="Multiply Puiseux windows",
        description="Compute the exact Cauchy product through the precision supported by both omitted tails.",
        request_type=PuiseuxBinaryRequest,
        result_type=TruncatedPuiseuxWindow,
        run=_puiseux_multiply,
        tags=("local-series", "puiseux", "arithmetic"),
        examples=(
            OperationExample(
                name="multiply_cusp_terms",
                description="Multiply two half-power terms.",
                input={"left": _example_puiseux(), "right": _example_puiseux()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.puiseux.derivative.compute",
        title="Differentiate a Puiseux window",
        description="Differentiate rational exponents exactly and lower the known cutoff by one.",
        request_type=PuiseuxUnaryRequest,
        result_type=TruncatedPuiseuxWindow,
        run=_puiseux_derivative,
        tags=("local-series", "puiseux", "derivative"),
        examples=(
            OperationExample(
                name="differentiate_fractional_term",
                description="Differentiate t^(1/2) + O(t^2).",
                input={"series": _example_puiseux()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.puiseux.inverse.compute",
        title="Invert a Puiseux window",
        description="Invert a prefix with a retained nonzero leading term and return only precision justified by its unknown tail.",
        request_type=PuiseuxUnaryRequest,
        result_type=TruncatedPuiseuxWindow,
        run=_puiseux_inverse,
        tags=("local-series", "puiseux", "inverse"),
        examples=(
            OperationExample(
                name="invert_fractional_monomial",
                description="Invert t^(1/2) + O(t^2).",
                input={"series": _example_puiseux()},
            ),
        ),
    ),
    MathTool(
        operation_id="local_series.puiseux.residue.compute",
        title="Extract a Puiseux residue",
        description=(
            "Return the exact t^-1 coefficient when the retained window contains "
            "that exponent; reject windows that leave it unknown."
        ),
        request_type=PuiseuxUnaryRequest,
        result_type=PuiseuxResidueResult,
        run=lambda r: residue_puiseux(r.series),
        tags=("local-series", "puiseux", "residue"),
        examples=(
            OperationExample(
                name="extract_puiseux_residue",
                description="Extract the t^-1 coefficient from a known window.",
                input={
                    "series": TruncatedPuiseuxWindow(
                        variable="t",
                        valuation_lower={"num": -2, "den": 1},
                        precision={"num": 1, "den": 1},
                        ramification_index=1,
                        terms=(
                            PuiseuxTerm(
                                exponent={"num": -1, "den": 1},
                                coefficient={"num": 3, "den": 2},
                            ),
                        ),
                    ).model_dump(mode="json")
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS", "compute_valuation_profile"]


def compute_valuation_profile(
    request: ValuationProfileRequest,
) -> ValuationProfileResult:
    return laurent_valuation_profile(request.series)
