"""Public manifests for structured rational-function transforms."""

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, MathTools, OperationExample

from .structured_models import (
    FormalAntiderivativeResult,
    GlobalResidueResult,
    LogarithmicDifferentialResult,
    RationalFunctionRequest,
    RationalPrimitiveResult,
)
from .structured_operations import (
    formal_antiderivative,
    global_residues,
    logarithmic_differential,
    rational_primitive,
    residue_at_infinity,
)

_EX = {
    "variables": ["x"],
    "numerator": {
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
    },
    "denominator": {
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
            {"coefficient": {"num": "-1", "den": "1"}, "exponents": [0]},
        ]
    },
}


def _d(req: RationalFunctionRequest) -> LogarithmicDifferentialResult:
    return logarithmic_differential(req.function)


def _a(req: RationalFunctionRequest) -> FormalAntiderivativeResult:
    return formal_antiderivative(req.function)


def _g(req: RationalFunctionRequest) -> GlobalResidueResult:
    return global_residues(req.function)


def _r(req: RationalFunctionRequest) -> CanonicalRational:
    return residue_at_infinity(req.function)


def _p(req: RationalFunctionRequest) -> RationalPrimitiveResult:
    return rational_primitive(req.function)


TOOLS: MathTools = (
    MathTool(
        operation_id="rational_function.formal_antiderivative.compute",
        title="Compose rational and logarithmic differential parts",
        description="Return a branch-free formal antiderivative decomposition.",
        request_type=RationalFunctionRequest,
        result_type=FormalAntiderivativeResult,
        run=_a,
        tags=("rational-function", "differential", "exact"),
        examples=(
            OperationExample(
                name="formal_antiderivative",
                description="Compose the formal parts.",
                input={"function": _EX},
            ),
        ),
    ),
    MathTool(
        operation_id="rational_function.logarithmic_differential.compute",
        title="Represent a rational logarithmic differential",
        description="Return structured dlog rows for the square-free Hermite remainder and replay its formal derivative identity; no analytic logarithm branch is selected.",
        request_type=RationalFunctionRequest,
        result_type=LogarithmicDifferentialResult,
        run=_d,
        tags=("rational-function", "dlog", "exact"),
        examples=(
            OperationExample(
                name="simple_logarithmic_pole",
                description="Represent dx/(x-1) as a formal logarithmic differential.",
                input={"function": _EX},
            ),
        ),
    ),
    MathTool(
        operation_id="rational_function.rational_primitive.compute",
        title="Decide rational primitiveness",
        description="Return the exact rational primitive branch or the nonzero Hermite remainder; elementary logarithmic primitives are not confused with rational ones.",
        request_type=RationalFunctionRequest,
        result_type=RationalPrimitiveResult,
        run=_p,
        tags=("rational-function", "primitive", "exact"),
        examples=(
            OperationExample(
                name="rational_primitive",
                description="Decide rational primitiveness.",
                input={"function": _EX},
            ),
        ),
    ),
    MathTool(
        operation_id="rational_function.residue_at_infinity.compute",
        title="Compute the residue at infinity",
        description="Return the exact residue of f(x)dx at infinity using the rational local transform.",
        request_type=RationalFunctionRequest,
        result_type=__import__(
            "jacobian._exact", fromlist=["CanonicalRational"]
        ).CanonicalRational,
        run=_r,
        tags=("rational-function", "residue", "infinity"),
        examples=(
            OperationExample(
                name="residue_infinity",
                description="Compute infinity residue.",
                input={"function": _EX},
            ),
        ),
    ),
    MathTool(
        operation_id="rational_function.global_residues.compute",
        title="Profile finite poles and infinity",
        description="Return every finite algebraic pole represented by its irreducible factor/root index, the exact infinity residue, and the finite-plus-infinity sum identity.",
        request_type=RationalFunctionRequest,
        result_type=GlobalResidueResult,
        run=_g,
        tags=("rational-function", "residue", "exact"),
        examples=(
            OperationExample(
                name="global_residues",
                description="Profile finite residues.",
                input={"function": _EX},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
