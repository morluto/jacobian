"""MathTool declarations for Boolean analysis operations."""

from __future__ import annotations

from jacobian.contracts.boolean_analysis import (
    BooleanErasureNoiseRequest,
    BooleanErasureNoiseResult,
    BooleanFourierRequest,
    BooleanFourierResult,
    BooleanInfluenceRequest,
    BooleanInfluenceResult,
    BooleanMultilinearExtensionRequest,
    BooleanMultilinearExtensionResult,
    BooleanTruthTable,
)
from jacobian.domains._examples import example
from jacobian.domains.boolean_analysis.operations import (
    compute_boolean_fourier,
    compute_boolean_influence,
    compute_erasure_noise,
    compute_multilinear_extension,
)
from jacobian.math_tools import MathTool


BOOLEAN_ANALYSIS_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="boolean.fourier.compute",
        version="1",
        title="Compute Walsh-Fourier coefficients of a Boolean function",
        description=(
            "Compute all Walsh-Fourier coefficients of a Boolean "
            "function f: {-1,+1}^n -> {-1,+1} from its truth table."
        ),
        request_type=BooleanFourierRequest,
        result_type=BooleanFourierResult,
        run=compute_boolean_fourier,
        tags=("boolean", "fourier", "walsh", "analysis", "exact"),
        examples=(
            example(
                "and_function",
                "Fourier coefficients of AND on 2 variables.",
                {
                    "truth_table": {
                        "variable_names": ["a", "b"],
                        "values": [-1, -1, -1, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.multilinear_extension.evaluate",
        version="1",
        title="Evaluate the multilinear extension of a Boolean function",
        description=(
            "Evaluate the multilinear extension of a Boolean function "
            "at a point in Z^n."
        ),
        request_type=BooleanMultilinearExtensionRequest,
        result_type=BooleanMultilinearExtensionResult,
        run=compute_multilinear_extension,
        tags=("boolean", "multilinear", "extension", "polynomial", "exact"),
        examples=(
            example(
                "and_at_origin",
                "Evaluate AND(0,0) via multilinear extension.",
                {
                    "truth_table": {
                        "variable_names": ["a", "b"],
                        "values": [-1, -1, -1, 1],
                    },
                    "point": [0, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.influence.compute",
        version="1",
        title="Compute the influence of each variable",
        description=(
            "Compute the influence (number of flippable edges) of each "
            "variable in a Boolean function."
        ),
        request_type=BooleanInfluenceRequest,
        result_type=BooleanInfluenceResult,
        run=compute_boolean_influence,
        tags=("boolean", "influence", "analysis", "exact"),
        examples=(
            example(
                "dictator_influence",
                "Influence of variables in the dictator function.",
                {
                    "truth_table": {
                        "variable_names": ["a", "b"],
                        "values": [-1, 1, -1, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.erasure_noise.compute",
        version="1",
        title="Compute the exact erasure noise expectation E|f(z)|",
        description=(
            "Compute the exact expected absolute value E|f(z)| when k "
            "randomly chosen variables are erased from the input."
        ),
        request_type=BooleanErasureNoiseRequest,
        result_type=BooleanErasureNoiseResult,
        run=compute_erasure_noise,
        tags=("boolean", "erasure", "noise", "analysis", "exact"),
        examples=(
            example(
                "and_1_erasure",
                "E|f(z)| for AND on 2 variables with 1 erasure.",
                {
                    "truth_table": {
                        "variable_names": ["a", "b"],
                        "values": [-1, -1, -1, 1],
                    },
                    "erasure_count": 1,
                },
            ),
        ),
    ),
)

__all__ = ["BOOLEAN_ANALYSIS_OPERATIONS"]
