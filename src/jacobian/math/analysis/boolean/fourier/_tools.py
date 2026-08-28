"""Exact Boolean function analysis operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.boolean.fourier._models import (
    ErasureNoiseRequest,
    ErasureNoiseResult,
    FourierSpectrumRequest,
    FourierSpectrumResult,
    MultilinearExtensionRequest,
    MultilinearExtensionResult,
    TruthTableRequest,
    TruthTableResult,
)
from jacobian.math.analysis.boolean.fourier._operations import (
    compute_erasure_noise,
    compute_fourier_spectrum,
    compute_multilinear_extension,
    compute_truth_table,
)


def boolean_analysis_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


def _z(n: str) -> dict[str, str]:
    return {"num": n, "den": "1"}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    boolean_analysis_operation(
        "boolean.truth_table.compute",
        "Evaluate a Boolean function over all 2^n inputs",
        "Return a complete Boolean truth table with its variable count and ordering convention.",
        TruthTableRequest,
        TruthTableResult,
        compute_truth_table,
        "boolean",
        "truth-table",
        "exact",
        examples=(
            example(
                "single_variable",
                "Return the truth table of a one-variable Boolean function.",
                {"truth_table": [_z("0"), _z("1")]},
            ),
        ),
    ),
    boolean_analysis_operation(
        "boolean.fourier_spectrum.compute",
        "Compute a Boolean Fourier spectrum",
        "Compute the exact Walsh-Hadamard spectrum of a Boolean function from its complete truth table.",
        FourierSpectrumRequest,
        FourierSpectrumResult,
        compute_fourier_spectrum,
        "boolean",
        "fourier",
        "walsh",
        "exact",
        examples=(
            example(
                "and_function",
                "Compute the Walsh spectrum of the two-bit AND function.",
                {"truth_table": [_z("0"), _z("0"), _z("0"), _z("1")]},
            ),
        ),
    ),
    boolean_analysis_operation(
        "boolean.multilinear_extension.compute",
        "Compute the multilinear extension polynomial of a Boolean function",
        "Compute the unique multilinear polynomial over the rationals that agrees with the Boolean function on {0,1}^n. Returns a canonical SymPy polynomial string.",
        MultilinearExtensionRequest,
        MultilinearExtensionResult,
        compute_multilinear_extension,
        "boolean",
        "multilinear",
        "polynomial",
        "exact",
        examples=(
            example(
                "single_variable",
                "Compute the multilinear extension of f(0)=0, f(1)=1 (the identity).",
                {"truth_table": [_z("0"), _z("1")]},
            ),
        ),
    ),
    boolean_analysis_operation(
        "boolean.erasure_noise.compute",
        "Compute the expected value of a Boolean function under erasure noise",
        "With probability p each coordinate of the supplied base assignment is kept; with probability (1-p) it is replaced by an independent uniform random bit. Returns the exact rational expected value T_p f(x), computed via the Fourier expansion weighted by p^|S| chi_S(x).",
        ErasureNoiseRequest,
        ErasureNoiseResult,
        compute_erasure_noise,
        "boolean",
        "noise",
        "erasure",
        "fourier",
        "exact",
        examples=(
            example(
                "single_variable_p_half",
                "Compute the erasure-noise expected value of f(0)=0, f(1)=1 at the origin with p=1/2.",
                {
                    "truth_table": [_z("0"), _z("1")],
                    "probability": {"num": "1", "den": "2"},
                    "base_input": [0],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
