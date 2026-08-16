"""Exact Boolean function analysis operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.boolean_analysis import (
    ErasureNoiseRequest,
    ErasureNoiseResult,
    FourierSpectrumRequest,
    FourierSpectrumResult,
    MultilinearExtensionRequest,
    MultilinearExtensionResult,
    TruthTableRequest,
    TruthTableResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.boolean_analysis.operations import (
    compute_erasure_noise,
    compute_fourier_spectrum,
    compute_multilinear_extension,
    compute_truth_table,
)
from jacobian.math_tools import MathTool


def boolean_analysis_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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


BOOLEAN_ANALYSIS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    boolean_analysis_operation(
        "boolean.truth_table.compute",
        "Evaluate a Boolean function over all 2^n inputs",
        "Given a truth table of 2^n 0/1 values, return it together with the variable count and ordering convention. Inputs are canonical rationals; the result echoes the table in canonical form.",
        TruthTableRequest,
        TruthTableResult,
        compute_truth_table,
        "boolean",
        "truth-table",
        "exact",
        examples=(
            example(
                "single_variable",
                "Return the truth table of a 1-variable Boolean function.",
                {"truth_table": [_z("0"), _z("1")]},
            ),
        ),
    ),
    boolean_analysis_operation(
        "boolean.fourier_spectrum.compute",
        "Compute the Fourier/Walsh-Hadamard spectrum of a Boolean function",
        "Compute the exact integer Walsh-Hadamard spectrum of a Boolean function from its complete truth table using the Fast Walsh-Hadamard Transform. No floating-point arithmetic is involved.",
        FourierSpectrumRequest,
        FourierSpectrumResult,
        compute_fourier_spectrum,
        "boolean",
        "fourier",
        "walsh",
        "hadamard",
        "exact-integer",
        examples=(
            example(
                "and_function",
                "Compute the Walsh spectrum of the 2-bit AND function.",
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
