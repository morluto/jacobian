"""Exact Boolean function analysis operation declarations."""

from typing import Any

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
from jacobian.math.analysis.boolean.fourier._pullback import WALSH_PULLBACK
from jacobian.math.analysis.boolean.fourier.operations import (
    erasure_noise,
    fourier_spectrum,
    multilinear_extension,
    truth_table,
)


def _run_truth_table(request: TruthTableRequest) -> TruthTableResult:
    return truth_table(request.truth_table)


def _run_fourier_spectrum(request: FourierSpectrumRequest) -> FourierSpectrumResult:
    return fourier_spectrum(request.truth_table)


def _run_multilinear_extension(
    request: MultilinearExtensionRequest,
) -> MultilinearExtensionResult:
    return multilinear_extension(request.truth_table)


def _run_erasure_noise(request: ErasureNoiseRequest) -> ErasureNoiseResult:
    return erasure_noise(request.truth_table, request.probability, request.base_input)


def _z(n: str) -> dict[str, str]:
    return {"num": n, "den": "1"}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    WALSH_PULLBACK,
    MathTool(
        operation_id="boolean.truth_table.compute",
        title="Evaluate a Boolean function over all 2^n inputs",
        description="Return a complete Boolean truth table with its variable count and ordering convention.",
        request_type=TruthTableRequest,
        result_type=TruthTableResult,
        run=_run_truth_table,
        tags=("boolean", "truth-table", "exact"),
        examples=(
            OperationExample(
                name="single_variable",
                description="Return the truth table of a one-variable Boolean function.",
                input={"truth_table": [_z("0"), _z("1")]},
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.fourier_spectrum.compute",
        title="Compute a Boolean Fourier spectrum",
        description="Compute sum_x f(x) (-1)^(s dot x) from the 0/1 values of a complete truth table with 0 through 12 variables. This BOOLEAN_VALUES convention differs from the BOOLEAN_SIGN transform of (-1)^f.",
        request_type=FourierSpectrumRequest,
        result_type=FourierSpectrumResult,
        run=_run_fourier_spectrum,
        tags=("boolean", "fourier", "walsh", "exact"),
        examples=(
            OperationExample(
                name="and_function",
                description="Compute the Walsh spectrum of the two-bit AND function.",
                input={"truth_table": [_z("0"), _z("0"), _z("0"), _z("1")]},
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.multilinear_extension.compute",
        title="Compute the multilinear extension polynomial of a Boolean function",
        description="Compute the unique multilinear polynomial over the rationals that agrees with the Boolean function on {0,1}^n. Returns exact rational coefficients indexed by the subset mask of each monomial, with the ambient variable count.",
        request_type=MultilinearExtensionRequest,
        result_type=MultilinearExtensionResult,
        run=_run_multilinear_extension,
        tags=("boolean", "multilinear", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="single_variable",
                description="Compute the multilinear extension of f(0)=0, f(1)=1 (the identity).",
                input={"truth_table": [_z("0"), _z("1")]},
            ),
        ),
    ),
    MathTool(
        operation_id="boolean.erasure_noise.compute",
        title="Compute the expected value of a Boolean function under erasure noise",
        description="With probability p each coordinate of the supplied base assignment is kept; with probability (1-p) it is replaced by an independent uniform random bit. Returns the exact rational expected value T_p f(x), computed via the Fourier expansion weighted by p^|S| chi_S(x).",
        request_type=ErasureNoiseRequest,
        result_type=ErasureNoiseResult,
        run=_run_erasure_noise,
        tags=("boolean", "noise", "erasure", "fourier", "exact"),
        examples=(
            OperationExample(
                name="single_variable_p_half",
                description="Compute the erasure-noise expected value of f(0)=0, f(1)=1 at the origin with p=1/2.",
                input={
                    "truth_table": [_z("0"), _z("1")],
                    "probability": {"num": "1", "den": "2"},
                    "base_input": [0],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
