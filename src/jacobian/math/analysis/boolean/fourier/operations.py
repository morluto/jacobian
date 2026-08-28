"""Domain-owned Boolean function analysis operations."""

from __future__ import annotations

from fractions import Fraction

import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.boolean.fourier._models import (
    MAX_VARIABLES,
    MIN_VARIABLES,
    ErasureNoiseRequest,
    ErasureNoiseResult,
    FourierSpectrumRequest,
    FourierSpectrumResult,
    MultilinearExtensionRequest,
    MultilinearExtensionResult,
    TruthTableRequest,
    TruthTableResult,
)


def _variable_count(truth_table_len: int) -> int:
    return truth_table_len.bit_length() - 1


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    return CanonicalRational.from_fraction(Fraction(value))


def _admit_truth_table(truth_table: tuple[CanonicalRational, ...]) -> int:
    size = len(truth_table)
    if size & (size - 1) != 0:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.truth_table_power",
            message="truth table length must be a power of two",
        )
    variable_count = _variable_count(size)
    if not MIN_VARIABLES <= variable_count <= MAX_VARIABLES:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.variable_count",
            message=(
                f"variable count must be between {MIN_VARIABLES} and {MAX_VARIABLES}"
            ),
        )
    if any(entry.as_fraction() not in (0, 1) for entry in truth_table):
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.truth_table_boolean",
            message="truth table entry must be 0 or 1",
        )
    return variable_count


def compute_truth_table(request: TruthTableRequest) -> TruthTableResult:
    """Return the truth table with variable count metadata."""
    variable_count = _admit_truth_table(request.truth_table)
    return TruthTableResult(
        truth_table=request.truth_table,
        variable_count=variable_count,
    )


def compute_fourier_spectrum(request: FourierSpectrumRequest) -> FourierSpectrumResult:
    """Compute the exact Walsh-Hadamard (Fourier) spectrum via FWHT."""
    variable_count = _admit_truth_table(request.truth_table)
    truth = request.as_int_list()
    spectrum = _fast_walsh_hadamard_transform(truth)
    return FourierSpectrumResult(
        spectrum=tuple(_rational(value) for value in spectrum),
        variable_count=variable_count,
    )


def compute_multilinear_extension(
    request: MultilinearExtensionRequest,
) -> MultilinearExtensionResult:
    """Compute the multilinear extension polynomial over the rationals.

    The multilinear extension of ``f: {0,1}^n -> R`` is the unique polynomial
    that agrees with ``f`` on the Boolean hypercube:

        f~(x) = (1/2^n) * sum_S W_f(S) * prod_{i in S} (1 - 2 x_i)

    where ``W_f(S)`` is the Walsh-Hadamard coefficient at subset ``S`` and
    the character ``prod_{i in S} (1 - 2 x_i)`` equals ``(-1)^{<S, x>}``.
    """
    n = _admit_truth_table(request.truth_table)
    truth = request.as_int_list()
    spectrum = _fast_walsh_hadamard_transform(truth)
    total = len(truth)

    symbols = sympy.symbols(f"x0:{n}")
    if not isinstance(symbols, tuple):
        symbols = (symbols,)

    poly = sympy.Integer(0)
    for subset_mask in range(total):
        weight = sympy.Rational(spectrum[subset_mask], total)
        character = sympy.Integer(1)
        for bit in range(n):
            if subset_mask & (1 << bit):
                character *= 1 - 2 * symbols[bit]
        poly += weight * character

    poly = sympy.expand(poly)
    return MultilinearExtensionResult(
        polynomial=str(poly),
        variable_count=n,
    )


def compute_erasure_noise(request: ErasureNoiseRequest) -> ErasureNoiseResult:
    """Compute the expected value of f under erasure noise.

    With probability ``p`` each coordinate is kept; with probability ``(1-p)``
    it is replaced by an independent uniform random bit.  By the Fourier
    characterization of erasure noise, the expected value equals
    ``sum_S f_hat(S) * p^|S| * chi_S(x)`` at the supplied base assignment
    ``x``, where ``f_hat(S) = W_f(S) / 2^n``.  All arithmetic is exact rational.
    """
    n = _admit_truth_table(request.truth_table)
    probability = request.probability.as_fraction()
    if not 0 <= probability <= 1:
        raise OperationDomainValidationError(
            location=("probability",),
            code="boolean_analysis.probability_range",
            message="probability must be in [0, 1]",
        )
    if len(request.base_input) != n:
        raise OperationDomainValidationError(
            location=("base_input",),
            code="boolean_analysis.base_input_length",
            message="base_input must have one bit per variable",
        )
    if any(bit not in (0, 1) for bit in request.base_input):
        raise OperationDomainValidationError(
            location=("base_input",),
            code="boolean_analysis.base_input_boolean",
            message="base_input bits must be 0 or 1",
        )
    truth = request.as_int_list()
    spectrum = _fast_walsh_hadamard_transform(truth)
    total = len(truth)
    one_mask = 0
    for bit_idx, bit in enumerate(request.base_input):
        if bit == 1:
            one_mask |= 1 << bit_idx

    p = probability
    result = Fraction(0)
    for subset_mask in range(total):
        subset_size = bin(subset_mask).count("1")
        sign = -1 if (bin(subset_mask & one_mask).count("1") % 2) else 1
        fourier_coeff = Fraction(spectrum[subset_mask], total)
        result += sign * fourier_coeff * (p**subset_size)

    return ErasureNoiseResult(
        expected_value=_rational(result),
        variable_count=n,
        probability=request.probability,
    )


def _fast_walsh_hadamard_transform(values: list[int]) -> list[int]:
    """Exact in-place Fast Walsh-Hadamard Transform (Hadamard order).

    Computes ``W[k] = sum_x f(x) * (-1)^{popcount(x & k)}`` for all ``k`` in
    ``{0,1}^n`` using only integer arithmetic.
    """
    n = len(values)
    result = list(values)
    step = 1
    while step < n:
        i = 0
        while i < n:
            for j in range(step):
                a = result[i + j]
                b = result[i + j + step]
                result[i + j] = a + b
                result[i + j + step] = a - b
            i += step * 2
        step *= 2
    return result
