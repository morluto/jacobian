"""Domain adapter for Boolean function analysis operations."""

from __future__ import annotations

from fractions import Fraction

import sympy

from jacobian.canonical import format_canonical_integer
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
from jacobian.contracts.exact import CanonicalRational


def _variable_count(truth_table_len: int) -> int:
    return truth_table_len.bit_length() - 1


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    frac = Fraction(value)
    return CanonicalRational(
        num=format_canonical_integer(frac.numerator),
        den=format_canonical_integer(frac.denominator),
    )


def compute_truth_table(request: TruthTableRequest) -> TruthTableResult:
    """Return the truth table with variable count metadata."""
    return TruthTableResult(
        truth_table=request.truth_table,
        variable_count=_variable_count(len(request.truth_table)),
    )


def compute_fourier_spectrum(request: FourierSpectrumRequest) -> FourierSpectrumResult:
    """Compute the exact Walsh-Hadamard (Fourier) spectrum via FWHT."""
    truth = request.as_int_list()
    spectrum = _fast_walsh_hadamard_transform(truth)
    variable_count = _variable_count(len(truth))
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
    truth = request.as_int_list()
    n = _variable_count(len(truth))
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
    truth = request.as_int_list()
    n = _variable_count(len(truth))
    spectrum = _fast_walsh_hadamard_transform(truth)
    total = len(truth)
    one_mask = 0
    for bit_idx, bit in enumerate(request.base_input):
        if bit == 1:
            one_mask |= 1 << bit_idx

    p = request.probability.as_fraction()
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
