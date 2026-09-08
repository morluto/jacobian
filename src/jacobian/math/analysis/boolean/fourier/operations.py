"""Domain-owned Boolean function analysis operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.boolean._models import (
    MAX_WALSH_VARIABLES,
    BooleanRationalVector,
    BooleanTruthTable,
)
from jacobian.math.analysis.boolean.fourier._models import (
    MAX_VARIABLES,
    MIN_VARIABLES,
    ErasureNoiseResult,
    FourierSpectrumResult,
    MultilinearExtensionResult,
    TruthTableResult,
)


def _variable_count(truth_table_len: int) -> int:
    return truth_table_len.bit_length() - 1


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational.

    Kernel outputs are plain ints; constructing those directly skips the
    intermediate Fraction allocation. Canonical inputs make this exactly
    equivalent to ``from_fraction``.
    """
    if isinstance(value, int):
        return CanonicalRational(num=value, den=1)
    return CanonicalRational.from_fraction(value)


def _admit_truth_table(
    truth_table: tuple[CanonicalRational, ...],
    *,
    minimum: int = MIN_VARIABLES,
    maximum: int = MAX_VARIABLES,
) -> int:
    size = len(truth_table)
    if size & (size - 1) != 0:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.truth_table_power",
            message="truth table length must be a power of two",
        )
    variable_count = _variable_count(size)
    if not minimum <= variable_count <= maximum:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.variable_count",
            message=(f"variable count must be between {minimum} and {maximum}"),
        )
    # Canonical rationals are validated reduced with a positive denominator,
    # so the 0/1 check below is exactly ``as_fraction() in (0, 1)`` without
    # allocating a Fraction per entry.
    if any((entry.num, entry.den) not in ((0, 1), (1, 1)) for entry in truth_table):
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean_analysis.truth_table_boolean",
            message="truth table entry must be 0 or 1",
        )
    return variable_count


def truth_table(values: tuple[CanonicalRational, ...]) -> TruthTableResult:
    """Return a source-owned Boolean truth table and its fixed cube axis."""
    _admit_truth_table(values)
    return TruthTableResult(truth_table=BooleanTruthTable(values=values))


def fourier_spectrum(values: tuple[CanonicalRational, ...]) -> FourierSpectrumResult:
    """Compute the exact Walsh-Hadamard (Fourier) spectrum via FWHT."""
    _admit_truth_table(values, minimum=0, maximum=MAX_WALSH_VARIABLES)
    spectrum = _fast_walsh_hadamard_transform(_truth_values(values))
    return FourierSpectrumResult(
        source=BooleanTruthTable(values=values),
        spectrum=BooleanRationalVector(
            values=tuple(_rational(value) for value in spectrum)
        ),
    )


def multilinear_extension(
    values: tuple[CanonicalRational, ...],
) -> MultilinearExtensionResult:
    """Compute the multilinear extension polynomial over the rationals.

    The multilinear extension of ``f: {0,1}^n -> R`` is the unique polynomial
    that agrees with ``f`` on the Boolean hypercube:

        f~(x) = (1/2^n) * sum_S W_f(S) * prod_{i in S} (1 - 2 x_i)

    where ``W_f(S)`` is the Walsh-Hadamard coefficient at subset ``S`` and
    the character ``prod_{i in S} (1 - 2 x_i)`` equals ``(-1)^{<S, x>}``.
    """
    _admit_truth_table(values)
    coefficients = _subset_mobius_transform(_truth_values(values))

    return MultilinearExtensionResult(
        source=BooleanTruthTable(values=values),
        coefficients=BooleanRationalVector(
            values=tuple(_rational(value) for value in coefficients)
        ),
    )


def erasure_noise(
    values: tuple[CanonicalRational, ...],
    probability_value: CanonicalRational,
    base_input: tuple[int, ...],
) -> ErasureNoiseResult:
    """Compute the expected value of f under erasure noise.

    With probability ``p`` each coordinate is kept; with probability ``(1-p)``
    it is replaced by an independent uniform random bit.  By the Fourier
    characterization of erasure noise, the expected value equals
    ``sum_S f_hat(S) * p^|S| * chi_S(x)`` at the supplied base assignment
    ``x``, where ``f_hat(S) = W_f(S) / 2^n``.  All arithmetic is exact rational.
    """
    n = _admit_truth_table(values)
    probability = probability_value.as_fraction()
    if not 0 <= probability <= 1:
        raise OperationDomainValidationError(
            location=("probability",),
            code="boolean_analysis.probability_range",
            message="probability must be in [0, 1]",
        )
    if len(base_input) != n:
        raise OperationDomainValidationError(
            location=("base_input",),
            code="boolean_analysis.base_input_length",
            message="base_input must have one bit per variable",
        )
    if any(bit not in (0, 1) for bit in base_input):
        raise OperationDomainValidationError(
            location=("base_input",),
            code="boolean_analysis.base_input_boolean",
            message="base_input bits must be 0 or 1",
        )
    truth = _truth_values(values)
    spectrum = _fast_walsh_hadamard_transform(truth)
    total = len(truth)
    one_mask = 0
    for bit_idx, bit in enumerate(base_input):
        if bit == 1:
            one_mask |= 1 << bit_idx

    p = probability
    powers = [Fraction(1)] * (n + 1)
    for degree in range(1, n + 1):
        powers[degree] = powers[degree - 1] * p
    result = Fraction(0)
    for subset_mask in range(total):
        sign = -1 if (subset_mask & one_mask).bit_count() % 2 else 1
        fourier_coeff = Fraction(spectrum[subset_mask], total)
        result += sign * fourier_coeff * powers[subset_mask.bit_count()]

    return ErasureNoiseResult(
        source=BooleanTruthTable(values=values),
        expected_value=_rational(result),
        probability=probability_value,
        base_input=base_input,
    )


def verify_fourier_spectrum(claim: FourierSpectrumResult) -> bool:
    """Verify a Fourier spectrum against its retained Boolean function."""

    if not isinstance(claim, FourierSpectrumResult):
        return False
    try:
        return fourier_spectrum(claim.source.values) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_multilinear_extension(claim: MultilinearExtensionResult) -> bool:
    """Verify multilinear coefficients against their retained truth table."""

    if not isinstance(claim, MultilinearExtensionResult):
        return False
    try:
        return multilinear_extension(claim.source.values) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_erasure_noise(claim: ErasureNoiseResult) -> bool:
    """Verify the exact noise expectation against source and base assignment."""

    if not isinstance(claim, ErasureNoiseResult):
        return False
    try:
        return (
            erasure_noise(claim.source.values, claim.probability, claim.base_input)
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def _truth_values(values: tuple[CanonicalRational, ...]) -> list[int]:
    # Callers admit 0/1 canonical entries first, so the numerator is the value.
    return [entry.num for entry in values]


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


def _subset_mobius_transform(values: list[int]) -> list[int]:
    """Return coefficients of the natural-order multilinear extension."""

    coefficients = list(values)
    bit = 1
    while bit < len(coefficients):
        for subset_mask in range(len(coefficients)):
            if subset_mask & bit:
                coefficients[subset_mask] -= coefficients[subset_mask ^ bit]
        bit <<= 1
    return coefficients


__all__ = [
    "erasure_noise",
    "fourier_spectrum",
    "multilinear_extension",
    "truth_table",
    "verify_erasure_noise",
    "verify_fourier_spectrum",
    "verify_multilinear_extension",
]
