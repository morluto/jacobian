"""Provider-independent native values for finite-table probability."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

MAX_FINITE_JOINT_TABLE_ROWS = 16
MAX_FINITE_JOINT_TABLE_COLUMNS = 16
MAX_FINITE_JOINT_TABLE_CELLS = 64
MAX_INPUT_RATIONAL_DIGITS = 256
MAX_MUTUAL_INFORMATION_SCALE_BITS = 1_024
MAX_MUTUAL_INFORMATION_POWER_COST_BITS = 32_768
MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS = (
    MAX_MUTUAL_INFORMATION_POWER_COST_BITS * 30103 // 100000 + 1
)

# These are derived from the largest producer support values.  A marginal is
# a sum of at most 16 input rationals; a likelihood ratio divides one input
# cell by two such marginals.  The small slack covers decimal addition.
MAX_MUTUAL_INFORMATION_MARGINAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_FINITE_JOINT_TABLE_ROWS + 2
)
MAX_MUTUAL_INFORMATION_LIKELIHOOD_RATIO_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * (1 + 2 * MAX_FINITE_JOINT_TABLE_ROWS) + 4
)
_MAX_INPUT_RATIONAL_MAGNITUDE = 10**MAX_INPUT_RATIONAL_DIGITS


def _require_native_labels(labels: tuple[str, ...], maximum: int, axis: str) -> None:
    if not 1 <= len(labels) <= maximum:
        raise ValueError(f"joint-table {axis} count lies outside the supported bound")
    if any(type(label) is not str or not label for label in labels):
        raise ValueError(f"joint-table {axis} labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"joint-table {axis} labels must be unique")


def _require_native_probability_shape(
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    probabilities: tuple[tuple[object, ...], ...],
) -> None:
    if len(probabilities) != len(row_labels):
        raise ValueError("joint-table row count must match row labels")
    if any(len(row) != len(column_labels) for row in probabilities):
        raise ValueError("joint-table rows must match column labels")
    if len(row_labels) * len(column_labels) > MAX_FINITE_JOINT_TABLE_CELLS:
        raise ValueError("joint table exceeds the bounded cell count")


def _require_native_probability_values(
    probabilities: tuple[tuple[object, ...], ...],
) -> None:
    total = Fraction()
    for row in probabilities:
        for probability in row:
            if type(probability) is not Fraction:
                raise TypeError("native joint-table probabilities must use Fractions")
            if (
                abs(probability.numerator) >= _MAX_INPUT_RATIONAL_MAGNITUDE
                or probability.denominator >= _MAX_INPUT_RATIONAL_MAGNITUDE
            ):
                raise ValueError(
                    "joint-table probability exceeds the "
                    f"{MAX_INPUT_RATIONAL_DIGITS}-digit bound"
                )
            if probability < 0:
                raise ValueError("joint-table probabilities must be nonnegative")
            total += probability
    if total != 1:
        raise ValueError("joint-table probabilities must sum exactly to 1")


@dataclass(frozen=True, slots=True)
class FiniteJointTable:
    """One bounded normalized joint table over native ``Fraction`` values."""

    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    probabilities: tuple[tuple[Fraction, ...], ...]
    log_base: int = 2

    def __post_init__(self) -> None:
        _require_native_labels(self.row_labels, MAX_FINITE_JOINT_TABLE_ROWS, "row")
        _require_native_labels(
            self.column_labels,
            MAX_FINITE_JOINT_TABLE_COLUMNS,
            "column",
        )
        _require_native_probability_shape(
            self.row_labels,
            self.column_labels,
            self.probabilities,
        )
        if type(self.log_base) is not int or not 2 <= self.log_base <= 36:
            raise ValueError("mutual-information log base must lie from 2 through 36")
        _require_native_probability_values(self.probabilities)


@dataclass(frozen=True, slots=True)
class MutualInformationTerm:
    """One positive joint-mass contribution and its exact likelihood ratio."""

    row_index: int
    column_index: int
    probability: Fraction
    row_marginal: Fraction
    column_marginal: Fraction
    likelihood_ratio: Fraction

    def __post_init__(self) -> None:
        if type(self.row_index) is not int or self.row_index < 0:
            raise ValueError("mutual-information row index must be nonnegative")
        if type(self.column_index) is not int or self.column_index < 0:
            raise ValueError("mutual-information column index must be nonnegative")
        values = (
            self.probability,
            self.row_marginal,
            self.column_marginal,
            self.likelihood_ratio,
        )
        if any(type(value) is not Fraction for value in values):
            raise TypeError("native mutual-information terms must use Fractions")
        if self.probability <= 0:
            raise ValueError("mutual-information support contains nonpositive mass")
        if self.row_marginal <= 0 or self.column_marginal <= 0:
            raise ValueError("positive joint mass must have positive marginal support")
        expected = self.probability / (self.row_marginal * self.column_marginal)
        if self.likelihood_ratio != expected:
            raise ValueError("mutual-information likelihood ratio is inconsistent")


def _require_native_result_shape(
    row_marginals: tuple[Fraction, ...],
    column_marginals: tuple[Fraction, ...],
    positive_support: tuple[MutualInformationTerm, ...],
) -> None:
    if not 1 <= len(row_marginals) <= MAX_FINITE_JOINT_TABLE_ROWS:
        raise ValueError("row marginals lie outside the supported bound")
    if not 1 <= len(column_marginals) <= MAX_FINITE_JOINT_TABLE_COLUMNS:
        raise ValueError("column marginals lie outside the supported bound")
    if not 1 <= len(positive_support) <= MAX_FINITE_JOINT_TABLE_CELLS:
        raise ValueError("positive support lies outside the supported bound")
    if any(type(value) is not Fraction for value in row_marginals):
        raise TypeError("native row marginals must be Fractions")
    if any(type(value) is not Fraction for value in column_marginals):
        raise TypeError("native column marginals must be Fractions")


def _small_prime_factorization(value: int) -> dict[int, int]:
    remaining = value
    factors: dict[int, int] = {}
    prime = 2
    while prime * prime <= remaining:
        while remaining % prime == 0:
            factors[prime] = factors.get(prime, 0) + 1
            remaining //= prime
        prime += 1
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def _valuations(value: int, primes: tuple[int, ...]) -> tuple[dict[int, int], int]:
    remaining = value
    exponents: dict[int, int] = {}
    for prime in primes:
        exponent = 0
        while remaining > 1 and remaining % prime == 0:
            remaining //= prime
            exponent += 1
        exponents[prime] = exponent
    return exponents, remaining


def _rational_base_exponent(value: Fraction, base: int) -> Fraction | None:
    """Return ``q`` exactly when ``value == base**q`` for rational ``q``."""

    base_factors = _small_prime_factorization(base)
    primes = tuple(base_factors)
    numerator_exponents, numerator_remainder = _valuations(value.numerator, primes)
    denominator_exponents, denominator_remainder = _valuations(
        value.denominator,
        primes,
    )
    if numerator_remainder != 1 or denominator_remainder != 1:
        return None
    exponent: Fraction | None = None
    for prime, base_exponent in base_factors.items():
        current = Fraction(
            numerator_exponents[prime] - denominator_exponents[prime],
            base_exponent,
        )
        if exponent is None:
            exponent = current
        elif current != exponent:
            return None
    return exponent if exponent is not None else Fraction()


def _require_bounded_product(
    scale: int,
    weighted_ratios: list[tuple[Fraction, Fraction]],
) -> None:
    if scale.bit_length() > MAX_MUTUAL_INFORMATION_SCALE_BITS:
        raise ValueError(
            "mutual-information logarithmic representation scale exceeds the bound"
        )
    power_cost = 0
    for probability, ratio in weighted_ratios:
        scaled_probability = scale * probability
        if scaled_probability.denominator != 1:
            raise ValueError(
                "mutual-information logarithmic representation scale does not clear support masses"
            )
        exponent = scaled_probability.numerator
        if ratio == 1:
            continue
        power_cost += exponent * (
            ratio.numerator.bit_length() + ratio.denominator.bit_length()
        )
        if power_cost > MAX_MUTUAL_INFORMATION_POWER_COST_BITS:
            raise ValueError(
                "mutual-information logarithmic representation product exceeds the output-cost bound"
            )


@dataclass(frozen=True, slots=True)
class MutualInformationLogRepresentation:
    """Exact finite representation of ``scale * I = log_base(product)``."""

    scale: int
    product: Fraction

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale <= 0:
            raise ValueError(
                "mutual-information logarithmic representation scale must be positive"
            )
        if self.scale.bit_length() > MAX_MUTUAL_INFORMATION_SCALE_BITS:
            raise ValueError(
                "mutual-information logarithmic representation scale exceeds the bound"
            )
        if type(self.product) is not Fraction or self.product <= 0:
            raise ValueError(
                "mutual-information logarithmic representation product must be positive"
            )


@dataclass(frozen=True, slots=True)
class MutualInformationResult:
    """Exact native marginals, support terms, log representation, and value."""

    row_marginals: tuple[Fraction, ...]
    column_marginals: tuple[Fraction, ...]
    positive_support: tuple[MutualInformationTerm, ...]
    log_base: int
    logarithmic_value: MutualInformationLogRepresentation
    exact_value: Fraction | None
    sign: Literal["ZERO", "POSITIVE"]

    def __post_init__(self) -> None:
        """Check only native representation shape; the producer owns the identity."""

        _require_native_result_shape(
            self.row_marginals,
            self.column_marginals,
            self.positive_support,
        )
        if type(self.log_base) is not int or not 2 <= self.log_base <= 36:
            raise ValueError("mutual-information log base must lie from 2 through 36")

    @classmethod
    def _computed_from_kernel(
        cls,
        *,
        row_marginals: tuple[Fraction, ...],
        column_marginals: tuple[Fraction, ...],
        positive_support: tuple[MutualInformationTerm, ...],
        log_base: int,
        logarithmic_value: MutualInformationLogRepresentation,
        exact_value: Fraction | None,
        sign: Literal["ZERO", "POSITIVE"],
    ) -> MutualInformationResult:
        """Build the value after the kernel established its exact identity."""

        return cls(
            row_marginals=row_marginals,
            column_marginals=column_marginals,
            positive_support=positive_support,
            log_base=log_base,
            logarithmic_value=logarithmic_value,
            exact_value=exact_value,
            sign=sign,
        )


__all__ = [
    "FiniteJointTable",
    "MutualInformationLogRepresentation",
    "MutualInformationResult",
    "MutualInformationTerm",
]
