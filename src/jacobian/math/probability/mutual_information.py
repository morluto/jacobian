"""Exact finite-table mutual-information operation."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability.values import (
    MAX_MUTUAL_INFORMATION_POWER_COST_BITS,
    MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
    MAX_MUTUAL_INFORMATION_SCALE_BITS,
    FiniteJointTable,
    MutualInformationLogRepresentation,
    MutualInformationResult,
    MutualInformationTerm,
)

MAX_MUTUAL_INFORMATION_POSITIVE_SUPPORT = 64


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
        raise OperationResourceAdmissionError(
            location=("probabilities",),
            code="probability.mutual_information.scale_bound",
            message=(
                "mutual-information logarithmic representation scale exceeds the bound"
            ),
        )
    power_cost = 0
    for probability, ratio in weighted_ratios:
        scaled_probability = scale * probability
        if scaled_probability.denominator != 1:
            raise OperationResourceAdmissionError(
                location=("probabilities",),
                code="probability.mutual_information.scale_bound",
                message=(
                    "mutual-information logarithmic representation scale does not clear support masses"
                ),
            )
        exponent = scaled_probability.numerator
        if ratio == 1:
            continue
        power_cost += exponent * (
            ratio.numerator.bit_length() + ratio.denominator.bit_length()
        )
        if power_cost > MAX_MUTUAL_INFORMATION_POWER_COST_BITS:
            raise OperationResourceAdmissionError(
                location=("probabilities",),
                code="probability.mutual_information.product_bound",
                message=(
                    "mutual-information logarithmic representation product exceeds the output-cost bound"
                ),
            )


def mutual_information(table: FiniteJointTable) -> MutualInformationResult:
    """Compute one exact native mutual-information value."""

    probabilities = tuple(
        tuple(value.as_fraction() for value in row) for row in table.probabilities
    )
    row_marginals = tuple(sum(row, Fraction()) for row in probabilities)
    if sum(row_marginals, Fraction()) != 1:
        raise OperationDomainValidationError(
            location=("probabilities",),
            code="probability.joint_table_not_normalized",
            message="joint-table probabilities must sum exactly to 1",
        )
    column_marginals = tuple(
        sum(
            (probabilities[row][column] for row in range(len(table.probabilities))),
            Fraction(),
        )
        for column in range(len(table.column_labels))
    )
    positive_support_count = sum(
        probability != 0 for row in probabilities for probability in row
    )
    if positive_support_count > MAX_MUTUAL_INFORMATION_POSITIVE_SUPPORT:
        raise OperationResourceAdmissionError(
            location=("probabilities",),
            code="probability.mutual_information.support_bound",
            message=(
                "mutual information accepts at most "
                f"{MAX_MUTUAL_INFORMATION_POSITIVE_SUPPORT} positive cells"
            ),
        )
    support: list[MutualInformationTerm] = []
    weighted_ratios: list[tuple[Fraction, Fraction]] = []
    for row_index, row in enumerate(probabilities):
        for column_index, probability in enumerate(row):
            if probability == 0:
                continue
            product_marginal = row_marginals[row_index] * column_marginals[column_index]
            if product_marginal == 0:
                raise ValueError("positive joint mass has zero marginal support")
            ratio = probability / product_marginal
            weighted_ratios.append((probability, ratio))
            support.append(
                MutualInformationTerm(
                    row_index=row_index,
                    column_index=column_index,
                    probability=CanonicalRational.from_fraction(probability),
                    row_marginal=CanonicalRational.from_fraction(
                        row_marginals[row_index]
                    ),
                    column_marginal=CanonicalRational.from_fraction(
                        column_marginals[column_index]
                    ),
                    likelihood_ratio=CanonicalRational.from_fraction(ratio),
                )
            )
    scale = lcm(*(probability.denominator for probability, _ in weighted_ratios))
    _require_bounded_product(scale, weighted_ratios)
    product = Fraction(1)
    for probability, ratio in weighted_ratios:
        exponent = scale * probability.numerator // probability.denominator
        product *= ratio**exponent
    if product < 1:
        raise ValueError("mutual-information log product contradicts nonnegativity")

    base_exponent = _rational_base_exponent(product, table.log_base)
    exact_value = None if base_exponent is None else base_exponent / scale
    return MutualInformationResult(
        row_labels=table.row_labels,
        column_labels=table.column_labels,
        row_marginals=tuple(
            CanonicalRational.from_fraction(value) for value in row_marginals
        ),
        column_marginals=tuple(
            CanonicalRational.from_fraction(value) for value in column_marginals
        ),
        positive_support=tuple(support),
        log_base=table.log_base,
        exact_logarithmic_value=MutualInformationLogRepresentation(
            scale=scale, product=CanonicalRational.from_fraction(product)
        ),
        exact_value=None
        if exact_value is None
        else CanonicalRational.from_fraction(exact_value),
        sign="ZERO" if product == 1 else "POSITIVE",
    )


__all__ = [
    "MAX_MUTUAL_INFORMATION_POSITIVE_SUPPORT",
    "MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS",
    "FiniteJointTable",
    "MutualInformationLogRepresentation",
    "MutualInformationResult",
    "MutualInformationTerm",
    "mutual_information",
]
