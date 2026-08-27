"""Exact finite-table mutual-information operation."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from jacobian.math.probability.values import (
    MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
    FiniteJointTable,
    MutualInformationLogRepresentation,
    MutualInformationResult,
    MutualInformationTerm,
    _rational_base_exponent,
    _require_bounded_product,
)


def mutual_information(table: FiniteJointTable) -> MutualInformationResult:
    """Compute one exact native mutual-information value."""

    row_marginals = tuple(sum(row, Fraction()) for row in table.probabilities)
    column_marginals = tuple(
        sum(
            (
                table.probabilities[row][column]
                for row in range(len(table.probabilities))
            ),
            Fraction(),
        )
        for column in range(len(table.column_labels))
    )
    support: list[MutualInformationTerm] = []
    weighted_ratios: list[tuple[Fraction, Fraction]] = []
    for row_index, row in enumerate(table.probabilities):
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
                    probability=probability,
                    row_marginal=row_marginals[row_index],
                    column_marginal=column_marginals[column_index],
                    likelihood_ratio=ratio,
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
    return MutualInformationResult._computed_from_kernel(
        row_marginals=row_marginals,
        column_marginals=column_marginals,
        positive_support=tuple(support),
        log_base=table.log_base,
        logarithmic_value=MutualInformationLogRepresentation(
            scale=scale, product=product
        ),
        exact_value=exact_value,
        sign="ZERO" if product == 1 else "POSITIVE",
    )


__all__ = [
    "MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS",
    "FiniteJointTable",
    "MutualInformationLogRepresentation",
    "MutualInformationResult",
    "MutualInformationTerm",
    "mutual_information",
]
