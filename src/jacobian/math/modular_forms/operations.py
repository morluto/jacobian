"""Exact bounded native construction of reviewed level-one named forms."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.formal_power_series._models import TruncatedSeries
from jacobian.math.formal_power_series._operations import (
    compute_power,
    compute_scalar_multiply,
    compute_subtract,
)
from jacobian.math.modular_forms.kernel import (
    NamedLevelOneModularForm,
    eisenstein_coefficients,
    metadata,
    require_level_one_admission,
)
from jacobian.math.modular_forms.values import LevelOneModularQExpansion


def _series(coefficients: tuple[Fraction, ...]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="q",
        truncation_order=len(coefficients),
        coefficients=tuple(
            CanonicalRational(
                num=format_canonical_integer(coefficient.numerator),
                den=format_canonical_integer(coefficient.denominator),
            )
            for coefficient in coefficients
        ),
    )


def _delta_series(truncation_order: int) -> TruncatedSeries:
    """Build Delta via its defining E4/E6 identity using the shared carrier."""
    e4 = _series(eisenstein_coefficients("E4", truncation_order))
    e6 = _series(eisenstein_coefficients("E6", truncation_order))
    e4_cubed = compute_power(e4, 3).result
    e6_squared = compute_power(e6, 2).result
    difference = compute_subtract(e4_cubed, e6_squared).result
    return compute_scalar_multiply(
        difference, CanonicalRational(num="1", den="1728")
    ).result


def level_one_named_q_expansion(
    form: NamedLevelOneModularForm, truncation_order: int
) -> LevelOneModularQExpansion:
    """Construct E4, E6, or Delta through one declared q-precision."""
    require_level_one_admission(form, truncation_order)
    if form == "DELTA":
        q_expansion = _delta_series(truncation_order)
    else:
        q_expansion = _series(eisenstein_coefficients(form, truncation_order))
    weight, space_kind, normalization = metadata(form)
    return LevelOneModularQExpansion(
        form=form,
        weight=cast(Literal[4, 6, 12], weight),
        space_kind=space_kind,
        normalization=normalization,
        q_expansion=q_expansion,
    )


__all__ = ["level_one_named_q_expansion"]
