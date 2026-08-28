"""Exact bounded native construction of reviewed level-one named forms."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.modular_forms.kernel import (
    NamedLevelOneModularForm,
    eisenstein_coefficients,
    expected_coefficients,
    metadata,
    require_level_one_admission,
)
from jacobian.math.number_theory.modular_forms.values import LevelOneModularQExpansion
from jacobian.math.polynomials.series._models import TruncatedSeries


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
    """Build Delta from its exact owner-local defining formula.

    The modular-form kernel has a wider finite-prefix envelope than the
    general-purpose formal-series arithmetic operations.  Keeping this
    closed-form construction here lets the modular operation admit and
    compute its own advertised precision without inheriting an unrelated
    intermediate-power ceiling.
    """

    return _series(expected_coefficients("DELTA", truncation_order))


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
    return LevelOneModularQExpansion._from_kernel(
        form=form,
        weight=cast(Literal[4, 6, 12], weight),
        space_kind=space_kind,
        normalization=normalization,
        q_expansion=q_expansion,
    )


__all__ = ["level_one_named_q_expansion"]
