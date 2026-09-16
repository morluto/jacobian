"""Exact bounded native construction of reviewed level-one named forms."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms._models import SpaceDimensionResult
from jacobian.math.number_theory.modular_forms.kernel import (
    NamedLevelOneModularForm,
    eisenstein_coefficients,
    expected_coefficients,
    metadata,
    require_level_one_admission,
)
from jacobian.math.number_theory.modular_forms.values import (
    LevelOneModularQExpansion,
    ModularFormSpace,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def _series(coefficients: tuple[Fraction, ...]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="q",
        truncation_order=len(coefficients),
        coefficients=tuple(
            CanonicalRational(
                num=coefficient.numerator,
                den=coefficient.denominator,
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


def require_space_dimension_admission(space: ModularFormSpace) -> None:
    """Admit the exact space-dimension domain once per call.

    Version 1 computes dimensions for level one only, where the closed
    floor/correction formula below is complete. Higher levels need
    genus, elliptic-point, and cusp-count data that no admitted value
    carries yet, so they are rejected as unsupported rather than answered
    from an incomplete formula.
    """

    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_space_type",
            message="space must be a modular-form space value",
        )
    if space.group != "GAMMA0" or space.character != "TRIVIAL":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_unsupported_space",
            message="only trivial-character Gamma0 spaces are supported",
        )
    if space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_unsupported_domain",
            message="only QQ coefficient domains are supported",
        )
    if space.level != 1:
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.space_dimension_unsupported_level",
            message="only level-one space dimensions are supported",
        )


def space_dimension(space: ModularFormSpace) -> SpaceDimensionResult:
    """Return the exact dimension of a supported level-one space.

    For SL(2, Z) with even weight k: dim M_0 = 1, dim M_2 = 0, and for even
    k >= 4, dim M_k = k//12 + 1 except k = 2 (mod 12) where it is k//12;
    S_k is M_k minus the one-dimensional Eisenstein line (absent for
    k < 4). Odd weights give the zero space.
    """

    require_space_dimension_admission(space)
    weight = space.weight
    if weight % 2 == 1:
        holomorphic, cusp, eisenstein = 0, 0, 0
    elif weight == 0:
        holomorphic, cusp, eisenstein = 1, 0, 1
    elif weight == 2:
        holomorphic, cusp, eisenstein = 0, 0, 0
    else:
        holomorphic = weight // 12 if weight % 12 == 2 else weight // 12 + 1
        eisenstein = 1
        cusp = holomorphic - 1
    dimension = holomorphic if space.kind == "M" else cusp
    return SpaceDimensionResult._from_kernel(
        space,
        dimension=dimension,
        eisenstein_dimension=eisenstein,
        cusp_dimension=cusp,
    )


__all__ = ["level_one_named_q_expansion", "space_dimension"]
