"""Exact bounded native construction of reviewed level-one named forms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Literal, cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_conjugate,
    require_complete_character_group,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms._models import SpaceDimensionResult
from jacobian.math.number_theory.modular_forms.kernel import (
    NamedLevelOneModularForm,
    eisenstein_coefficients,
    expected_coefficients,
    metadata,
    require_level_one_admission,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_MODULAR_FORM_WEIGHT,
    LevelOneModularQExpansion,
    ModularFormAtkinLehnerTarget,
    ModularFormSpace,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

from .transforms import named_q_expansion, sturm_bound


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


def modular_form_atkin_lehner_target(
    space: ModularFormSpace,
) -> ModularFormAtkinLehnerTarget:
    """Return the codomain parent of the full Fricke action ``W_N``.

    For integral weight, ``W_N`` preserves the Gamma0 level and changes the
    Nebentypus to its inverse. This reports only the typed parent correspondence;
    the normalized coefficient action is not represented here.
    """

    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.atkin_lehner_target_space_type",
            message="space must be an exact modular-form space value",
        )
    target_character = (
        "TRIVIAL"
        if space.character == "TRIVIAL"
        else dirichlet_character_conjugate(space.character)
    )
    target_space = ModularFormSpace(
        group=space.group,
        level=space.level,
        weight=space.weight,
        kind=space.kind,
        character=target_character,
        coefficient_domain=space.coefficient_domain,
    )
    return ModularFormAtkinLehnerTarget(
        source_space=space,
        target_space=target_space,
        divisor=space.level,
    )


def require_space_dimension_admission(space: ModularFormSpace) -> None:
    """Admit the exact space-dimension domain once per call."""

    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_space_type",
            message="space must be a modular-form space value",
        )
    if (
        type(space.level) is not int
        or not 1 <= space.level <= MAX_GAMMA0_OPERATION_LEVEL
    ):
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.space_dimension_unsupported_level",
            message=(
                "space-dimension level must be an integer in the exact "
                f"arithmetic envelope [1, {MAX_GAMMA0_OPERATION_LEVEL}]"
            ),
        )
    if (
        type(space.weight) is not int
        or not 0 <= space.weight <= MAX_MODULAR_FORM_WEIGHT
    ):
        raise OperationDomainValidationError(
            location=("space", "weight"),
            code="modular_form.space_dimension_unsupported_weight",
            message=(
                "space-dimension weight must be an integer in the exact "
                f"arithmetic envelope [0, {MAX_MODULAR_FORM_WEIGHT}]"
            ),
        )
    if space.kind not in ("M", "S"):
        raise OperationDomainValidationError(
            location=("space", "kind"),
            code="modular_form.space_dimension_unsupported_space",
            message="space kind must be M or S",
        )
    if space.group != "GAMMA0":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_unsupported_space",
            message="only Gamma0 spaces are supported",
        )
    if space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_dimension_unsupported_domain",
            message="only QQ coefficient domains are supported",
        )
    if space.character != "TRIVIAL":
        character = space.character
        if not isinstance(character, DirichletCharacter):
            raise OperationDomainValidationError(
                location=("space", "character"),
                code="modular_form.space_dimension_character",
                message="character must be a canonical Dirichlet character value",
            )
        group = require_complete_character_group(character.group)
        if (
            space.level != 4
            or space.weight not in (1, 3)
            or group.modulus != 4
            or group.generator_orders != (2,)
            or character.coordinates != (1,)
        ):
            raise OperationDomainValidationError(
                location=("space", "character"),
                code="modular_form.space_dimension_character_unsupported",
                message="only chi_{-4} at level 4 and weights 1 and 3 are admitted",
            )


def _prime_factorization(value: int) -> tuple[tuple[int, int], ...]:
    """Factor one admitted level by trial division.

    The public dimension envelope is 10,000, so this performs at most 100
    candidate divisions and avoids an unbounded general-purpose factorizer.
    """

    remaining = value
    factors: list[tuple[int, int]] = []
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            exponent = 0
            while remaining % prime == 0:
                remaining //= prime
                exponent += 1
            factors.append((prime, exponent))
        prime = 3 if prime == 2 else prime + 2
    if remaining > 1:
        factors.append((remaining, 1))
    return tuple(factors)


def _euler_phi_from_level_factors(value: int, primes: tuple[int, ...]) -> int:
    result = value
    for prime in primes:
        if value % prime == 0:
            result = result // prime * (prime - 1)
    return result


def _gamma0_geometry(
    level: int,
) -> tuple[int, int, int, int, int]:
    """Return index, genus, cusp count, and order-2/order-3 elliptic counts."""

    factors = _prime_factorization(level)
    primes = tuple(prime for prime, _ in factors)
    index = 1
    for prime, exponent in factors:
        index *= prime ** (exponent - 1) * (prime + 1)

    cusp_count = 0
    divisors = [1]
    for prime, exponent in factors:
        prime_power = 1
        new_divisors: list[int] = []
        for _ in range(exponent + 1):
            new_divisors.extend(divisor * prime_power for divisor in divisors)
            prime_power *= prime
        divisors = new_divisors
    for divisor in divisors:
        cusp_count += _euler_phi_from_level_factors(
            gcd(divisor, level // divisor),
            primes,
        )

    # The Kronecker symbols at 2 and 3 are zero; the exceptional divisibility
    # clauses encode the full local elliptic-point formulas at those primes.
    elliptic_2 = 0 if level % 4 == 0 else 1
    if elliptic_2:
        for prime in primes:
            symbol = 0 if prime == 2 else (1 if prime % 4 == 1 else -1)
            elliptic_2 *= 1 + symbol

    elliptic_3 = 0 if level % 2 == 0 or level % 9 == 0 else 1
    if elliptic_3:
        for prime in primes:
            symbol = 0 if prime == 3 else 1 if prime % 3 == 1 else -1
            elliptic_3 *= 1 + symbol

    genus_numerator = 12 + index - 3 * elliptic_2 - 4 * elliptic_3 - 6 * cusp_count
    if genus_numerator < 0 or genus_numerator % 12:
        raise RuntimeError("Gamma0 genus formula did not yield a nonnegative integer")
    genus = genus_numerator // 12
    return index, genus, cusp_count, elliptic_2, elliptic_3


def space_dimension(space: ModularFormSpace) -> SpaceDimensionResult:
    """Return the exact dimension of an admitted Gamma0 modular-form space.

    The owner admits the level before bounded trial factorization. It then
    computes the exact index, genus, elliptic-point counts and cusp count;
    those integers determine the Riemann-Roch dimension for every admitted
    nonnegative integral weight.
    """

    require_space_dimension_admission(space)
    weight = space.weight
    index, genus, cusp_count, elliptic_2, elliptic_3 = _gamma0_geometry(space.level)
    if space.character != "TRIVIAL":
        holomorphic, cusp, eisenstein = (1, 0, 1) if weight == 1 else (2, 0, 2)
    elif weight % 2 == 1:
        holomorphic, cusp, eisenstein = 0, 0, 0
    elif weight == 0:
        holomorphic, cusp, eisenstein = 1, 0, 1
    elif weight == 2:
        cusp = genus
        eisenstein = cusp_count - 1
        holomorphic = cusp + eisenstein
    else:
        cusp = (
            (weight - 1) * (genus - 1)
            + (weight // 2 - 1) * cusp_count
            + elliptic_2 * (weight // 4)
            + elliptic_3 * (weight // 3)
        )
        eisenstein = cusp_count
        holomorphic = cusp + eisenstein
    dimension = holomorphic if space.kind == "M" else cusp
    return SpaceDimensionResult._from_kernel(
        space,
        dimension=dimension,
        eisenstein_dimension=eisenstein,
        cusp_dimension=cusp,
        index=index,
        genus=genus,
        cusp_count=cusp_count,
        elliptic_points_order_2=elliptic_2,
        elliptic_points_order_3=elliptic_3,
    )


__all__ = [
    "level_one_named_q_expansion",
    "named_q_expansion",
    "space_dimension",
    "sturm_bound",
]
