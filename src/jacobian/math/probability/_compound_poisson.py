"""Exact compound-Poisson cumulant prefixes.

The compound-Poisson law is generally infinite, so this module deliberately
returns only the finite invariant that can be represented exactly here:
``kappa_n = lambda * E[J**n]`` for a requested prefix of orders.  It never
constructs a truncated PMF or claims that the returned rows are a complete
distribution.
"""

from fractions import Fraction
from math import gcd
from typing import Final

from pydantic import Field, StrictInt

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability._distribution import (
    FiniteDistributionAtom,
    FiniteRationalDistribution,
    require_input_distribution,
)
from jacobian.math.probability._models import MAX_RESULT_RATIONAL_DIGITS

MAX_COMPOUND_POISSON_ATOMS: Final = 256
MAX_COMPOUND_POISSON_ORDER: Final = 128
MAX_COMPOUND_POISSON_MOMENT_PRODUCTS: Final = (
    MAX_COMPOUND_POISSON_ATOMS * MAX_COMPOUND_POISSON_ORDER
)
_RESULT_VALUE_LIMIT: Final = 10**MAX_RESULT_RATIONAL_DIGITS - 1


class CompoundPoissonCumulantSource(StrictModel):
    intensity: CanonicalRational = Field(
        description=(
            "Nonnegative Poisson intensity. It is a reduced exact rational; "
            "execution admits at most 128 decimal digits per component."
        ),
        examples=[{"num": "2", "den": "1"}],
    )
    jump_distribution: FiniteRationalDistribution = Field(
        description=(
            "Normalized finite rational law for one independent jump, with "
            "strictly increasing support values. Execution admits at most "
            f"{MAX_COMPOUND_POISSON_ATOMS} atoms and 128 decimal digits per "
            "value or probability component."
        ),
        json_schema_extra={
            "x-jacobian-max-atoms": MAX_COMPOUND_POISSON_ATOMS,
            "x-jacobian-max-component-digits": 128,
        },
    )
    max_order: StrictInt = Field(
        ge=0,
        le=MAX_COMPOUND_POISSON_ORDER,
        description=(
            "Number of positive orders to return. Zero returns an empty "
            "prefix and is useful for representing no requested moments."
        ),
        examples=[3],
    )


class CompoundPoissonCumulantRequest(CompoundPoissonCumulantSource):
    """Transport request for the source-bound cumulant operation."""


class CompoundPoissonCumulantRow(StrictModel):
    order: StrictInt = Field(ge=1, le=MAX_COMPOUND_POISSON_ORDER)
    jump_raw_moment: CanonicalRational
    cumulant: CanonicalRational


class CompoundPoissonCumulantResult(StrictModel):
    source: CompoundPoissonCumulantSource
    cumulants: tuple[CompoundPoissonCumulantRow, ...] = Field(
        max_length=MAX_COMPOUND_POISSON_ORDER
    )


def _resource_error(
    *, location: tuple[str, ...], code: str, message: str
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=code, message=message
    )


def _domain_error(
    *, location: tuple[str, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _bounded_product(
    left: Fraction,
    right: Fraction,
    *,
    location: tuple[str, ...],
    label: str,
) -> Fraction:
    """Multiply with cross-cancellation before enforcing exact-height bounds."""

    left_numerator, left_denominator = left.numerator, left.denominator
    right_numerator, right_denominator = right.numerator, right.denominator
    first = gcd(abs(left_numerator), right_denominator)
    second = gcd(abs(right_numerator), left_denominator)
    left_numerator //= first
    right_denominator //= first
    right_numerator //= second
    left_denominator //= second
    if (
        right_numerator != 0
        and abs(left_numerator) > _RESULT_VALUE_LIMIT // abs(right_numerator)
    ) or left_denominator > _RESULT_VALUE_LIMIT // right_denominator:
        raise _resource_error(
            location=location,
            code="probability.compound_poisson.intermediate_height_bound",
            message=(
                f"{label} exceeds the {MAX_RESULT_RATIONAL_DIGITS}-digit "
                "exact intermediate bound"
            ),
        )
    numerator = left_numerator * right_numerator
    denominator = left_denominator * right_denominator
    return Fraction(numerator, denominator)


def _bounded_sum(
    left: Fraction,
    right: Fraction,
    *,
    location: tuple[str, ...],
    label: str,
) -> Fraction:
    """Add two exact terms while bounding common-denominator arithmetic."""

    common = gcd(left.denominator, right.denominator)
    left_scale = right.denominator // common
    right_scale = left.denominator // common
    if (
        left_scale > _RESULT_VALUE_LIMIT // max(1, abs(left.numerator))
        or right_scale > _RESULT_VALUE_LIMIT // max(1, abs(right.numerator))
        or left_scale > _RESULT_VALUE_LIMIT // left.denominator
    ):
        raise _resource_error(
            location=location,
            code="probability.compound_poisson.intermediate_height_bound",
            message=(
                f"{label} exceeds the {MAX_RESULT_RATIONAL_DIGITS}-digit "
                "exact intermediate bound"
            ),
        )
    left_numerator = left.numerator * left_scale
    right_numerator = right.numerator * right_scale
    denominator = left.denominator * left_scale
    numerator = left_numerator + right_numerator
    if abs(numerator) > _RESULT_VALUE_LIMIT:
        raise _resource_error(
            location=location,
            code="probability.compound_poisson.intermediate_height_bound",
            message=(
                f"{label} exceeds the {MAX_RESULT_RATIONAL_DIGITS}-digit "
                "exact intermediate bound"
            ),
        )
    return Fraction(numerator, denominator)


def _require_canonical_rational(
    value: object,
    *,
    location: tuple[str, ...],
    label: str,
) -> CanonicalRational:
    """Reject forged rationals that skip CanonicalRational's reduced-form checks."""

    if not isinstance(value, CanonicalRational):
        raise _domain_error(
            location=location,
            code="probability.compound_poisson.canonical_rational_type",
            message=f"{label} must be a canonical rational",
        )
    numerator = value.num
    denominator = value.den
    if type(numerator) is not int or type(denominator) is not int:
        raise _domain_error(
            location=location,
            code="probability.compound_poisson.canonical_rational_components",
            message=f"{label} components must be exact integers",
        )
    if denominator <= 0:
        raise _domain_error(
            location=location,
            code="probability.compound_poisson.canonical_rational",
            message=f"{label} must have a positive denominator",
        )
    if gcd(abs(numerator), denominator) != 1 or (
        numerator == 0 and denominator != 1
    ):
        raise _domain_error(
            location=location,
            code="probability.compound_poisson.canonical_rational",
            message=f"{label} must be reduced with canonical zero 0/1",
        )
    return value


def _admit_and_plan(
    intensity: CanonicalRational,
    jump_distribution: FiniteRationalDistribution,
    max_order: int,
) -> tuple[CompoundPoissonCumulantSource, tuple[tuple[int, Fraction, Fraction], ...]]:
    """Admit all semantic work once and return its reusable arithmetic ledger."""

    if (
        type(max_order) is not int
        or not 0 <= max_order <= MAX_COMPOUND_POISSON_ORDER
    ):
        raise _domain_error(
            location=("max_order",),
            code="probability.compound_poisson.order_bound",
            message=(
                "compound-Poisson order must be between 0 and "
                f"{MAX_COMPOUND_POISSON_ORDER}"
            ),
        )
    intensity = _require_canonical_rational(
        intensity, location=("intensity",), label="compound-Poisson intensity"
    )
    try:
        require_bounded_rational(
            intensity,
            max_digits=128,
            label="compound-Poisson intensity",
        )
    except ValueError as exc:
        raise _resource_error(
            location=("intensity",),
            code="probability.compound_poisson.input_height_bound",
            message=str(exc),
        ) from exc
    if intensity.num < 0:
        raise _domain_error(
            location=("intensity",),
            code="probability.compound_poisson.nonnegative_intensity",
            message="compound-Poisson intensity must be nonnegative",
        )
    if not isinstance(jump_distribution, FiniteRationalDistribution):
        raise _domain_error(
            location=("jump_distribution",),
            code="probability.compound_poisson.distribution_type",
            message="jump_distribution must be a FiniteRationalDistribution",
        )
    atoms = jump_distribution.atoms
    if type(atoms) is not tuple or not all(
        isinstance(atom, FiniteDistributionAtom) for atom in atoms
    ):
        raise _domain_error(
            location=("jump_distribution", "atoms"),
            code="probability.compound_poisson.atom_type",
            message="jump_distribution atoms must be finite-distribution atoms",
        )
    if not 1 <= len(atoms) <= MAX_COMPOUND_POISSON_ATOMS:
        raise _resource_error(
            location=("jump_distribution", "atoms"),
            code="probability.compound_poisson.support_bound",
            message=(
                "compound-Poisson jump laws accept at most "
                f"{MAX_COMPOUND_POISSON_ATOMS} support atoms"
            ),
        )
    for index, atom in enumerate(atoms):
        _require_canonical_rational(
            atom.value,
            location=("jump_distribution", "atoms", str(index), "value"),
            label="jump value",
        )
        probability = _require_canonical_rational(
            atom.probability,
            location=("jump_distribution", "atoms", str(index), "probability"),
            label="jump probability",
        )
        if probability.num < 0:
            raise _domain_error(
                location=("jump_distribution", "atoms", str(index), "probability"),
                code="probability.compound_poisson.nonnegative_probability",
                message="jump probabilities must be nonnegative canonical rationals",
            )
    try:
        require_input_distribution(
            atoms,
            require_canonical=True,
            max_digits=128,
        )
    except (AttributeError, TypeError) as exc:
        raise _domain_error(
            location=("jump_distribution",),
            code="probability.compound_poisson.distribution_admission",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        detail = str(exc)
        if "digit" in detail or "normalization exceeds" in detail:
            raise _resource_error(
                location=("jump_distribution",),
                code="probability.compound_poisson.input_height_bound",
                message=detail,
            ) from exc
        raise _domain_error(
            location=("jump_distribution",),
            code="probability.compound_poisson.distribution_admission",
            message=detail,
        ) from exc
    products = len(atoms) * max_order
    if products > MAX_COMPOUND_POISSON_MOMENT_PRODUCTS:
        raise _resource_error(
            location=("jump_distribution", "max_order"),
            code="probability.compound_poisson.work_bound",
            message="compound-Poisson moment work exceeds the admitted bound",
        )

    values = tuple(atom.value.as_fraction() for atom in atoms)
    probabilities = tuple(atom.probability.as_fraction() for atom in atoms)
    intensity_value = intensity.as_fraction()
    powers = [Fraction(1) for _ in atoms]
    rows: list[tuple[int, Fraction, Fraction]] = []
    for order in range(1, max_order + 1):
        for index, value in enumerate(values):
            powers[index] = _bounded_product(
                powers[index],
                value,
                location=("jump_distribution", "atoms", str(index), "value"),
                label="powered jump value",
            )
        moment = Fraction()
        for index, (probability, power) in enumerate(
            zip(probabilities, powers, strict=True)
        ):
            contribution = _bounded_product(
                probability,
                power,
                location=("jump_distribution", "atoms", str(index)),
                label="jump-moment contribution",
            )
            moment = _bounded_sum(
                moment,
                contribution,
                location=("jump_distribution", "atoms"),
                label="jump raw moment",
            )
        cumulant = _bounded_product(
            intensity_value,
            moment,
            location=("intensity",),
            label="compound-Poisson cumulant",
        )
        rows.append((order, moment, cumulant))
    source = CompoundPoissonCumulantSource(
        intensity=intensity,
        jump_distribution=jump_distribution,
        max_order=max_order,
    )
    return source, tuple(rows)


def compound_poisson_cumulant_prefix(
    intensity: CanonicalRational,
    jump_distribution: FiniteRationalDistribution,
    max_order: int,
) -> CompoundPoissonCumulantResult:
    source, plan = _admit_and_plan(intensity, jump_distribution, max_order)
    rows = tuple(
        CompoundPoissonCumulantRow(
            order=order,
            jump_raw_moment=CanonicalRational.from_fraction(moment),
            cumulant=CanonicalRational.from_fraction(cumulant),
        )
        for order, moment, cumulant in plan
    )
    return CompoundPoissonCumulantResult(source=source, cumulants=rows)


__all__ = [
    "CompoundPoissonCumulantRequest",
    "CompoundPoissonCumulantResult",
    "CompoundPoissonCumulantSource",
    "compound_poisson_cumulant_prefix",
]
