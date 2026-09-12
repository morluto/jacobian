"""Exact compound-Poisson cumulant tests."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability._compound_poisson import (
    CompoundPoissonCumulantRequest,
    CompoundPoissonCumulantSource,
    MAX_COMPOUND_POISSON_ATOMS,
    compound_poisson_cumulant_prefix,
)
from jacobian.math.probability._distribution import (
    FiniteDistributionAtom,
    FiniteRationalDistribution,
)


def _q(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def test_compound_poisson_cumulants_are_intensity_times_jump_moments() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 2))
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(2)), probability=_q(Fraction(1, 2))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction(3, 2)), jumps, 4)
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [
        1,
        2,
        4,
        8,
    ]
    assert [row.cumulant.as_fraction() for row in result.cumulants] == [
        Fraction(3, 2),
        3,
        6,
        12,
    ]


def test_order_zero_returns_empty_prefix_without_a_pmf() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(value=_q(Fraction(1)), probability=_q(Fraction(1))),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction()), jumps, 0)
    assert result.cumulants == ()


def test_intensity_growth_is_admitted_before_cumulant_construction() -> None:
    large = 10**255 - 1
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(large)), probability=_q(Fraction(1))
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError):
        compound_poisson_cumulant_prefix(_q(Fraction(large)), jumps, 128)


def test_zero_intensity_and_signed_deterministic_jumps() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(-3, 2)), probability=_q(Fraction(1))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction(2)), jumps, 4)
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [
        Fraction(-3, 2),
        Fraction(9, 4),
        Fraction(-27, 8),
        Fraction(81, 16),
    ]
    assert [row.cumulant.as_fraction() for row in result.cumulants] == [
        Fraction(-3),
        Fraction(9, 2),
        Fraction(-27, 4),
        Fraction(81, 8),
    ]

    zero = compound_poisson_cumulant_prefix(_q(Fraction()), jumps, 4)
    assert all(row.cumulant.as_fraction() == 0 for row in zero.cumulants)


def test_native_boundary_rejects_non_value_arguments_with_owner_error() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(value=_q(Fraction(1)), probability=_q(Fraction(1))),
        )
    )
    with pytest.raises(OperationDomainValidationError) as intensity_error:
        compound_poisson_cumulant_prefix({}, jumps, 1)  # type: ignore[arg-type]
    assert (
        intensity_error.value.errors()[0]["type"]
        == "probability.compound_poisson.canonical_rational_type"
    )

    with pytest.raises(OperationDomainValidationError) as request_error:
        compound_poisson_cumulant_prefix(
            CompoundPoissonCumulantRequest(
                intensity=_q(Fraction(1)), jump_distribution=jumps, max_order=1
            ),  # type: ignore[arg-type]
            jumps,
            1,
        )
    assert (
        request_error.value.errors()[0]["type"]
        == "probability.compound_poisson.canonical_rational_type"
    )


def test_semantic_admission_rejects_negative_rate_and_unnormalized_jumps() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 2))
            ),
        )
    )
    with pytest.raises(OperationDomainValidationError) as rate_error:
        compound_poisson_cumulant_prefix(_q(Fraction(-1)), jumps, 1)
    assert rate_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.nonnegative_intensity"
    )

    with pytest.raises(OperationDomainValidationError) as mass_error:
        compound_poisson_cumulant_prefix(_q(Fraction(1)), jumps, 1)
    assert mass_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.distribution_admission"
    )


def test_forged_negative_jump_mass_is_rejected_at_admission() -> None:
    negative = CanonicalRational.model_construct(num=-1, den=1)
    unit = CanonicalRational.model_construct(num=2, den=1)
    atom_negative = FiniteDistributionAtom.model_construct(
        value=_q(Fraction(0)), probability=negative
    )
    atom_overflow = FiniteDistributionAtom.model_construct(
        value=_q(Fraction(1)), probability=unit
    )
    forged = FiniteRationalDistribution.model_construct(
        atoms=(atom_negative, atom_overflow)
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compound_poisson_cumulant_prefix(_q(Fraction(1)), forged, 1)
    assert exc_info.value.errors()[0]["type"] == (
        "probability.compound_poisson.nonnegative_probability"
    )


def test_forged_noncanonical_rationals_are_rejected_before_arithmetic() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(value=_q(Fraction(1)), probability=_q(Fraction(1))),
        )
    )
    zero_den = CanonicalRational.model_construct(num=1, den=0)
    with pytest.raises(OperationDomainValidationError) as intensity_den:
        compound_poisson_cumulant_prefix(zero_den, jumps, 1)
    assert intensity_den.value.errors()[0]["type"] == (
        "probability.compound_poisson.canonical_rational"
    )

    unreduced = CanonicalRational.model_construct(num=2, den=2)
    with pytest.raises(OperationDomainValidationError) as intensity_reduced:
        compound_poisson_cumulant_prefix(unreduced, jumps, 1)
    assert intensity_reduced.value.errors()[0]["type"] == (
        "probability.compound_poisson.canonical_rational"
    )

    missing_den = CanonicalRational.model_construct(num=1)
    with pytest.raises(OperationDomainValidationError) as intensity_missing:
        compound_poisson_cumulant_prefix(missing_den, jumps, 1)
    assert intensity_missing.value.errors()[0]["type"] == (
        "probability.compound_poisson.canonical_rational_components"
    )

    forged_value = FiniteDistributionAtom.model_construct(
        value=CanonicalRational.model_construct(num=2, den=2),
        probability=_q(Fraction(1)),
    )
    forged_prob = FiniteDistributionAtom.model_construct(
        value=_q(Fraction(0)),
        probability=CanonicalRational.model_construct(num=1, den=0),
    )
    with pytest.raises(OperationDomainValidationError) as value_error:
        compound_poisson_cumulant_prefix(
            _q(Fraction(1)),
            FiniteRationalDistribution.model_construct(atoms=(forged_value,)),
            1,
        )
    assert value_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.canonical_rational"
    )
    with pytest.raises(OperationDomainValidationError) as probability_error:
        compound_poisson_cumulant_prefix(
            _q(Fraction(1)),
            FiniteRationalDistribution.model_construct(atoms=(forged_prob,)),
            1,
        )
    assert probability_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.canonical_rational"
    )


def test_jump_height_outside_execution_envelope_is_a_resource_error() -> None:
    tall = 10**128
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(tall)), probability=_q(Fraction(1))
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        compound_poisson_cumulant_prefix(_q(Fraction(1)), jumps, 1)
    assert exc_info.value.errors()[0]["type"] == (
        "probability.compound_poisson.input_height_bound"
    )


def test_zero_mass_atoms_are_excluded_from_the_power_plan() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1))
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(10**127)), probability=_q(Fraction(0))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction(3)), jumps, 5)
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [0] * 5
    assert [row.cumulant.as_fraction() for row in result.cumulants] == [0] * 5


def test_missing_atoms_member_is_a_typed_domain_error() -> None:
    forged = FiniteRationalDistribution.model_construct()
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compound_poisson_cumulant_prefix(_q(Fraction(1)), forged, 1)
    assert exc_info.value.errors()[0]["type"] == (
        "probability.compound_poisson.atom_type"
    )


def test_empty_support_is_a_domain_error() -> None:
    empty = FiniteRationalDistribution.model_construct(atoms=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compound_poisson_cumulant_prefix(_q(Fraction(1)), empty, 1)
    assert exc_info.value.errors()[0]["type"] == (
        "probability.compound_poisson.empty_support"
    )


def test_probability_weighted_powers_admit_cancelled_tall_jumps() -> None:
    tall = 10**127
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)),
                probability=_q(Fraction(tall - 1, tall)),
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(tall)),
                probability=_q(Fraction(1, tall)),
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction(1)), jumps, 5)
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [
        1,
        tall,
        tall**2,
        tall**3,
        tall**4,
    ]


def test_jump_law_schema_publishes_the_execution_envelope() -> None:
    schema = CompoundPoissonCumulantSource.model_json_schema()["properties"][
        "jump_distribution"
    ]
    assert str(MAX_COMPOUND_POISSON_ATOMS) in schema["description"]
    assert "128" in schema["description"]
    assert schema["x-jacobian-max-atoms"] == MAX_COMPOUND_POISSON_ATOMS
    assert schema["x-jacobian-max-component-digits"] == 128


def test_prefix_round_trips_with_its_source_parent() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 3))
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(2)), probability=_q(Fraction(2, 3))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(_q(Fraction(5, 7)), jumps, 3)
    assert result.model_validate_json(result.model_dump_json()) == result
    assert result.source.jump_distribution == jumps
