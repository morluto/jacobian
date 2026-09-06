"""Defining and boundary evidence for exact i.i.d. convolution powers (#2556)."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import gcd, prod
from time import perf_counter

import pytest
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability._distribution import (
    MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS,
    MAX_FINITE_DISTRIBUTION_ATOMS,
    FiniteConvolutionPowerRequest,
    FiniteConvolutionPowerResult,
    FiniteDistributionAtom,
    FiniteRationalDistribution,
)
from jacobian.math.probability._models import MAX_RESULT_RATIONAL_DIGITS
from jacobian.math.probability.operations import (
    MAX_CONVOLUTION_POWER_COEFFICIENT_PRODUCTS,
    _admit_convolution_peak,
    _lcm_within_result_digits,
    condition,
    convolution,
    convolution_peak,
    convolution_power,
    event_probability,
)


def _distribution(
    atoms: tuple[tuple[Fraction, Fraction], ...],
) -> FiniteRationalDistribution:
    return FiniteRationalDistribution(
        atoms=tuple(
            FiniteDistributionAtom(
                value=CanonicalRational.from_fraction(value),
                probability=CanonicalRational.from_fraction(probability),
            )
            for value, probability in atoms
        )
    )


def _fair_bit() -> FiniteRationalDistribution:
    return _distribution(
        (
            (Fraction(0), Fraction(1, 2)),
            (Fraction(1), Fraction(1, 2)),
        )
    )


def _three_point_lattice(
    degree: int, denominator: int = 3
) -> FiniteRationalDistribution:
    return _distribution(
        (
            (Fraction(0), Fraction(1, denominator)),
            (Fraction(1), Fraction(1, denominator)),
            (Fraction(degree), Fraction(denominator - 2, denominator)),
        )
    )


def _mass_map(distribution: FiniteRationalDistribution) -> dict[Fraction, Fraction]:
    return {
        atom.value.as_fraction(): atom.probability.as_fraction()
        for atom in distribution.atoms
    }


def test_fair_bit_power_and_even_odd_peak_conventions() -> None:
    source = _fair_bit()
    powered = convolution_power(source, 4)
    assert _mass_map(powered.distribution) == {
        Fraction(index): Fraction(coefficient, 16)
        for index, coefficient in enumerate((1, 4, 6, 4, 1))
    }

    even = convolution_peak(source, 4)
    assert even.maximum_probability.as_fraction() == Fraction(3, 8)
    assert tuple(value.as_fraction() for value in even.maximizing_values) == (
        Fraction(2),
    )

    odd = convolution_peak(source, 3)
    assert odd.maximum_probability.as_fraction() == Fraction(3, 8)
    assert tuple(value.as_fraction() for value in odd.maximizing_values) == (
        Fraction(1),
        Fraction(2),
    )


def test_power_matches_exhaustive_ordered_product_measure() -> None:
    source = _distribution(
        (
            (Fraction(-1, 2), Fraction(1, 3)),
            (Fraction(1), Fraction(2, 3)),
        )
    )
    expected: dict[Fraction, Fraction] = defaultdict(Fraction)
    source_atoms = tuple(
        (atom.value.as_fraction(), atom.probability.as_fraction())
        for atom in source.atoms
    )
    for choices in product(source_atoms, repeat=3):
        expected[sum((value for value, _ in choices), Fraction())] += prod(
            probability for _, probability in choices
        )

    result = convolution_power(source, 3)
    assert _mass_map(result.distribution) == dict(expected)
    assert sum(_mass_map(result.distribution).values(), Fraction()) == 1


def test_translation_and_positive_scaling_preserve_peak_masses() -> None:
    source = _distribution(
        (
            (Fraction(0), Fraction(1, 4)),
            (Fraction(2), Fraction(3, 4)),
        )
    )
    transformed = _distribution(
        tuple(
            (3 * atom.value.as_fraction() + 5, atom.probability.as_fraction())
            for atom in source.atoms
        )
    )
    original = convolution_peak(source, 5)
    changed = convolution_peak(transformed, 5)

    assert changed.maximum_probability == original.maximum_probability
    assert tuple(value.as_fraction() for value in changed.maximizing_values) == tuple(
        3 * value.as_fraction() + 25 for value in original.maximizing_values
    )


def test_degenerate_power_retains_source_binding_at_large_exponent() -> None:
    source = _distribution(((Fraction(2, 3), Fraction(1)),))
    result = convolution_power(source, 10**12)
    assert result.source == source
    assert result.exponent == 10**12
    assert _mass_map(result.distribution) == {Fraction(2 * 10**12, 3): Fraction(1)}


def test_odd_square_source_case_exceeds_the_binary_output_ceiling() -> None:
    source = _distribution(
        tuple((Fraction(odd * odd), Fraction(1, 17)) for odd in range(1, 34, 2))
    )
    result = convolution_power(source, 13)
    peak = convolution_peak(source, 13)
    masses = _mass_map(result.distribution)

    assert len(masses) > 256
    assert all((value - 13) % 8 == 0 for value in masses)
    assert sum(masses.values(), Fraction()) == 1
    maximum = max(masses.values())
    assert peak.maximum_probability.as_fraction() == maximum
    assert tuple(value.as_fraction() for value in peak.maximizing_values) == tuple(
        value for value, probability in masses.items() if probability == maximum
    )
    assert (
        FiniteConvolutionPowerResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_widened_power_result_composes_into_finite_distribution_consumers() -> None:
    powered = convolution_power(_fair_bit(), 430)
    assert len(powered.distribution.atoms) > MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS

    identity = convolution_power(powered.distribution, 1)
    assert identity.source == powered.distribution
    assert identity.distribution == powered.distribution

    masses = _mass_map(powered.distribution)
    peak = convolution_peak(powered.distribution, 1)
    maximum = max(masses.values())
    assert peak.maximum_probability.as_fraction() == maximum
    assert tuple(value.as_fraction() for value in peak.maximizing_values) == tuple(
        value for value, probability in masses.items() if probability == maximum
    )

    atom = powered.distribution.atoms[0]
    event = event_probability(powered.distribution, (atom.value,))
    assert event.event_probability == atom.probability
    assert event.selected_atoms == (atom,)

    conditioned = condition(powered.distribution, (atom.value,))
    assert _mass_map(conditioned.distribution) == {
        atom.value.as_fraction(): Fraction(1)
    }

    replayed = FiniteConvolutionPowerRequest.model_validate(
        {
            "distribution": powered.distribution.model_dump(mode="json"),
            "exponent": 1,
        }
    )
    assert (
        convolution_power(replayed.distribution, replayed.exponent).distribution
        == powered.distribution
    )

    degenerate = _distribution(((Fraction(0), Fraction(1)),))
    with pytest.raises(
        OperationDomainValidationError,
        match=rf"{MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS}-atom output bound",
    ):
        convolution(powered.distribution, degenerate)


def test_identity_power_preserves_mixed_large_denominator_source() -> None:
    first = 10**300 + 1
    second = 10**300 + 3
    source = _distribution(
        (
            (Fraction(0), Fraction(1, 2 * first)),
            (Fraction(1), Fraction(first - 1, 2 * first)),
            (Fraction(2), Fraction(1, 2 * second)),
            (Fraction(3), Fraction(second - 1, 2 * second)),
        )
    )

    identity = convolution_power(source, 1)
    assert identity.source == source
    assert identity.distribution == source

    peak = convolution_peak(source, 1)
    maximum = max(atom.probability.as_fraction() for atom in source.atoms)
    assert peak.maximum_probability.as_fraction() == maximum
    assert tuple(value.as_fraction() for value in peak.maximizing_values) == tuple(
        atom.value.as_fraction()
        for atom in source.atoms
        if atom.probability.as_fraction() == maximum
    )


def test_wider_canonical_carrier_does_not_widen_binary_convolution() -> None:
    source = _distribution(
        tuple((Fraction(2**index), Fraction(1, 23)) for index in range(23))
    )
    with pytest.raises(
        OperationDomainValidationError,
        match=rf"{MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS}-atom output bound",
    ):
        convolution(source, source)


def test_power_rejects_dense_lattice_span_before_backend_execution() -> None:
    boundary = _distribution(
        (
            (Fraction(0), Fraction(1, 3)),
            (Fraction(1), Fraction(1, 3)),
            (Fraction(MAX_FINITE_DISTRIBUTION_ATOMS - 1), Fraction(1, 3)),
        )
    )
    assert convolution_power(boundary, 1).distribution == boundary

    source = _distribution(
        (
            (Fraction(0), Fraction(1, 3)),
            (Fraction(1), Fraction(1, 3)),
            (Fraction(MAX_FINITE_DISTRIBUTION_ATOMS), Fraction(1, 3)),
        )
    )
    assert convolution_power(source, 1).distribution == source
    with pytest.raises(
        OperationDomainValidationError,
        match=rf"at most {MAX_FINITE_DISTRIBUTION_ATOMS} lattice positions",
    ):
        convolution_power(source, 2)


def test_power_rejects_work_and_height_envelopes_separately() -> None:
    admitted = _three_point_lattice(971)
    plan = _admit_convolution_peak(admitted, 12)
    charged_products = sum(left * right for left, right in plan.multiplication_shapes)
    assert charged_products == MAX_CONVOLUTION_POWER_COEFFICIENT_PRODUCTS - 292
    assert convolution_peak(admitted, 12).maximum_probability.as_fraction() > 0

    with pytest.raises(OperationDomainValidationError, match="product work bound"):
        convolution_peak(_three_point_lattice(121), 96)

    assert convolution_peak(_fair_bit(), 1_700).maximum_probability.as_fraction() > 0
    with pytest.raises(OperationDomainValidationError, match="digit result bound"):
        convolution_power(_fair_bit(), 1_701)


def test_complete_power_uses_the_structural_output_envelope() -> None:
    accepted = convolution_power(_three_point_lattice(100, 10_000), 100)
    assert len(accepted.distribution.atoms) > 256

    wider = convolution_power(_three_point_lattice(100, 100_000), 100)

    assert len(wider.distribution.atoms) == len(accepted.distribution.atoms)
    assert wider.source != accepted.source


def test_power_rejects_coprime_support_denominator_lcm_growth() -> None:
    dens: list[int] = []
    candidate = 10**90
    while len(dens) < 8:
        candidate += 1
        if all(gcd(candidate, den) == 1 for den in dens):
            dens.append(candidate)
    atoms = tuple(
        (Fraction(index, den), Fraction(1, 8)) for index, den in enumerate(dens)
    )
    started = perf_counter()
    with pytest.raises(
        OperationDomainValidationError,
        match="lattice denominators exceed",
    ):
        convolution_power(_distribution(atoms), 2)
    assert perf_counter() - started < 2.0


def test_lcm_height_guard_rejects_before_over_budget_product() -> None:
    with pytest.raises(
        OperationDomainValidationError,
        match="lattice denominators exceed",
    ):
        _lcm_within_result_digits(
            10**256,
            10**256 + 1,
            location=("distribution",),
            code="probability.convolution_power.height_bound",
            message="lattice denominators exceed the result bound",
        )


def test_power_rejects_a_lattice_value_over_the_exact_decimal_bound() -> None:
    denominator = 10**MAX_RESULT_RATIONAL_DIGITS
    source = _distribution(
        (
            (Fraction(0), Fraction(1, 3)),
            (Fraction(2, denominator), Fraction(1, 3)),
            (Fraction(5, denominator), Fraction(1, 3)),
        )
    )

    with pytest.raises(OperationDomainValidationError, match="512-digit"):
        convolution_power(source, 2)


def test_power_rejects_an_interior_value_with_a_larger_reduced_numerator() -> None:
    numerator = 2 * 10**511 + 3
    denominator = 18 * (10**510 + 1)
    source = _distribution(
        (
            (Fraction(0), Fraction(1, 3)),
            (Fraction(numerator, denominator), Fraction(1, 3)),
            (Fraction(9 * numerator, denominator), Fraction(1, 3)),
        )
    )

    with pytest.raises(OperationDomainValidationError, match="512-digit"):
        convolution_power(source, 2)


def test_peak_admits_when_a_nonmaximal_lattice_value_exceeds_height_bound() -> None:
    numerator = 2 * 10**511 + 3
    denominator = 18 * (10**510 + 1)
    source = _distribution(
        (
            (Fraction(0), Fraction(499, 500)),
            (Fraction(numerator, denominator), Fraction(1, 1000)),
            (Fraction(9 * numerator, denominator), Fraction(1, 1000)),
        )
    )

    peak = convolution_peak(source, 2)

    assert peak.maximum_probability.as_fraction() == Fraction(249001, 250000)
    assert tuple(value.as_fraction() for value in peak.maximizing_values) == (
        Fraction(0),
    )


def test_distribution_normalization_bounds_intermediate_denominators() -> None:
    denominators = tuple(10**100 + offset for offset in (1, 3, 7, 9, 13, 19))
    assert all(
        gcd(left, right) == 1
        for index, left in enumerate(denominators)
        for right in denominators[index + 1 :]
    )
    atoms = tuple(
        FiniteDistributionAtom(
            value=CanonicalRational.from_fraction(Fraction(index)),
            probability=CanonicalRational.from_fraction(probability),
        )
        for index, probability in enumerate(
            probability
            for probability in tuple(
                Fraction(1, 6 * denominator) for denominator in denominators
            )
            + tuple(
                Fraction(denominator - 1, 6 * denominator)
                for denominator in denominators
            )
        )
    )

    distribution = FiniteRationalDistribution(atoms=atoms)
    with pytest.raises(OperationDomainValidationError, match="intermediate bound"):
        event_probability(distribution, (atoms[0].value,))


def test_result_deserialization_does_not_repeat_power_admission() -> None:
    source = _distribution(
        (
            (Fraction(0), Fraction(1, 3)),
            (Fraction(1), Fraction(1, 3)),
            (Fraction(MAX_FINITE_DISTRIBUTION_ATOMS), Fraction(1, 3)),
        )
    )
    restored = FiniteConvolutionPowerResult.model_validate(
        {
            "source": source.model_dump(mode="json"),
            "exponent": 1,
            "distribution": source.model_dump(mode="json"),
        }
    )
    assert restored.source == source
