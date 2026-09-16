"""Independent checks for Kempner reciprocal-series enclosures."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import KempnerDigitSet
from jacobian.math.number_theory.kempner import (
    enclose_kempner_series,
    require_series_admission,
)
from jacobian.math.number_theory.kempner._models import (
    MAX_KEMPNER_SERIES_NUMERALS,
    KempnerSeriesEnclosure,
    KempnerSeriesEnclosureRequest,
)
from jacobian.math.number_theory.kempner._tools import TOOLS


def _family(base: int, digits: tuple[int, ...]) -> list[int]:
    """Enumerate accepted positive integers below base**3 independently."""

    members = []
    for value in range(1, base**3):
        remaining, ok = value, True
        while remaining:
            remaining, digit = divmod(remaining, base)
            if digit not in digits:
                ok = False
                break
        if ok:
            members.append(value)
    return members


def _partial(base: int, digits: tuple[int, ...], cutoff: int) -> Fraction:
    """Recompute the partial sum by direct canonical-numeral enumeration."""

    nonzero = tuple(d for d in digits if d != 0)
    total = Fraction(0)
    for length in range(1, cutoff + 1):
        for first in nonzero:
            for rest in product(digits, repeat=length - 1):
                value = first
                for digit in rest:
                    value = value * base + digit
                total += Fraction(1, value)
    return total


def test_repunit_family_known_answer() -> None:
    digit_set = KempnerDigitSet(base=10, allowed_digits=(1,))
    result = enclose_kempner_series(digit_set, 2)

    # 1 + 1/11 = 12/11; tail = (1/10)^2/(1 - 1/10) = 1/90.
    assert result.partial_sum == CanonicalRational(num=12, den=11)
    assert result.tail_upper_bound == CanonicalRational(num=1, den=90)
    assert result.lower == result.partial_sum
    assert result.upper.as_fraction() == Fraction(12, 11) + Fraction(1, 90)


def test_enumeration_reproduces_partial_and_tail_formula() -> None:
    cases = [
        (10, (1,), 3),
        (10, (0, 1), 2),
        (3, (1, 2), 3),
        (2, (1,), 4),
        (8, (0, 3, 5), 2),
    ]
    for base, digits, cutoff in cases:
        digit_set = KempnerDigitSet(base=base, allowed_digits=digits)
        result = enclose_kempner_series(digit_set, cutoff)
        expected_partial = _partial(base, digits, cutoff)

        assert result.partial_sum.as_fraction() == expected_partial
        assert result.lower.as_fraction() == expected_partial
        size, nonzero = len(digits), sum(1 for d in digits if d != 0)
        expected_tail = Fraction(
            nonzero * size**cutoff * base, base**cutoff * (base - size)
        )
        assert result.tail_upper_bound.as_fraction() == expected_tail
        assert result.upper.as_fraction() == expected_partial + expected_tail


def test_tail_dominates_omitted_terms() -> None:
    base, digits, cutoff = 10, (1,), 2
    digit_set = KempnerDigitSet(base=base, allowed_digits=digits)
    result = enclose_kempner_series(digit_set, cutoff)

    # Every omitted m-digit member is at least b^(m-1); bound layers 3..6
    # directly and require the returned tail to dominate their total.
    omitted = Fraction(0)
    for length in range(cutoff + 1, 7):
        layer_count = 1 * 1 ** (length - 1)
        omitted += Fraction(layer_count, base ** (length - 1))
    assert result.tail_upper_bound.as_fraction() >= omitted

    # The three-digit layer alone is exactly 1/111 <= tail.
    assert result.tail_upper_bound.as_fraction() >= Fraction(1, 111)


def test_empty_positive_family_yields_zero_interval() -> None:
    result = enclose_kempner_series(KempnerDigitSet(base=10, allowed_digits=(0,)), 5)

    assert result.partial_sum.as_fraction() == 0
    assert result.tail_upper_bound.as_fraction() == 0
    assert result.lower.as_fraction() == 0
    assert result.upper.as_fraction() == 0


def test_zero_cutoff_covers_nothing_with_full_tail() -> None:
    digit_set = KempnerDigitSet(base=10, allowed_digits=(1,))
    result = enclose_kempner_series(digit_set, 0)

    assert result.partial_sum.as_fraction() == 0
    assert result.tail_upper_bound.as_fraction() == Fraction(1, 1 - Fraction(1, 10))
    assert result.upper.as_fraction() == result.tail_upper_bound.as_fraction()


def test_leading_zero_exclusion_matches_family() -> None:
    # S = {0, 1}: one-digit members are {1} only, not {0, 1}.
    digit_set = KempnerDigitSet(base=10, allowed_digits=(0, 1))
    result = enclose_kempner_series(digit_set, 1)

    assert result.partial_sum.as_fraction() == Fraction(1)
    assert result.tail_upper_bound.as_fraction() == Fraction(2 * 10, 10 * 8)


def test_monotonic_nesting_as_cutoff_grows() -> None:
    digit_set = KempnerDigitSet(base=3, allowed_digits=(1, 2))
    narrower = enclose_kempner_series(digit_set, 2)
    wider = enclose_kempner_series(digit_set, 3)

    assert wider.lower.as_fraction() >= narrower.lower.as_fraction()
    assert wider.upper.as_fraction() <= narrower.upper.as_fraction()
    assert wider.lower.as_fraction() <= wider.upper.as_fraction()


def test_agrees_with_direct_family_enumeration() -> None:
    base, digits = 3, (1, 2)
    digit_set = KempnerDigitSet(base=base, allowed_digits=digits)
    result = enclose_kempner_series(digit_set, 2)
    members = [n for n in _family(base, digits) if n < base**2]

    assert result.partial_sum.as_fraction() == sum(
        (Fraction(1, n) for n in members), Fraction(0)
    )
    assert len(members) == 2 + 4


def test_native_and_catalog_results_agree() -> None:
    digit_set = KempnerDigitSet(base=10, allowed_digits=(1,))
    native = enclose_kempner_series(digit_set, 2)
    assert len(TOOLS) == 1
    assert TOOLS[0].operation_id == "number_theory.kempner_series.enclose"
    catalog = TOOLS[0].run(KempnerSeriesEnclosureRequest(digit_set=digit_set, cutoff=2))

    assert catalog == native
    assert (
        KempnerSeriesEnclosure.model_validate_json(catalog.model_dump_json()) == catalog
    )


def test_noncanonical_digit_set_rejected() -> None:
    forged = KempnerDigitSet.model_construct(base=10, allowed_digits=(2, 1))

    with pytest.raises(OperationDomainValidationError):
        enclose_kempner_series(forged, 2)


def test_full_alphabet_rejected() -> None:
    forged = KempnerDigitSet.model_construct(base=2, allowed_digits=(0, 1))

    with pytest.raises(OperationDomainValidationError):
        enclose_kempner_series(forged, 2)


def test_negative_cutoff_rejected() -> None:
    digit_set = KempnerDigitSet(base=10, allowed_digits=(1,))

    with pytest.raises(OperationDomainValidationError):
        enclose_kempner_series(digit_set, -1)


def test_envelope_refuses_before_enumeration() -> None:
    # Base-2 {1} with a huge cutoff: one numeral per length, so the rational
    # height (not the count) forces refusal before any enumeration runs.
    digit_set = KempnerDigitSet(base=2, allowed_digits=(1,))

    with pytest.raises(OperationResourceAdmissionError):
        require_series_admission(digit_set, 999)
    with pytest.raises(OperationResourceAdmissionError):
        enclose_kempner_series(digit_set, 999)


def test_numeral_count_refuses_dense_families() -> None:
    # Excluding one decimal digit through 6 digits needs 9·(9⁶-1)/8 numerals.
    digit_set = KempnerDigitSet(base=10, allowed_digits=(0, 1, 2, 3, 4, 5, 6, 7, 8))

    with pytest.raises(OperationResourceAdmissionError):
        enclose_kempner_series(digit_set, 6)


def test_boundary_numeral_count_is_admitted() -> None:
    digit_set = KempnerDigitSet(base=2, allowed_digits=(1,))
    assert require_series_admission(digit_set, 10) == 10
    result = enclose_kempner_series(digit_set, 10)
    assert result.lower.as_fraction() <= result.upper.as_fraction()


def test_negative_tail_rejected_structurally() -> None:
    digit_set = KempnerDigitSet(base=10, allowed_digits=(1,))
    good = enclose_kempner_series(digit_set, 1)

    with pytest.raises(ValidationError):
        KempnerSeriesEnclosure(
            digit_set=digit_set,
            cutoff=1,
            partial_sum=good.partial_sum,
            tail_upper_bound=CanonicalRational(num=-1, den=2),
            lower=good.lower,
            upper=good.upper,
        )


def test_decimal_excluding_nine_partial_against_reference() -> None:
    # Convention-sensitive fixture: the Kempner sum omitting digit 9 starts
    # 1/1 + ... + 1/8 + 1/10 + ...; check the two-digit partial exactly.
    digit_set = KempnerDigitSet(base=10, allowed_digits=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    result = enclose_kempner_series(digit_set, 2)
    one_digit = sum((Fraction(1, d) for d in range(1, 9)), Fraction(0))
    two_digit = sum(
        (
            Fraction(1, 10 * first + second)
            for first in range(1, 9)
            for second in range(9)
        ),
        Fraction(0),
    )

    assert result.partial_sum.as_fraction() == one_digit + two_digit
    assert require_series_admission(digit_set, 2) == 8 + 72
    assert MAX_KEMPNER_SERIES_NUMERALS == 50_000
