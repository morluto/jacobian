"""Tests for finite divisibility poset construction."""

from __future__ import annotations

import pytest

from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.number_theory._divisibility_poset import (
    compute_divisibility_poset,
    divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    DivisibilityPosetRequest,
)


def _set(*values: str) -> FiniteIntegerSet:
    return FiniteIntegerSet(elements=values)


def _request(*values: str) -> DivisibilityPosetRequest:
    return DivisibilityPosetRequest(values=_set(*values))


def test_basic_fixture() -> None:
    """For {2,3,6}, the poset has 2<6 and 3<6, with 2 and 3 incomparable."""
    request = _request("2", "3", "6")
    result = compute_divisibility_poset(request)
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert ("2", "6") in pairs
    assert ("3", "6") in pairs
    assert ("2", "3") not in pairs
    assert ("3", "2") not in pairs


def test_no_reflexive_edges() -> None:
    """Strict order pairs must not be reflexive."""
    request = _request("4", "8", "16")
    result = compute_divisibility_poset(request)
    assert all(pair.lower != pair.upper for pair in result.strict_order_pairs)


def test_singleton_set() -> None:
    """A single-element set has no strict order pairs."""
    request = _request("7")
    result = compute_divisibility_poset(request)
    assert result.strict_order_pairs == ()


def test_chain() -> None:
    """A chain of divisibility."""
    request = _request("2", "4", "8")
    result = compute_divisibility_poset(request)
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert ("2", "4") in pairs
    assert ("2", "8") in pairs
    assert ("4", "8") in pairs
    assert ("4", "2") not in pairs


def test_incomparable_primes() -> None:
    """Two distinct primes are incomparable."""
    request = _request("5", "7")
    result = compute_divisibility_poset(request)
    assert result.strict_order_pairs == ()


def test_one_divides_everything() -> None:
    """1 divides every positive integer."""
    request = _request("1", "2", "3")
    result = compute_divisibility_poset(request)
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert ("1", "2") in pairs
    assert ("1", "3") in pairs


def test_transitive_closure() -> None:
    """The result includes both direct and transitive pairs."""
    request = _request("2", "6", "12")
    result = compute_divisibility_poset(request)
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert ("2", "6") in pairs
    assert ("6", "12") in pairs
    assert ("2", "12") in pairs


def test_native_admission_rejects_zero_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        divisibility_poset(_set("0", "2"))
    with pytest.raises(ValueError, match="between 1"):
        divisibility_poset(_set())


def test_native_admission_matches_finite_poset_envelope() -> None:
    with pytest.raises(ValueError, match="between 1 and 64"):
        divisibility_poset(_set(*(str(value) for value in range(1, 66))))
    with pytest.raises(ValueError, match="no longer than 32"):
        divisibility_poset(_set("1" * 33))
