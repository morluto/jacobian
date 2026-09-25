"""Exact finite-factor language construction over free-generator alphabets."""

from itertools import product
from random import Random

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.free_algebras._models import (
    FreeAlgebraFactorAvoidanceDFA,
    FreeAlgebraFactorAvoidanceRequest,
)
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras.operations import factor_avoidance_dfa
from jacobian.math.logic.languages.regular.operations import dfa_run


def _request(
    alphabet: list[str], patterns: list[list[str]]
) -> FreeAlgebraFactorAvoidanceRequest:
    return FreeAlgebraFactorAvoidanceRequest(
        alphabet=tuple(alphabet),
        forbidden_factors=tuple(tuple(word) for word in patterns),
    )


def test_factor_avoidance_dfa_matches_direct_factor_search_and_round_trips() -> None:
    result = factor_avoidance_dfa(("x", "y"), (("x", "y"), ("y", "x", "y")))
    restored = FreeAlgebraFactorAvoidanceDFA.model_validate(result.model_dump())
    assert restored.forbidden_factors == (("x", "y"), ("y", "x", "y"))

    for length in range(7):
        for word in product(range(2), repeat=length):
            named = tuple(restored.alphabet[symbol] for symbol in word)
            expected = not any(
                named[start : start + len(pattern)] == pattern
                for pattern in restored.forbidden_factors
                for start in range(len(named) - len(pattern) + 1)
            )
            accepted, _ = dfa_run(restored.dfa, word)
            assert accepted is expected


def test_empty_family_and_empty_pattern_have_exact_monoid_edges() -> None:
    full = factor_avoidance_dfa((), ())
    assert full.dfa.state_count == 1
    assert full.dfa.accepting_states == (0,)
    assert dfa_run(full.dfa, ())[0]

    empty = factor_avoidance_dfa(("x",), ((),))
    assert empty.forbidden_factors == ((),)
    assert empty.dfa.state_count == 1
    assert empty.dfa.accepting_states == ()
    assert not dfa_run(empty.dfa, ())[0]
    assert not dfa_run(empty.dfa, (0,))[0]


def test_published_operation_output_is_usable_by_regular_language_consumer() -> None:
    invocation = invoke_operation(
        "free_algebra.factor_avoidance.dfa.compute",
        {"alphabet": ["x", "y"], "forbidden_factors": [["x", "y"]]},
        Catalog.open(),
    )
    carrier = FreeAlgebraFactorAvoidanceDFA.model_validate(invocation.output)
    assert dfa_run(carrier.dfa, (1, 0))[0]
    assert not dfa_run(carrier.dfa, (0, 1))[0]


def test_patterns_containing_another_forbidden_factor_are_removed() -> None:
    result = factor_avoidance_dfa(("x", "y"), (("x", "y", "x"), ("y",), ("y",)))
    assert result.forbidden_factors == (("y",), ("x", "y", "x"))
    for length in range(5):
        for word in product(range(2), repeat=length):
            accepted, _ = dfa_run(result.dfa, word)
            assert accepted is (1 not in word)


def test_multiple_pattern_families_match_independent_membership_oracle() -> None:
    rng = Random(1889)
    candidates = [
        word for length in range(1, 4) for word in product(("a", "b"), repeat=length)
    ]
    for _ in range(50):
        supplied = [list(word) for word in candidates if rng.randrange(4) == 0]
        result = factor_avoidance_dfa(("a", "b"), tuple(tuple(w) for w in supplied))
        patterns = tuple(tuple(word) for word in supplied)
        for length in range(6):
            for word in product(range(2), repeat=length):
                named = tuple(result.alphabet[symbol] for symbol in word)
                expected = not any(
                    named[start : start + len(pattern)] == pattern
                    for pattern in patterns
                    for start in range(len(named) - len(pattern) + 1)
                )
                assert dfa_run(result.dfa, word)[0] is expected


def test_factor_avoidance_checks_cancellation_during_normalization_and_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def checkpoint(stage: str) -> None:
        observed.append(stage)

    monkeypatch.setattr(operations, "request_checkpoint", checkpoint)
    factor_avoidance_dfa(("x",), tuple(("x",) * i for i in range(1, 21)))
    assert "during factor-avoidance normalization" in observed
    assert "during factor-avoidance transition construction" in observed


def test_factor_family_rejects_dfa_larger_than_shared_carrier() -> None:
    accepted = factor_avoidance_dfa(("x",), (("x",) * 63,))
    assert accepted.dfa.state_count == 64

    request = _request(["x"], [["x"] * 64])
    with pytest.raises(OperationResourceAdmissionError, match="64-state"):
        factor_avoidance_dfa(request.alphabet, request.forbidden_factors)


def test_prefix_state_preflight_accounts_for_shared_pattern_prefixes() -> None:
    alphabet = [f"g{index:02}" for index in range(26)]
    shared_prefix = alphabet[:3]
    patterns = [[*shared_prefix, letter] for letter in alphabet]
    result = factor_avoidance_dfa(tuple(alphabet), tuple(tuple(w) for w in patterns))
    assert result.dfa.state_count == 5
