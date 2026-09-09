"""Exact DFA equivalence and least distinguishing words."""

import itertools

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.languages.regular._models import EquivalenceRequest
from jacobian.math.logic.languages.regular._tools import compute_equivalence
from jacobian.math.logic.languages.regular.operations import dfa_run
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def dfa(
    transitions: tuple[tuple[int, int, int], ...],
    accepting: tuple[int, ...],
    *,
    alphabet_size: int = 2,
) -> DFA:
    return DFA(
        state_count=max(source for source, _, _ in transitions) + 1,
        alphabet_size=alphabet_size,
        transitions=tuple(
            DFATransition(source=source, symbol=symbol, target=target)
            for source, symbol, target in transitions
        ),
        initial_state=0,
        accepting_states=accepting,
    )


def test_equivalent_nonisomorphic_presentations() -> None:
    left = dfa(((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)), (1,))
    right = dfa(
        (
            (0, 0, 2),
            (0, 1, 1),
            (1, 0, 2),
            (1, 1, 1),
            (2, 0, 2),
            (2, 1, 1),
        ),
        (1,),
    )
    result = compute_equivalence(EquivalenceRequest(left=left, right=right))
    assert result.equivalent
    assert result.distinguishing_word is None


def test_shortest_lexicographically_least_word_and_replayable_traces() -> None:
    left = dfa(((0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)), (1,))
    right = dfa(((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)), (1,))
    result = compute_equivalence(EquivalenceRequest(left=left, right=right))
    assert not result.equivalent
    assert result.distinguishing_word == (0,)
    assert result.left_state_trace == (0, 1)
    assert result.right_state_trace == (0, 0)
    assert (
        dfa_run(left, result.distinguishing_word)[0]
        != dfa_run(right, result.distinguishing_word)[0]
    )
    for length in range(len(result.distinguishing_word) + 1):
        for word in itertools.product(range(2), repeat=length):
            if word >= result.distinguishing_word and length == len(
                result.distinguishing_word
            ):
                break
            assert dfa_run(left, word)[0] == dfa_run(right, word)[0]


def test_alphabet_mismatch_is_a_domain_error() -> None:
    binary = dfa(((0, 0, 0), (0, 1, 0)), (), alphabet_size=2)
    unary = dfa(((0, 0, 0),), (), alphabet_size=1)
    with pytest.raises(OperationDomainValidationError, match="same ordered alphabet"):
        compute_equivalence(EquivalenceRequest(left=binary, right=unary))
