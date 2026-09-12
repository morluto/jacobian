"""Exact DFA equivalence and least distinguishing words."""

import itertools
from typing import cast

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.languages.regular import operations
from jacobian.math.logic.languages.regular._models import (
    EquivalenceRequest,
    EquivalenceResult,
)
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


def test_empty_alphabet_has_only_the_empty_word() -> None:
    accepting = DFA(
        state_count=2,
        alphabet_size=0,
        transitions=(),
        initial_state=0,
        accepting_states=(0,),
    )
    rejecting = accepting.model_copy(update={"accepting_states": ()})
    equivalent = compute_equivalence(
        EquivalenceRequest(left=accepting, right=accepting)
    )
    assert equivalent.equivalent
    result = compute_equivalence(EquivalenceRequest(left=accepting, right=rejecting))
    assert not result.equivalent
    assert result.distinguishing_word == ()
    assert result.left_state_trace == result.right_state_trace == (0,)


def test_reachable_product_ignores_different_unreachable_states() -> None:
    left = DFA(
        state_count=3,
        alphabet_size=1,
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=1, symbol=0, target=1),
            DFATransition(source=2, symbol=0, target=2),
        ),
        initial_state=0,
        accepting_states=(),
    )
    right = left.model_copy(
        update={
            "accepting_states": (1, 2),
            "transitions": (
                DFATransition(source=0, symbol=0, target=0),
                DFATransition(source=1, symbol=0, target=2),
                DFATransition(source=2, symbol=0, target=1),
            ),
        }
    )
    assert compute_equivalence(EquivalenceRequest(left=left, right=right)).equivalent


def test_native_guard_rejects_partial_and_untyped_dfas() -> None:
    partial = DFA.model_construct(
        state_count=2,
        alphabet_size=1,
        transitions=(DFATransition(source=0, symbol=0, target=0),),
        initial_state=0,
        accepting_states=(),
    )
    valid = DFA(
        state_count=2,
        alphabet_size=1,
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=1, symbol=0, target=1),
        ),
        initial_state=0,
        accepting_states=(),
    )
    with pytest.raises(OperationDomainValidationError, match="one transition"):
        operations.dfa_equivalence(partial, valid)
    with pytest.raises(OperationDomainValidationError, match="canonical DFA"):
        operations.dfa_equivalence(cast(DFA, {"not": "a DFA"}), valid)


def test_native_guard_rejects_incompletely_constructed_values() -> None:
    valid = dfa(((0, 0, 0), (0, 1, 0)), ())
    incomplete = DFA.model_construct()
    incomplete_transition = DFA.model_construct(
        state_count=1,
        alphabet_size=1,
        transitions=(DFATransition.model_construct(),),
        initial_state=0,
        accepting_states=(),
    )
    with pytest.raises(OperationDomainValidationError) as incomplete_error:
        operations.dfa_equivalence(incomplete, valid)
    assert "structurally valid" in str(incomplete_error.value)
    with pytest.raises(OperationDomainValidationError) as transition_error:
        operations.dfa_equivalence(incomplete_transition, valid)
    assert "invalid transition" in str(transition_error.value)


def test_equivalence_honors_request_cancellation_before_execution() -> None:
    class Cancelled:
        def is_set(self) -> bool:
            return True

    left = dfa(((0, 0, 0), (0, 1, 0)), ())
    with (
        request_execution(0.0, cancellation_signal=Cancelled()),
        pytest.raises(OperationExecutionCancelledError),
    ):
        operations.dfa_equivalence(left, left)


def test_equivalence_admits_before_product_bfs(monkeypatch: pytest.MonkeyPatch) -> None:
    left = dfa(
        ((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)),
        (),
    )
    right = left
    monkeypatch.setattr(operations, "MAX_DFA_EQUIVALENCE_PRODUCT_STATES", 1)
    with pytest.raises(OperationResourceAdmissionError, match="product_states"):
        operations.dfa_equivalence(left, right)


def test_result_shape_checks_are_structural_and_do_not_replay() -> None:
    left = dfa(((0, 0, 0), (0, 1, 0)), ())
    right = left
    with pytest.raises(ValueError, match="symbol outside"):
        EquivalenceResult(
            left=left,
            right=right,
            equivalent=False,
            distinguishing_word=(2,),
            left_state_trace=(0, 0),
            right_state_trace=(0, 0),
        )
    with pytest.raises(ValueError, match="start at each initial"):
        EquivalenceResult(
            left=left,
            right=right,
            equivalent=False,
            distinguishing_word=(0,),
            left_state_trace=(1, 0),
            right_state_trace=(0, 0),
        )
