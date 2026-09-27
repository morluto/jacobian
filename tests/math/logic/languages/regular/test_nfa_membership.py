"""Bounded exact membership for finite epsilon-NFAs."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.finite_alphabet import FiniteAlphabet
from jacobian.math.logic.languages.regular import nfa_membership
from jacobian.math.logic.languages.regular.values import NFA, NFATransition


def test_membership_follows_epsilon_cycles_and_accepts_empty_word() -> None:
    nfa = NFA(
        state_count=2,
        alphabet_size=1,
        alphabet_id="letters",
        alphabet=FiniteAlphabet(symbols=("a",)),
        transitions=(
            NFATransition(transition_id=0, source=0, symbol=None, target=1),
            NFATransition(transition_id=1, source=1, symbol=None, target=0),
            NFATransition(transition_id=2, source=1, symbol=0, target=1),
        ),
        initial_state=0,
        accepting_states=(1,),
    )
    assert nfa_membership(nfa, ())
    assert nfa_membership(nfa, (0, 0, 0))


def test_membership_rejects_symbols_outside_parented_alphabet() -> None:
    nfa = NFA(
        state_count=1,
        alphabet_size=1,
        alphabet=FiniteAlphabet(symbols=("a",)),
        transitions=(),
        initial_state=0,
        accepting_states=(),
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="word symbols must be in the NFA alphabet",
    ):
        nfa_membership(nfa, (1,))


def test_membership_rejects_partially_constructed_nfa() -> None:
    with pytest.raises(OperationDomainValidationError):
        nfa_membership(NFA.model_construct(), ())
