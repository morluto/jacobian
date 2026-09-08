"""Resource refusal cannot refute a DFA count claim."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.regular._models import CountResult
from jacobian.math.logic.languages.regular.operations import verify_accepted_word_count
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def test_count_claim_resource_refusal_propagates() -> None:
    dfa = DFA(
        state_count=1,
        alphabet_size=2,
        initial_state=0,
        accepting_states=(0,),
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=0, symbol=1, target=0),
        ),
    )
    claim = CountResult(dfa=dfa, word_length=200_000, count=0)
    with pytest.raises(OperationResourceAdmissionError):
        verify_accepted_word_count(claim)


def test_transition_profile_resource_refusal_propagates_through_consumers() -> None:
    from jacobian.math.logic.languages.regular._models import (
        TransitionParikhProfileRequest,
    )
    from jacobian.math.logic.languages.regular._tools import (
        compute_transition_parikh_profile,
    )
    from jacobian.math.logic.languages.regular.operations import (
        verify_transition_parikh_profile,
    )
    from jacobian.math.logic.languages.regular.values import (
        AutomatonTransition,
        FiniteLabeledAutomaton,
        TransitionParikhProfile,
    )

    automaton = FiniteLabeledAutomaton(
        state_count=1,
        alphabet_size=9,
        transitions=tuple(
            AutomatonTransition(transition_id=i, source=0, target=0, symbol=i)
            for i in range(9)
        ),
    )
    request = TransitionParikhProfileRequest(
        automaton=automaton, source_state=0, target_state=0, path_length=10
    )
    claim = TransitionParikhProfile(
        automaton=automaton,
        source_state=0,
        target_state=0,
        path_length=10,
        entries=(),
        total_path_count=0,
    )
    with pytest.raises(OperationResourceAdmissionError, match="profile-cell"):
        compute_transition_parikh_profile(request)
    with pytest.raises(OperationResourceAdmissionError, match="profile-cell"):
        verify_transition_parikh_profile(claim)
