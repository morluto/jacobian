"""Native profiles retain validated canonical automaton refinements."""

import json
from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.tree import (
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    ReachableStateProfile,
    TreeAutomatonTransition,
    TreeLanguageProfile,
    complement_tree_automaton,
    complete_deterministic_tree_automaton,
    reachable_state_profile,
    run_tree_automaton,
    tree_language_profile,
)
from jacobian.math.logic.automata.tree._models import TreeAutomatonComplementRequest

type ProfileFunction = Callable[
    [BottomUpTreeAutomaton], ReachableStateProfile | TreeLanguageProfile
]


@pytest.mark.parametrize("profile", [reachable_state_profile, tree_language_profile])
@pytest.mark.parametrize(
    "automaton_type",
    [
        BottomUpTreeAutomaton,
        DeterministicBottomUpTreeAutomaton,
        CompleteDeterministicBottomUpTreeAutomaton,
    ],
)
def test_profile_preserves_each_canonical_carrier(
    profile: ProfileFunction, automaton_type: type[BottomUpTreeAutomaton]
) -> None:
    machine = automaton_type(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0,),
    )

    result = profile(machine)

    assert type(result.automaton) is automaton_type
    assert result.automaton == machine
    assert result.reachable_states == (0,)
    assert result.unreachable_states == ()
    if isinstance(result.automaton, DeterministicBottomUpTreeAutomaton):
        completed = complete_deterministic_tree_automaton(result.automaton)
        assert completed.sink_state is None
        assert completed.completed.final_states == (0,)


@pytest.mark.parametrize("profile", [reachable_state_profile, tree_language_profile])
@pytest.mark.parametrize("empty_alphabet", [False, True])
def test_completion_profile_composes_with_complement(
    profile: ProfileFunction, empty_alphabet: bool
) -> None:
    # With a nullary symbol and no source transitions, completion adds a
    # nonfinal sink. Its complement accepts the only ground tree: the leaf.
    source = DeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=() if empty_alphabet else (0,),
        transitions=(),
        final_states=(),
    )
    completed = complete_deterministic_tree_automaton(source).completed
    retained = profile(completed).automaton

    assert isinstance(retained, CompleteDeterministicBottomUpTreeAutomaton)
    complement = complement_tree_automaton(retained).complement
    assert complement.final_states == tuple(range(completed.state_count))
    if empty_alphabet:
        assert complement.arity == ()
        assert complement.transitions == ()
    else:
        leaf = RankedTree(symbol=0)
        assert not set(run_tree_automaton(retained, leaf)) & set(retained.final_states)
        assert set(run_tree_automaton(complement, leaf)) & set(complement.final_states)

    # Wire payloads carry the structural fields; the consumer's request schema
    # re-establishes completeness rather than requiring a Python subtype tag.
    decoded = TreeAutomatonComplementRequest.model_validate_json(
        json.dumps({"automaton": retained.model_dump(mode="json")})
    )
    assert complement_tree_automaton(decoded.automaton).complement == complement


@pytest.mark.parametrize("profile", [reachable_state_profile, tree_language_profile])
@pytest.mark.parametrize("incomplete", [False, True])
def test_profile_revalidates_the_claimed_refinement(
    profile: ProfileFunction, incomplete: bool
) -> None:
    # Both carriers have valid base fields but violate their refined contract.
    automaton_type = (
        CompleteDeterministicBottomUpTreeAutomaton
        if incomplete
        else DeterministicBottomUpTreeAutomaton
    )
    rows = (
        ()
        if incomplete
        else (
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        )
    )
    forged = automaton_type.model_construct(
        state_count=2, arity=(0,), transitions=rows, final_states=()
    )

    with pytest.raises(OperationDomainValidationError, match="canonical bottom-up"):
        profile(forged)
