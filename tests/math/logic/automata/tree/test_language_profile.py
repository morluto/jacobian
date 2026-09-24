from jacobian.math.logic.automata.tree._models import TreeLanguageProfileRequest
from jacobian.math.logic.automata.tree.operations import (
    accepted_tree_count,
    run_tree_automaton,
    tree_language_profile,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def test_language_profile_returns_minimum_witness_per_reachable_final():
    machine = BottomUpTreeAutomaton(
        state_count=3,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=2),
        ),
        final_states=(1, 2),
    )

    profile = tree_language_profile(TreeLanguageProfileRequest(automaton=machine))

    assert profile.reachable_states == (0, 1, 2)
    assert profile.reachable_final_states == (1, 2)
    assert tuple(w.state for w in profile.witnesses) == (1, 2)
    assert profile.witnesses[0].tree == RankedTree(
        symbol=1, children=(RankedTree(symbol=0),)
    )
    assert profile.witnesses[1].tree == RankedTree(
        symbol=1, children=(profile.witnesses[0].tree,)
    )
    assert profile.empty is False


def test_language_profile_identifies_empty_language_without_tree_enumeration():
    machine = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0,),
        transitions=(TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),),
        final_states=(1,),
    )

    profile = tree_language_profile(TreeLanguageProfileRequest(automaton=machine))

    assert profile.reachable_states == (0,)
    assert profile.unreachable_states == (1,)
    assert profile.reachable_final_states == ()
    assert profile.witnesses == ()
    assert profile.empty is True


def test_nondeterministic_runs_do_not_change_language_profile_or_tree_count():
    machine = BottomUpTreeAutomaton(
        state_count=3,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=2),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=2),
        ),
        final_states=(2,),
    )
    leaf = RankedTree(symbol=0)
    accepted = (RankedTree(symbol=1, children=(leaf,)),)

    profile = tree_language_profile(TreeLanguageProfileRequest(automaton=machine))
    enumerated = tuple(tree for tree in accepted if set(run_tree_automaton(machine, tree)) & {2})
    count = accepted_tree_count(machine, 2)

    assert enumerated == accepted
    assert profile.reachable_final_states == (2,)
    assert profile.witnesses[0].tree == accepted[0]
    assert count == 1
