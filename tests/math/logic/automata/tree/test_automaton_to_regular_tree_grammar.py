from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.tree._models import (
    TreeAutomatonToRegularTreeGrammarRequest,
    TreeAutomatonToRegularTreeGrammarResult,
)
from jacobian.math.logic.automata.tree._tools import TOOLS
from jacobian.math.logic.automata.tree.operations import (
    regular_tree_grammar_to_automaton,
    run_tree_automaton,
    tree_automaton_to_regular_tree_grammar,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    RankedTree,
    RegularTreeGrammar,
    RegularTreeProduction,
    TreeAutomatonTransition,
)


def _automaton() -> BottomUpTreeAutomaton:
    # Constants a,b; f(a,a) and f(b,b) are accepted, but f(a,b) is not.
    return BottomUpTreeAutomaton(
        state_count=3,
        arity=(0, 0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=2),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=1),
        ),
        final_states=(1, 2),
    )


def _trees_by_size(max_size: int) -> tuple[RankedTree, ...]:
    by_size: dict[int, list[RankedTree]] = {
        1: [RankedTree(symbol=0), RankedTree(symbol=1)]
    }
    for size in range(2, max_size + 1):
        trees: list[RankedTree] = []
        for symbol, rank in enumerate((0, 0, 2)):
            if rank == 0:
                continue
            for left_size in range(1, size):
                right_size = size - 1 - left_size
                if right_size not in by_size or left_size not in by_size:
                    continue
                trees.extend(
                    RankedTree(symbol=symbol, children=(left, right))
                    for left, right in product(by_size[left_size], by_size[right_size])
                )
        by_size[size] = trees
    return tuple(tree for size in sorted(by_size) for tree in by_size[size])


def _grammar_states(grammar: RegularTreeGrammar, tree: RankedTree) -> frozenset[int]:
    child_states = tuple(_grammar_states(grammar, child) for child in tree.children)
    return frozenset(
        production.nonterminal
        for production in grammar.productions
        if production.symbol == tree.symbol
        and len(production.children) == len(child_states)
        and all(
            state in possible
            for state, possible in zip(production.children, child_states, strict=True)
        )
    )


def test_multiple_final_states_preserve_exact_language_exhaustively() -> None:
    automaton = _automaton()
    grammar = tree_automaton_to_regular_tree_grammar(automaton)
    assert grammar.nonterminal_count == 4
    assert grammar.start_nonterminal == 3
    assert (
        any(
            row.symbol == 2 and row.child_states == (0, 1)
            for row in automaton.transitions
        )
        is False
    )
    assert grammar.arity == automaton.arity

    generated_automaton = regular_tree_grammar_to_automaton(grammar)
    for tree in _trees_by_size(5):
        accepts_from_source = bool(
            set(run_tree_automaton(automaton, tree)) & set(automaton.final_states)
        )
        accepts_from_grammar = grammar.start_nonterminal in _grammar_states(
            grammar, tree
        )
        accepts_from_roundtrip = bool(
            set(run_tree_automaton(generated_automaton, tree))
            & set(generated_automaton.final_states)
        )
        assert accepts_from_grammar == accepts_from_source
        assert accepts_from_roundtrip == accepts_from_source


def test_single_final_state_preserves_state_ids_without_synthetic_start() -> None:
    source = BottomUpTreeAutomaton(
        state_count=3,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=2),
        ),
        final_states=(2,),
    )

    grammar = tree_automaton_to_regular_tree_grammar(source)

    assert grammar.nonterminal_count == 3
    assert grammar.start_nonterminal == 2
    assert len(grammar.productions) == 1


@pytest.mark.parametrize(
    "automaton",
    [
        BottomUpTreeAutomaton(
            state_count=3,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            ),
            final_states=(),
        ),
        BottomUpTreeAutomaton(
            state_count=64,
            arity=(1,),
            transitions=tuple(
                TreeAutomatonTransition(
                    symbol=0, child_states=(state,), target_state=state
                )
                for state in range(64)
            ),
            final_states=(0, 1),
        ),
    ],
)
def test_empty_language_returns_empty_grammar(automaton: BottomUpTreeAutomaton) -> None:
    grammar = tree_automaton_to_regular_tree_grammar(automaton)
    assert grammar.nonterminal_count == 1
    assert grammar.productions == ()
    assert grammar.arity == automaton.arity


def test_unrepresentable_synthetic_start_is_a_typed_refusal() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=64,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        ),
        final_states=(0, 1),
    )

    with pytest.raises(OperationResourceAdmissionError):
        tree_automaton_to_regular_tree_grammar(automaton)


def test_production_copy_above_grammar_bound_is_refused_before_expansion() -> None:
    transitions = [
        TreeAutomatonTransition(symbol=0, child_states=(), target_state=state)
        for state in range(64)
    ]
    transitions.extend(
        TreeAutomatonTransition(symbol=1, child_states=(child,), target_state=target)
        for child in range(1, 64)
        for target in range(64)
    )
    automaton = BottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 1),
        transitions=tuple(transitions),
        final_states=tuple(range(64)),
    )

    with pytest.raises(OperationResourceAdmissionError):
        tree_automaton_to_regular_tree_grammar(automaton)


def test_full_4096_production_output_is_admitted_at_boundary() -> None:
    transitions = [TreeAutomatonTransition(symbol=0, child_states=(), target_state=0)]
    transitions.extend(
        TreeAutomatonTransition(symbol=1, child_states=(child,), target_state=target)
        for child in range(64)
        for target in range(64)
        if (child, target) != (0, 0)
    )
    automaton = BottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 1),
        transitions=tuple(transitions),
        final_states=(0,),
    )

    grammar = tree_automaton_to_regular_tree_grammar(automaton)

    assert len(grammar.productions) == 4096
    assert grammar.start_nonterminal == 0


def test_source_bound_result_and_operation_declaration() -> None:
    request = TreeAutomatonToRegularTreeGrammarRequest(automaton=_automaton())
    grammar = tree_automaton_to_regular_tree_grammar(request.automaton)
    result = TreeAutomatonToRegularTreeGrammarResult._from_kernel(
        automaton=request.automaton, grammar=grammar
    )
    assert (
        TreeAutomatonToRegularTreeGrammarResult.model_validate(result.model_dump())
        == result
    )
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "tree_automaton.to_regular_tree_grammar.compute"
    )
    assert operation.result_type is TreeAutomatonToRegularTreeGrammarResult


def test_unproductive_finals_do_not_require_a_synthetic_start() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(62, 63),
    )

    grammar = tree_automaton_to_regular_tree_grammar(automaton)

    assert grammar.nonterminal_count == 1
    assert grammar.productions == ()


def test_nonproductive_final_larger_than_productive_one_keeps_state_start() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=64,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0, 63),
    )

    grammar = tree_automaton_to_regular_tree_grammar(automaton)

    assert grammar.nonterminal_count == 64
    assert grammar.start_nonterminal == 0
    assert grammar.productions == (
        RegularTreeProduction(nonterminal=0, symbol=0, children=()),
    )
    accepted = RankedTree(symbol=0)
    assert grammar.start_nonterminal in _grammar_states(grammar, accepted)


def test_only_productive_finals_contribute_root_rules() -> None:
    # State 0 accepts a constant, state 1 accepts f(0); state 5 is final but has
    # no incoming transition. Two productive finals force a synthetic start,
    # whose copied root rules must come only from the productive finals.
    automaton = BottomUpTreeAutomaton(
        state_count=6,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
        ),
        final_states=(0, 1, 5),
    )

    grammar = tree_automaton_to_regular_tree_grammar(automaton)

    assert grammar.nonterminal_count == 7
    assert grammar.start_nonterminal == 6
    assert {
        (production.nonterminal, production.symbol, production.children)
        for production in grammar.productions
    } == {
        (0, 0, ()),
        (1, 1, (0,)),
        (6, 0, ()),
        (6, 1, (0,)),
    }


def test_mixed_productive_finals_preserve_language_exhaustively() -> None:
    # States 1 and 2 are productive final states; state 3 is final but has no
    # incoming transition. Restricting the synthetic start to productive finals
    # must leave the accepted language unchanged.
    automaton = BottomUpTreeAutomaton(
        state_count=4,
        arity=(0, 0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=2),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=1),
        ),
        final_states=(1, 2, 3),
    )

    grammar = tree_automaton_to_regular_tree_grammar(automaton)

    assert grammar.nonterminal_count == 5
    assert grammar.start_nonterminal == 4
    generated_automaton = regular_tree_grammar_to_automaton(grammar)
    for tree in _trees_by_size(5):
        accepts_from_source = bool(
            set(run_tree_automaton(automaton, tree)) & set(automaton.final_states)
        )
        accepts_from_grammar = grammar.start_nonterminal in _grammar_states(
            grammar, tree
        )
        accepts_from_roundtrip = bool(
            set(run_tree_automaton(generated_automaton, tree))
            & set(generated_automaton.final_states)
        )
        assert accepts_from_grammar == accepts_from_source
        assert accepts_from_roundtrip == accepts_from_source
