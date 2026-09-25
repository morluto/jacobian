"""Exact input-language restriction for finite rational relations."""

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
    restrict_rational_input,
)
from jacobian.math.logic.automata.transducers._models import (
    RationalRelationRestrictInputRequest,
)
from jacobian.math.logic.automata.transducers._tools import (
    TOOLS,
    compute_relation_restrict_input,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _even_length_dfa(alphabet: FiniteAlphabet) -> DFA:
    return DFA(
        state_count=2,
        alphabet_size=2,
        alphabet=alphabet,
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=0, symbol=1, target=1),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=1, symbol=1, target=0),
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def _accepted_pairs(
    relation: RationalTransducer, dfa: DFA
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Independent finite accepting-path oracle for an acyclic fixture."""
    rows: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    outgoing: list[list[RationalEdge]] = [[] for _ in range(relation.state_count)]
    for edge in relation.edges:
        outgoing[edge.source].append(edge)
    dfa_edges = {(edge.source, edge.symbol): edge.target for edge in dfa.transitions}
    for initial in relation.initial_states:
        pending = [(initial, (), ())]
        while pending:
            state, input_word, output_word = pending.pop()
            if state in relation.accepting_states:
                dfa_state = dfa.initial_state
                for symbol in input_word:
                    dfa_state = dfa_edges[(dfa_state, symbol)]
                if dfa_state in dfa.accepting_states:
                    rows.add((input_word, output_word))
            for edge in outgoing[state]:
                pending.append(
                    (
                        edge.target,
                        input_word + edge.input_label,
                        output_word + edge.output_label,
                    )
                )
    return rows


def test_restrict_input_matches_independent_path_oracle() -> None:
    alphabet = FiniteAlphabet(symbols=("a", "b"))
    relation = RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet=alphabet,
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        state_count=4,
        initial_states=(0,),
        accepting_states=(3,),
        edges=(
            # Multi-symbol input label followed by one symbol: odd total length.
            RationalEdge(source=0, target=1, input_label=(0, 1), output_label=(1,)),
            RationalEdge(source=1, target=3, input_label=(0,), output_label=(1, 0)),
            # Epsilon-input edge followed by one symbol: odd total length.
            RationalEdge(source=0, target=2, input_label=(), output_label=(0,)),
            RationalEdge(source=2, target=3, input_label=(1,), output_label=(1,)),
            # Even length and therefore retained.
            RationalEdge(source=0, target=3, input_label=(1, 1), output_label=(0,)),
        ),
    )
    dfa = _even_length_dfa(alphabet)

    restricted, product_states, source_edge_indices = restrict_rational_input(
        relation, dfa
    )
    result_pairs = _accepted_pairs(restricted, dfa)
    assert result_pairs == _accepted_pairs(relation, dfa)
    assert result_pairs == {((1, 1), (0,))}
    assert product_states[0] == (0, 0)
    assert len(source_edge_indices) == len(restricted.edges)
    assert all(
        restricted.edges[i].input_label == relation.edges[source_index].input_label
        and restricted.edges[i].output_label
        == relation.edges[source_index].output_label
        for i, source_index in enumerate(source_edge_indices)
    )


def test_restriction_catalog_operation_and_wrong_parent_rejection() -> None:
    alphabet = FiniteAlphabet(symbols=("a",))
    relation = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet=alphabet,
        output_alphabet=FiniteAlphabet(symbols=("x",)),
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=(RationalEdge(source=0, target=0, input_label=(0,), output_label=(0,)),),
    )
    dfa = DFA(
        state_count=1,
        alphabet_size=1,
        alphabet=alphabet,
        transitions=(DFATransition(source=0, symbol=0, target=0),),
        initial_state=0,
        accepting_states=(0,),
    )
    result = compute_relation_restrict_input(
        RationalRelationRestrictInputRequest(transducer=relation, dfa=dfa)
    )
    assert result.restricted == restrict_rational_input(relation, dfa)[0]
    assert type(result).model_validate(result.model_dump(), strict=True) == result
    assert "transducer.relation.restrict_input.compute" in {
        tool.operation_id for tool in TOOLS
    }

    wrong_parent = dfa.model_copy(
        update={"alphabet": FiniteAlphabet(symbols=("other",))}
    )
    with pytest.raises(OperationDomainValidationError):
        restrict_rational_input(relation, wrong_parent)


def _cycle_relation(state_count: int) -> RationalTransducer:
    return RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=1,
        input_alphabet=FiniteAlphabet(symbols=("a", "b")),
        output_alphabet=FiniteAlphabet(symbols=("x",)),
        state_count=state_count,
        initial_states=(0,),
        accepting_states=(tuple(range(state_count))),
        edges=tuple(
            RationalEdge(
                source=state,
                target=(state + 1) % state_count if symbol == 0 else state,
                input_label=(symbol,),
                output_label=(0,),
            )
            for state in range(state_count)
            for symbol in range(2)
        ),
    )


def _cycle_dfa(state_count: int) -> DFA:
    alphabet = FiniteAlphabet(symbols=("a", "b"))
    return DFA(
        state_count=state_count,
        alphabet_size=2,
        alphabet=alphabet,
        transitions=tuple(
            DFATransition(
                source=state,
                symbol=symbol,
                target=state if symbol == 0 else (state + 1) % state_count,
            )
            for state in range(state_count)
            for symbol in range(2)
        ),
        initial_state=0,
        accepting_states=tuple(range(state_count)),
    )


def test_product_state_limit_accepts_at_and_rejects_above_bound() -> None:
    accepted, states, _ = restrict_rational_input(_cycle_relation(8), _cycle_dfa(8))
    assert accepted.state_count == len(states) == 64
    with pytest.raises(OperationResourceAdmissionError):
        restrict_rational_input(_cycle_relation(9), _cycle_dfa(8))


def test_product_edge_limit_accepts_at_and_rejects_above_bound() -> None:
    alphabet = FiniteAlphabet(symbols=("a", "b"))
    dfa = DFA(
        state_count=2,
        alphabet_size=2,
        alphabet=alphabet,
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=0, symbol=1, target=0),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=1, symbol=1, target=1),
        ),
        initial_state=0,
        accepting_states=(0, 1),
    )

    def relation(edge_count: int) -> RationalTransducer:
        return RationalTransducer(
            input_alphabet_size=2,
            output_alphabet_size=1,
            input_alphabet=alphabet,
            output_alphabet=FiniteAlphabet(symbols=("x",)),
            state_count=1,
            initial_states=(0,),
            accepting_states=(0,),
            edges=tuple(
                RationalEdge(
                    source=0,
                    target=0,
                    input_label=(0,),
                    output_label=(0,),
                )
                for _ in range(edge_count)
            ),
        )

    at_limit, _, _ = restrict_rational_input(relation(2048), dfa)
    assert len(at_limit.edges) == 4096
    with pytest.raises(OperationResourceAdmissionError):
        restrict_rational_input(relation(2049), dfa)
