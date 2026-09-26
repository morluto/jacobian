"""Exact output-tape restriction and an independent path oracle."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    RestrictRationalOutputRequest,
    RestrictRationalOutputResult,
)
from jacobian.math.logic.automata.transducers.output_restriction._tools import TOOLS
from jacobian.math.logic.automata.transducers.output_restriction.operations import (
    restrict_rational_output,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _source() -> RationalTransducer:
    return RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="input",
        output_alphabet_id="output",
        input_alphabet=FiniteAlphabet(symbols=("i", "j")),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        state_count=5,
        initial_states=(0, 4),
        accepting_states=(3,),
        edges=(
            # An empty output label, followed later by a multi-symbol label.
            RationalEdge(source=0, target=1, input_label=(0,), output_label=()),
            RationalEdge(source=0, target=2, input_label=(1,), output_label=(1,)),
            RationalEdge(source=1, target=3, input_label=(), output_label=(1, 1, 0)),
            RationalEdge(source=2, target=3, input_label=(0,), output_label=(0,)),
            RationalEdge(source=4, target=3, input_label=(1,), output_label=(1,)),
        ),
    )


def _output_language() -> DFA:
    # Accept exactly words with an even number of y (symbol 1).
    return DFA(
        state_count=2,
        alphabet_size=2,
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=0, symbol=1, target=1),
            DFATransition(source=1, symbol=0, target=1),
            DFATransition(source=1, symbol=1, target=0),
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def _accepted_paths(
    relation: RationalTransducer,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    outgoing: dict[int, list[RationalEdge]] = {
        state: [] for state in range(relation.state_count)
    }
    for edge in relation.edges:
        outgoing[edge.source].append(edge)
    accepting = set(relation.accepting_states)

    def visit(
        state: int,
        input_word: tuple[int, ...],
        output_word: tuple[int, ...],
        seen: frozenset[int],
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
        if state in accepting:
            yield input_word, output_word
        for edge in outgoing[state]:
            assert edge.target not in seen, "oracle fixture must be acyclic"
            yield from visit(
                edge.target,
                input_word + edge.input_label,
                output_word + edge.output_label,
                seen | {edge.target},
            )

    for initial in relation.initial_states:
        yield from visit(initial, (), (), frozenset({initial}))


def _dfa_accepts(dfa: DFA, word: tuple[int, ...]) -> bool:
    transitions = {(row.source, row.symbol): row.target for row in dfa.transitions}
    state = dfa.initial_state
    for symbol in word:
        state = transitions[(state, symbol)]
    return state in dfa.accepting_states


def _relation_pairs(
    relation: RationalTransducer,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    return set(_accepted_paths(relation))


def _restrict(request: RestrictRationalOutputRequest) -> RestrictRationalOutputResult:
    return restrict_rational_output(
        request.transducer,
        request.output_language,
        request.output_alphabet,
    )


def test_output_restriction_matches_independent_acyclic_path_oracle() -> None:
    source = _source()
    output_language = _output_language()
    request = RestrictRationalOutputRequest(
        transducer=source,
        output_language=output_language,
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
    )

    result = _restrict(request)

    oracle = {
        (input_word, output_word)
        for input_word, output_word in _accepted_paths(source)
        if _dfa_accepts(output_language, output_word)
    }
    assert _relation_pairs(result.restricted) == oracle
    assert oracle == {((0,), (1, 1, 0))}
    assert result.restricted.initial_states == (0, 1)
    assert result.restricted.accepting_states == (5,)
    assert len(result.edge_sources) == len(result.restricted.edges)
    for edge, provenance in zip(
        result.restricted.edges, result.edge_sources, strict=True
    ):
        original = source.edges[provenance.source_edge]
        assert (edge.input_label, edge.output_label) == (
            original.input_label,
            original.output_label,
        )


def test_output_restriction_round_trips_and_is_registered() -> None:
    request = RestrictRationalOutputRequest(
        transducer=_source(),
        output_language=_output_language(),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
    )
    restored_request = RestrictRationalOutputRequest.model_validate_json(
        request.model_dump_json()
    )
    assert _restrict(restored_request) == _restrict(request)

    result = _restrict(request)
    restored_result = RestrictRationalOutputResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored_result == result
    assert len(TOOLS) == 1
    assert TOOLS[0].operation_id == "transducer.relation.restrict_output.compute"
    assert TOOLS[0].run(request) == result


def test_decoded_result_rejects_false_in_range_edge_transport() -> None:
    result = _restrict(
        RestrictRationalOutputRequest(
            transducer=_source(),
            output_language=_output_language(),
            output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        )
    )
    first, second = result.edge_sources[:2]
    forged = result.model_copy(
        update={
            "edge_sources": (
                first.model_copy(update={"source_edge": second.source_edge}),
                *result.edge_sources[1:],
            )
        }
    )
    with pytest.raises(ValidationError, match="edge_transport_mismatch"):
        RestrictRationalOutputResult.model_validate(forged.model_dump())


@pytest.mark.parametrize(
    "transport_patch",
    [
        {"source_edge": 4095},
        {"restricted_edge": 4095},
    ],
)
def test_decoded_result_rejects_out_of_range_edge_transport(
    transport_patch: dict[str, int],
) -> None:
    result = _restrict(
        RestrictRationalOutputRequest(
            transducer=_source(),
            output_language=_output_language(),
            output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        )
    )
    payload = result.model_dump()
    payload["edge_sources"][0].update(transport_patch)
    with pytest.raises(ValidationError):
        RestrictRationalOutputResult.model_validate(payload)


def test_output_restriction_rejects_mismatched_alphabet_context() -> None:
    source = _source().model_copy(
        update={"output_alphabet": FiniteAlphabet(symbols=("a", "b"))}
    )
    request = RestrictRationalOutputRequest.model_construct(
        transducer=source,
        output_language=_output_language(),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
    )
    with pytest.raises(OperationDomainValidationError) as caught:
        restrict_rational_output(
            request.transducer,
            request.output_language,
            request.output_alphabet,
        )
    assert caught.value.errors()[0]["type"] == (
        "rational_transducer.restrict_output.output_context_mismatch"
    )
