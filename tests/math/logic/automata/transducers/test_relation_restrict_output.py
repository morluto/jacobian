"""Exact output-tape restriction and an independent path oracle."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from pydantic import ConfigDict, ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    MAX_RESTRICT_OUTPUT_WORK,
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
    with pytest.raises(ValidationError) as exc_info:
        RestrictRationalOutputResult.model_validate(forged.model_dump())
    assert (
        exc_info.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_transport_mismatch"
    )


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


def test_native_transition_rows_must_be_canonical_types() -> None:
    """An attribute-only check admits foreign mutable rows.

    ``DFA.model_construct`` can hold any object with ``source``/``symbol``/
    ``target``. Because the DFA is retained in the result without nested
    revalidation, accepting such a row would put a noncanonical mutable object
    inside a declared typed result.
    """
    # The shape matches the source's output alphabet, so the request reaches
    # the transition-row check rather than being refused for a size mismatch.
    forged = DFA.model_construct(
        state_count=2,
        alphabet_size=2,
        initial_state=0,
        accepting_states=(1,),
        transitions=tuple(
            SimpleNamespace(source=state, symbol=symbol, target=state)
            for state in range(2)
            for symbol in range(2)
        ),
    )
    with pytest.raises(OperationDomainValidationError) as refusal:
        restrict_rational_output(_source(), forged, FiniteAlphabet(symbols=("x", "y")))
    assert (
        refusal.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.dfa_transition_invalid"
    )


def test_native_relation_edge_rows_must_be_canonical_types() -> None:
    """The same hole exists on the relation side, which the result retains."""
    relation = _source()
    forged = relation.model_copy(
        update={
            "edges": (
                SimpleNamespace(
                    source=0, target=1, input_label=(0,), output_label=(0,)
                ),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as refusal:
        restrict_rational_output(
            forged, _output_language(), FiniteAlphabet(symbols=("x", "y"))
        )
    assert (
        refusal.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_invalid"
    )


def test_edge_transport_replay_is_charged_against_the_admitted_work_envelope() -> None:
    """The decoder's replay is admitted work, so it is charged before it runs.

    Confirming edge transport replays the output language once per edge. The
    carriers bound that at roughly 2.1 million steps, which is inside the
    operation's published envelope, so the charge is what makes the decoder's
    cost accounted for rather than incidental.
    """
    source = _source()
    language = _output_language()
    result = restrict_rational_output(
        source, language, FiniteAlphabet(symbols=("x", "y"))
    )
    replay_cells = sum(
        len(result.restricted.edges[t.restricted_edge].output_label)
        for t in result.edge_sources
    )
    assert replay_cells <= MAX_RESTRICT_OUTPUT_WORK

    # A genuine result still decodes, so the charge refuses nothing legitimate.
    decoded = RestrictRationalOutputResult.model_validate(result.model_dump())
    assert decoded.product_states == result.product_states


@pytest.mark.parametrize("through_adapter", (False, True))
@pytest.mark.parametrize("row_kind", ("dfa", "relation"))
def test_mutable_row_subclasses_are_rejected_at_native_admission(
    through_adapter: bool, row_kind: str
) -> None:
    class MutableTransition(DFATransition):
        model_config = ConfigDict(frozen=False)

    class MutableEdge(RationalEdge):
        model_config = ConfigDict(frozen=False)

    source = _source()
    language = _output_language()
    if row_kind == "dfa":
        transition = MutableTransition(**language.transitions[0].model_dump())
        transition.target = 1
        assert transition.target == 1
        transition.target = 0
        language = language.model_copy(
            update={"transitions": (transition, *language.transitions[1:])}
        )
    else:
        edge = MutableEdge(**source.edges[0].model_dump())
        edge.target = 2
        assert edge.target == 2
        edge.target = 1
        source = source.model_copy(update={"edges": (edge, *source.edges[1:])})
    request = RestrictRationalOutputRequest.model_construct(
        transducer=source,
        output_language=language,
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        if through_adapter:
            TOOLS[0].run(request)
        else:
            _restrict(request)
    reason = "dfa_transition_invalid" if row_kind == "dfa" else "edge_invalid"
    assert (
        error.value.errors()[0]["type"]
        == f"rational_transducer.restrict_output.{reason}"
    )
