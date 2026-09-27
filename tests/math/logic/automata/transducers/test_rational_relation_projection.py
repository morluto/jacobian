"""Exact input/output projections of finite rational relations."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
    invert_rational,
    project_rational_relation,
)
from jacobian.math.logic.automata.transducers._tools import TOOLS
from jacobian.math.logic.languages.regular.operations import nfa_membership
from jacobian.math.logic.languages.regular.values import NFA


def _relation() -> RationalTransducer:
    return RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="letters",
        output_alphabet_id="bits",
        input_alphabet=FiniteAlphabet(symbols=("a", "b")),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        state_count=4,
        initial_states=(0, 2),
        accepting_states=(3,),
        edges=(
            RationalEdge(source=0, target=1, input_label=(0, 1), output_label=(0, 1)),
            RationalEdge(source=1, target=3, input_label=(), output_label=(1,)),
            RationalEdge(source=0, target=3, input_label=(0, 1), output_label=(1,)),
            RationalEdge(source=2, target=3, input_label=(), output_label=(0,)),
        ),
    )


def _projected_path_words(
    relation: RationalTransducer, *, tape: str
) -> set[tuple[int, ...]]:
    """Independent acyclic path enumerator used as a finite exact oracle."""
    edge_rows: dict[int, list[RationalEdge]] = {}
    for edge in relation.edges:
        edge_rows.setdefault(edge.source, []).append(edge)
    selected: set[tuple[int, ...]] = set()

    def visit(state: int, word: tuple[int, ...]) -> None:
        if state in relation.accepting_states:
            selected.add(word)
        for edge in edge_rows.get(state, ()):
            label = edge.input_label if tape == "input" else edge.output_label
            visit(edge.target, word + label)

    for state in relation.initial_states:
        visit(state, ())
    return selected


def _words(alphabet_size: int, max_length: int) -> list[tuple[int, ...]]:
    return [
        word
        for length in range(max_length + 1)
        for word in product(range(alphabet_size), repeat=length)
    ]


@pytest.mark.parametrize("tape", ["input", "output"])
def test_projection_matches_independent_accepting_path_oracle(tape: str) -> None:
    relation = _relation()
    result = project_rational_relation(relation, tape)
    expected = _projected_path_words(relation, tape=tape)
    assert isinstance(result, NFA)
    expected_alphabet = (
        relation.input_alphabet if tape == "input" else relation.output_alphabet
    )
    expected_identity = (
        relation.input_alphabet_id if tape == "input" else relation.output_alphabet_id
    )
    assert result.alphabet == expected_alphabet
    assert result.alphabet_id == expected_identity
    assert result.accepting_states == relation.accepting_states
    assert {word for word in _words(2, 3) if nfa_membership(result, word)} == expected


def test_projection_swaps_consistently_with_relation_inverse() -> None:
    relation = _relation()
    inverse = invert_rational(relation)
    assert project_rational_relation(relation, "input") == project_rational_relation(
        inverse, "output"
    )
    assert project_rational_relation(relation, "output") == project_rational_relation(
        inverse, "input"
    )


def test_projection_manifest_exposes_tape_choice_and_exact_example() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "transducer.relation.projection.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.alphabet_size == 2
    assert result.state_count == 4


def test_rational_alphabet_identity_is_bounded_for_reusable_projection() -> None:
    with pytest.raises(ValidationError):
        RationalTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            input_alphabet_id="a" * 129,
            state_count=1,
            initial_states=(0,),
            accepting_states=(0,),
            edges=(),
        )


def test_large_projection_is_rejected_before_transition_construction(
    monkeypatch,
) -> None:
    relation = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=tuple(
            RationalEdge(
                source=0,
                target=0,
                input_label=(0,) * 512,
                output_label=(0,),
            )
            for _ in range(600)
        ),
    )

    def unexpected_transition(*args, **kwargs):
        raise AssertionError("projection allocated an NFA transition before admission")

    monkeypatch.setattr(
        "jacobian.math.logic.automata.transducers.operations.NFATransition",
        unexpected_transition,
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        project_rational_relation(relation, "input")
    assert error.value.errors()[0]["type"] == (
        "finite_state_transducer.relation_projection_bound_exceeded"
    )
