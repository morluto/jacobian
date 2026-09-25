"""Exact fixed-input output fibers of finite rational relations."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
    operations,
    rational_relation_outputs_for_input,
)
from jacobian.math.logic.automata.transducers._models import (
    RationalRelationFiberRequest,
)
from jacobian.math.logic.automata.transducers._tools import TOOLS
from jacobian.math.logic.languages.regular.operations import nfa_membership
from jacobian.math.logic.languages.regular.values import NFA


def _finite_relation() -> RationalTransducer:
    return RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="letters",
        output_alphabet_id="symbols",
        input_alphabet=FiniteAlphabet(symbols=("a", "b")),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        state_count=5,
        initial_states=(0, 3),
        accepting_states=(2, 4),
        edges=(
            RationalEdge(source=0, target=1, input_label=(0,), output_label=()),
            RationalEdge(source=1, target=2, input_label=(1,), output_label=(1, 0)),
            RationalEdge(source=0, target=2, input_label=(0, 1), output_label=(0,)),
            RationalEdge(source=3, target=4, input_label=(0, 1), output_label=()),
        ),
    )


def _direct_fiber(
    relation: RationalTransducer, fixed_input: tuple[int, ...]
) -> set[tuple[int, ...]]:
    """Enumerate complete accepting paths in this acyclic fixture independently."""

    outgoing: dict[int, list[RationalEdge]] = {}
    for edge in relation.edges:
        outgoing.setdefault(edge.source, []).append(edge)
    outputs: set[tuple[int, ...]] = set()

    def visit(state: int, consumed: tuple[int, ...], emitted: tuple[int, ...]) -> None:
        if consumed == fixed_input and state in relation.accepting_states:
            outputs.add(emitted)
        for edge in outgoing.get(state, ()):
            next_input = consumed + edge.input_label
            if fixed_input[: len(next_input)] == next_input:
                visit(edge.target, next_input, emitted + edge.output_label)

    for initial in relation.initial_states:
        visit(initial, (), ())
    return outputs


def _words(alphabet_size: int, max_length: int) -> set[tuple[int, ...]]:
    return {
        word
        for length in range(max_length + 1)
        for word in product(range(alphabet_size), repeat=length)
    }


@pytest.mark.parametrize("fixed_input", [(), (0,), (1,), (0, 1), (1, 0)])
def test_fixed_input_fiber_matches_direct_path_enumeration(
    fixed_input: tuple[int, ...],
) -> None:
    relation = _finite_relation()
    result = rational_relation_outputs_for_input(relation, fixed_input)

    assert isinstance(result, NFA)
    assert result.alphabet == relation.output_alphabet
    assert result.alphabet_id == relation.output_alphabet_id
    expected = _direct_fiber(relation, fixed_input)
    if fixed_input == (0, 1):
        assert expected == {(), (0,), (1, 0)}
    assert {
        word for word in _words(result.alphabet_size, 3) if nfa_membership(result, word)
    } == expected
    decoded = NFA.model_validate_json(result.model_dump_json(), strict=True)
    assert {
        word
        for word in _words(decoded.alphabet_size, 3)
        if nfa_membership(decoded, word)
    } == expected


def test_epsilon_output_cycle_represents_an_infinite_fiber() -> None:
    relation = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=2,
        input_alphabet=FiniteAlphabet(symbols=("a",)),
        output_alphabet=FiniteAlphabet(symbols=("x", "y")),
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=(RationalEdge(source=0, target=0, input_label=(), output_label=(1,)),),
    )
    result = rational_relation_outputs_for_input(relation, ())

    assert isinstance(result, NFA)
    assert any(
        edge.source == edge.target and edge.symbol == 1 for edge in result.transitions
    )
    for length in range(12):
        assert nfa_membership(result, (1,) * length)
    assert not nfa_membership(result, (0,))


def test_empty_input_and_empty_output_pair_is_the_epsilon_word() -> None:
    relation = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet=FiniteAlphabet(symbols=("a",)),
        output_alphabet=FiniteAlphabet(symbols=("x",)),
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=(),
    )
    result = rational_relation_outputs_for_input(relation, ())
    assert nfa_membership(result, ())
    assert not nfa_membership(result, (0,))


def test_input_word_is_bound_to_relation_input_alphabet() -> None:
    relation = _finite_relation()
    with pytest.raises(OperationDomainValidationError, match="input_word"):
        rational_relation_outputs_for_input(relation, (2,))
    with pytest.raises(OperationDomainValidationError, match="input_word"):
        rational_relation_outputs_for_input(relation, (True,))  # type: ignore[arg-type]


def test_output_parent_is_required_for_regular_language_composition() -> None:
    relation = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=(),
    )
    with pytest.raises(OperationDomainValidationError, match="output alphabet"):
        rational_relation_outputs_for_input(relation, ())


def test_work_admission_precedes_input_pattern_matching(monkeypatch) -> None:
    relation = _finite_relation()

    def forbidden(*_args):
        raise AssertionError("pattern matching began before admission")

    monkeypatch.setattr(operations, "MAX_RATIONAL_FIBER_WORK", 0)
    monkeypatch.setattr(operations, "_matching_positions", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        rational_relation_outputs_for_input(relation, (0, 1))


def test_output_growth_admission_precedes_product_construction(monkeypatch) -> None:
    relation = _finite_relation()

    def forbidden(*_args):
        raise AssertionError("product construction began before output admission")

    monkeypatch.setattr(operations, "MAX_NFA_TRANSITIONS", 0)
    monkeypatch.setattr(operations, "_build_rational_fiber", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        rational_relation_outputs_for_input(relation, (0, 1))


def test_catalog_operation_returns_the_reusable_nfa_value() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "transducer.relation.outputs_for_input_automaton.compute"
    )
    result = tool.run(
        RationalRelationFiberRequest(transducer=_finite_relation(), input_word=(0, 1))
    )
    assert isinstance(result, NFA)
