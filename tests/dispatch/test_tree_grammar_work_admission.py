"""Automaton productivity and grammar expansion share one operation budget."""

import json
from typing import NoReturn

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.automata.tree import operations as tree_operations
from jacobian.math.logic.automata.tree.operations import (
    MAX_TREE_GRAMMAR_CONVERSION_WORK,
    tree_automaton_to_regular_tree_grammar,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    TreeAutomatonTransition,
)

_CONVERT = "tree_automaton.to_regular_tree_grammar.compute"


def _wide_chain(row_count: int) -> BottomUpTreeAutomaton:
    rows = [
        TreeAutomatonTransition(symbol=1, child_states=(index + 1,), target_state=index)
        for index in range(63)
    ]
    rows.extend(
        TreeAutomatonTransition(
            symbol=2, child_states=(index // 64, index % 64) + (0,) * 14, target_state=0
        )
        for index in range(row_count - 64)
    )
    rows.append(TreeAutomatonTransition(symbol=0, child_states=(), target_state=63))
    return BottomUpTreeAutomaton(
        state_count=64, arity=(0, 1, 16), transitions=tuple(rows), final_states=(0,)
    )


@pytest.mark.parametrize("row_count", [2302, 4096])
def test_combined_work_refuses_before_production_construction(
    row_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    automaton = _wide_chain(row_count)
    saturation = 65 * (row_count + 1) * 17
    conversion = (
        17 * (2 + 63 * 3 + (row_count - 64) * 18)
        + 4 * row_count * (row_count + 1).bit_length() * 16
        + 8 * 64
        + 4 * 3
    )
    assert max(saturation, conversion) <= MAX_TREE_GRAMMAR_CONVERSION_WORK
    assert saturation + conversion > MAX_TREE_GRAMMAR_CONVERSION_WORK

    def production_must_not_be_constructed(**_values: object) -> NoReturn:
        raise AssertionError("production expansion ran before combined work admission")

    monkeypatch.setattr(
        tree_operations, "RegularTreeProduction", production_must_not_be_constructed
    )
    with pytest.raises(OperationResourceAdmissionError) as native:
        tree_automaton_to_regular_tree_grammar(automaton)
    with pytest.raises(OperationResourceAdmissionError) as public:
        invoke_operation(
            _CONVERT, {"automaton": automaton.model_dump(mode="json")}, Catalog.open()
        )
    for error in (native.value, public.value):
        assert error.errors()[0]["type"] == "tree_automata.grammar_conversion_bound"


def test_combined_work_accepts_the_near_boundary_grammar_and_consumer() -> None:
    # 2,301 rows cost 4,999,171 total units; one more row costs 5,001,350.
    automaton = _wide_chain(2301)
    native = tree_automaton_to_regular_tree_grammar(automaton)
    catalog = Catalog.open()
    result = invoke_operation(
        _CONVERT, {"automaton": automaton.model_dump(mode="json")}, catalog
    )
    grammar_payload = json.loads(result.model_dump_json())["output"]["grammar"]

    assert len(native.productions) == len(automaton.transitions) == 2301
    assert native.start_nonterminal == 0
    assert grammar_payload == native.model_dump(mode="json")
    consumed = invoke_operation(
        "regular_tree_grammar.to_automaton.compute",
        {"grammar": grammar_payload},
        catalog,
    )
    revived = BottomUpTreeAutomaton.model_validate_json(
        json.dumps(consumed.output["automaton"])
    )
    assert revived.state_count == automaton.state_count
    assert revived.arity == automaton.arity
    assert revived.final_states == automaton.final_states
    assert set(revived.transitions) == set(automaton.transitions)
