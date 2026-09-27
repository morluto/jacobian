"""Published tree-automaton operations and discovery at the catalog boundary."""

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest
from jacobian.math.logic.automata.tree import (
    DeterministicBottomUpTreeAutomaton,
    TreeAutomatonTransition,
)
from jacobian.math.logic.automata.tree._models import TreeAutomatonMinimizeRequest


def test_boolean_product_example_states_the_completeness_precondition() -> None:
    operation = Catalog.open().operation("tree_automaton.boolean_product.compute")
    assert operation is not None
    example = operation.examples[0]
    assert "partial" not in example.name
    assert "complete" in example.description.lower()
    request = operation.request_type.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    assert operation.run(request).product.state_count == 1


def test_catalog_discovery_surfaces_tree_language_products() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(
            need="exact intersection and union product for deterministic ranked tree automata",
            limit=5,
        )
    )
    assert result.matches[0].operation_id == "tree_automaton.boolean_product.compute"


def test_catalog_complement_operation_uses_the_typed_contract() -> None:
    tool = Catalog.open().operation("tree_automaton.complement.compute")
    assert tool is not None
    request = tool.request_type.model_validate(
        {
            "automaton": {
                "state_count": 2,
                "arity": [0],
                "transitions": [{"symbol": 0, "child_states": [], "target_state": 1}],
                "final_states": [1],
            }
        }
    )
    assert tool.run(request).complement.final_states == (1,)


def test_catalog_completion_operation_has_typed_json_contract() -> None:
    operation = Catalog.open().operation(
        "tree_automaton.deterministic.complete.compute"
    )
    assert operation is not None
    request = operation.request_type.model_validate(
        {
            "automaton": {
                "state_count": 1,
                "arity": [0, 1],
                "transitions": [{"symbol": 0, "child_states": [], "target_state": 0}],
                "final_states": [0],
            }
        }
    )
    result = operation.run(request)
    assert result.completed.state_count == 2
    assert result.sink_state == 1


def test_minimize_is_published_in_catalog() -> None:
    tool = Catalog.open().operation("tree_automaton.deterministic.minimize.compute")
    assert tool is not None
    request = TreeAutomatonMinimizeRequest(
        automaton=DeterministicBottomUpTreeAutomaton(
            state_count=1,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            ),
            final_states=(0,),
        )
    )
    assert tool.run(request).minimized.state_count == 1
