"""Independent exact tests for root-first ranked-tree position enumeration."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree import (
    RankedTree,
    ranked_tree_positions,
)
from jacobian.math.logic.automata.tree._tools import TOOLS


def _oracle_positions(
    tree: RankedTree, prefix: tuple[int, ...] = ()
) -> tuple[tuple[int, ...], ...]:
    rows = [prefix]
    for index, child in enumerate(tree.children):
        rows.extend(_oracle_positions(child, (*prefix, index)))
    return tuple(rows)


def test_positions_match_independent_preorder_oracle_and_round_trip() -> None:
    tree = RankedTree(
        symbol=0,
        children=(
            RankedTree(symbol=1),
            RankedTree(symbol=2, children=(RankedTree(symbol=3),)),
        ),
    )

    result = ranked_tree_positions(tree)

    assert result.positions == _oracle_positions(tree)
    assert result.positions == ((), (0,), (1,), (1, 0))
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result


def test_positions_for_degenerate_leaf_and_wide_node() -> None:
    leaf = RankedTree(symbol=0)
    assert ranked_tree_positions(leaf).positions == ((),)

    wide = RankedTree(symbol=0, children=tuple(RankedTree(symbol=i) for i in range(16)))
    result = ranked_tree_positions(wide)
    assert result.positions == ((), *((index,) for index in range(16)))


def test_positions_preflight_rejects_tree_over_depth_bound() -> None:
    tree = RankedTree(symbol=0)
    for _ in range(128):
        tree = RankedTree(symbol=0, children=(tree,))

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ranked_tree_positions(tree)

    assert exc_info.value.errors()[0]["type"] == "tree_automata.positions.depth_bound"


def test_positions_reject_validation_bypassed_tree_shape() -> None:
    tree = RankedTree.model_construct(symbol=True, children=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ranked_tree_positions(tree)

    assert exc_info.value.errors()[0]["type"] == "tree_automata.positions.tree_shape"


def test_positions_catalog_example_is_registered() -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "ranked_tree.positions.compute"
    )
    result = operation.run(
        operation.request_type.model_validate(operation.examples[0].input)
    )
    assert result.positions == ((), (0,), (1,), (1, 0))
