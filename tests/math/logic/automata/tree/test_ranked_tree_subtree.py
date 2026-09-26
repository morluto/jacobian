"""Exact subtree extraction against an independent recursive oracle."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.tree import (
    RankedTree,
    ranked_tree_subtree,
)
from jacobian.math.logic.automata.tree._tools import TOOLS


def _recursive_at(tree: RankedTree, position: tuple[int, ...]) -> RankedTree:
    if not position:
        return RankedTree(
            symbol=tree.symbol,
            children=tuple(_recursive_at(child, ()) for child in tree.children),
        )
    head, *tail = position
    return _recursive_at(tree.children[head], tuple(tail))


def test_subtree_selection_retains_source_and_matches_recursive_oracle() -> None:
    source = RankedTree(
        symbol=0,
        children=(
            RankedTree(symbol=1, children=(RankedTree(symbol=2),)),
            RankedTree(
                symbol=3,
                children=(RankedTree(symbol=4), RankedTree(symbol=5)),
            ),
        ),
    )
    for position in ((), (0,), (0, 0), (1,), (1, 1)):
        result = ranked_tree_subtree(source, position)
        assert result.tree is source
        assert result.position == position
        assert result.subtree == _recursive_at(source, position)


def test_root_position_returns_exact_source_tree() -> None:
    source = RankedTree(symbol=0, children=(RankedTree(symbol=1),))
    result = ranked_tree_subtree(source, ())
    assert result.subtree == source
    assert result.tree == source


def test_missing_child_index_is_rejected_as_invalid_address() -> None:
    source = RankedTree(symbol=0, children=(RankedTree(symbol=1),))
    with pytest.raises(OperationDomainValidationError, match="does not identify"):
        ranked_tree_subtree(source, (1,))


def test_subtree_catalog_example_is_published_and_round_trips() -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "ranked_tree.subtree.compute"
    )
    result = operation.run(
        operation.request_type.model_validate(operation.examples[0].input)
    )
    assert result.position == (0,)
    assert result.subtree == RankedTree(symbol=1)
    assert type(result).model_validate_json(result.model_dump_json()) == result
