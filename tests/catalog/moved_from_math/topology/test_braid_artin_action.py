"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_braid_artin_action.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology.edge_paths._models import FiniteGroupWord, WordLetter
from jacobian.math.topology.links import BraidLetter, BraidWord
from jacobian.math.topology.links._extensions_models import BraidWordRequest


def _word(strands: int, *letters: tuple[int, int]) -> BraidWord:
    return BraidWord(
        strand_count=strands,
        letters=tuple(
            BraidLetter(generator=generator, exponent=exponent)
            for generator, exponent in letters
        ),
    )


def _free_word(*letters: tuple[int, int]) -> FiniteGroupWord:
    return FiniteGroupWord(
        letters=tuple(
            WordLetter(generator=generator, exponent=exponent)
            for generator, exponent in letters
        )
    )


def test_action_is_published_as_one_typed_catalog_operation() -> None:
    tools = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
    operation = tools["braid.word.artin_action.compute"]
    result = operation.run(BraidWordRequest(word=_word(2, (1, 1), (1, -1))))
    assert result.generator_images == (_free_word((0, 1)), _free_word((1, 1)))
