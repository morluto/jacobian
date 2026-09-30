"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/test_character_tensor_product.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from itertools import permutations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterTensorProductRequest,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_table():
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _element(table, coordinates):
    return CharacterRingElement(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )


def _direct_s3_tensor_square_multiplicities() -> tuple[int, int, int]:
    # The standard module is the quotient of C^3 by the invariant line.
    # Thus its trace at a permutation is fixed_points(g)-1. Tensor traces
    # square pointwise; exact averaging over the six represented elements
    # gives multiplicities against the independently described irreducibles.
    totals = [0, 0, 0]
    for p in permutations(range(3)):
        fixed = sum(p[i] == i for i in range(3))
        standard = fixed - 1
        sign = (
            -1 if sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1
        )
        for index, irreducible in enumerate((1, sign, standard)):
            totals[index] += standard * standard * irreducible
    return tuple(value // 6 for value in totals)


def test_operation_is_published_and_has_exact_result_type() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "character.tensor_product.compute"
    )
    assert tool.request_type is CharacterTensorProductRequest
    assert tool.result_type is CharacterRingElement
    assert "cyclic" in tool.description
