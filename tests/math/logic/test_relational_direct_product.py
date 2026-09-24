"""Exact finite relational direct products and their coordinate maps."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    RelationalProductRequest,
    check_homomorphism,
    direct_product_structure,
)


def _structure(size: int, rows: list[list[int]]) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=size,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(tuple(tuple(row) for row in rows),),
    )


def test_direct_product_uses_lexicographic_pairs_and_coordinatewise_relation() -> None:
    left = _structure(2, [[0, 1], [1, 0]])
    right = _structure(3, [[0, 2], [2, 1]])

    result = direct_product_structure(left, right)

    assert result.product.carrier_size == 6
    assert result.product.relation_tables == (((0, 5), (2, 4), (3, 2), (5, 1)),)
    assert result.left_projection == (0, 0, 0, 1, 1, 1)
    assert result.right_projection == (0, 1, 2, 0, 1, 2)
    left_projection = check_homomorphism(result.product, left, result.left_projection)
    right_projection = check_homomorphism(
        result.product, right, result.right_projection
    )
    assert left_projection.status is HomomorphismStatus.HOMOMORPHISM
    assert right_projection.status is HomomorphismStatus.HOMOMORPHISM
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result


def test_product_preserves_nullary_truth_values_and_empty_carriers() -> None:
    true = FiniteRelationalStructure(
        carrier_size=1,
        signature=(FiniteRelationSymbol(symbol_id="P", arity=0),),
        relation_tables=(((),),),
    )
    false = FiniteRelationalStructure.model_validate(
        {
            "carrier_size": 1,
            "signature": [{"symbol_id": "P", "arity": 0}],
            "relation_tables": [[]],
        }
    )
    assert direct_product_structure(true, true).product.relation_tables == (((),),)
    assert direct_product_structure(true, false).product.relation_tables == ((),)

    empty = FiniteRelationalStructure(carrier_size=0)
    empty_product = direct_product_structure(empty, empty)
    assert empty_product.product.carrier_size == 0
    assert empty_product.left_projection == empty_product.right_projection == ()

    empty_true = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="P", arity=0),),
        relation_tables=(((),),),
    )
    assert direct_product_structure(empty_true, empty_true).product.relation_tables == (
        ((),),
    )
    assert direct_product_structure(false, empty_true).product.relation_tables == ((),)


def test_product_rejects_signature_mismatch_and_overlarge_cartesian_table() -> None:
    left = _structure(2, [[0, 1]])
    mismatched = FiniteRelationalStructure(carrier_size=2)
    with pytest.raises(
        OperationDomainValidationError, match="identical ranked signatures"
    ):
        direct_product_structure(left, mismatched)

    # Both operands fit their own 4,096-row contract and their Cartesian
    # carrier has exactly 64 labels. Their product relation has 4,096^2 rows,
    # so admission rejects before making those rows.
    four_place = FiniteRelationSymbol(symbol_id="T", arity=4)
    rows = tuple(product(range(8), repeat=4))
    a = FiniteRelationalStructure(
        carrier_size=8, signature=(four_place,), relation_tables=(rows,)
    )
    b = a
    with pytest.raises(OperationResourceAdmissionError, match="product relation"):
        direct_product_structure(a, b)


def test_product_operation_is_discoverable_and_example_executes() -> None:
    operation_id = "relational.structure.direct_product.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    result = tool.run(RelationalProductRequest.model_validate(tool.examples[0].input))
    assert result.product.relation_tables == (((0, 3),),)
