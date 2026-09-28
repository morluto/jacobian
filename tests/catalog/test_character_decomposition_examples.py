"""Value-specific assertions for published character-decomposition examples.

These check the example's exact output values, which the catalog-wide
example lane does not assert. They live here rather than under tests/math
because they boot the product catalog and dispatch boundary.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_character_ring_decomposition_catalog_example_executes() -> None:
    catalog = Catalog.open()
    operation_id = "finite_group.class_function.character_ring_decompose.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    assert result.output["ring_element"]["irreducible_multiplicities"] == [
        "0",
        "0",
        "1",
    ]


def test_character_tensor_decomposition_catalog_example_executes() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(
        "finite_group.character.tensor_product.decompose.compute"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["multiplicities"] == [1, 1, 1]
