"""Publication checks for the elliptic-curve subgroup and zeta operations.

These open the catalog, so they live in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_generated_subgroup_membership_is_publicly_discoverable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(
        "elliptic_curve.finite_field.point.membership_in_generated_subgroup.decide"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["belongs"] is True
    assert len(result.output["generators"]) == 1


def test_full_zeta_function_is_publicly_discoverable() -> None:
    tool = Catalog.open().operation("elliptic_curve.finite_field.zeta.compute")
    assert tool is not None
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    assert result.output["cardinality"] == 9
    assert result.output["trace"] == -3
    assert result.output["zeta_function"]["variables"] == ["T"]
