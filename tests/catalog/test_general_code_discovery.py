"""Catalog coverage for general linear-code operations."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_catalog_examples_use_the_canonical_encoder_carrier() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("code.minimum_distance.compute")
    assert operation is not None

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    assert result.output["minimum_distance"] == 2
    assert result.output["request"]["encoder"]["coordinate_axis"] == ["x0", "x1"]
