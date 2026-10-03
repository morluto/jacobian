"""Advertised delta-matroid examples through the published catalog."""

from __future__ import annotations

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids.delta.extra import BinaryMatrixResult


def test_zero_matrix_twist_example_composes_through_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.binary.from_matrix_twist.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    decoded = BinaryMatrixResult.model_validate_json(json.dumps(result.output))
    assert decoded.matrix.entries == ((0, 0), (0, 0))
    assert decoded.twist == (0, 1)
    assert decoded.delta_matroid.feasible == ((0, 1),)


def test_published_operation_example_runs_through_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.distance_profile.compute")
    assert operation is not None

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    assert result.output["distance_by_mask"] == [0, 0, 0, 0]
    assert result.output["nearest_feasible_count_by_mask"] == [1, 1, 1, 1]
    assert result.output["distance_histogram"] == [4, 0, 0]
