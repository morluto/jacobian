"""Dispatch-boundary tests for linear matroid operations."""

from __future__ import annotations

import pytest
from jsonschema.validators import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids._models import MAX_GROUND_SIZE


def test_closure_schema_and_dispatch_share_subset_contract() -> None:
    """The discoverable rule is the same one that ``math.run`` applies."""

    catalog = Catalog.open()
    descriptor = catalog.inspect("matroid.closure.compute")
    assert descriptor is not None
    subset_schema = descriptor.input_schema["properties"]["subset"]
    assert subset_schema["maxItems"] == MAX_GROUND_SIZE
    assert subset_schema["uniqueItems"] is True
    assert subset_schema["description"] == (
        "Distinct ground-set indices. Every index must lie in "
        "0..matroid.matrix.columns-1; at most "
        f"{MAX_GROUND_SIZE} indices are admitted."
    )
    validator = Draft202012Validator(descriptor.input_schema)

    operation = catalog.operation("matroid.closure.compute")
    assert operation is not None
    advertised = operation.examples[0].input
    assert not list(validator.iter_errors(advertised))
    result = invoke_operation("matroid.closure.compute", advertised, catalog)
    assert result.output["closure"] == [0, 1, 2]
    assert result.output["rank"] == 2

    duplicate_subset = {**advertised, "subset": [0, 0]}
    assert list(validator.iter_errors(duplicate_subset))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        invoke_operation("matroid.closure.compute", duplicate_subset, catalog)
    assert exc_info.value.errors()[0]["type"] == "matroid.subset.invalid"
