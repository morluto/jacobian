"""Catalog-owned contract checks for the Coxeter polynomial operation."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.groups.root_systems import operations


def test_catalog_example_returns_the_canonical_integer_polynomial() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("root_system.coxeter_polynomial.compute")
    assert operation is not None

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    assert result.output["coefficients"] == ["1", "1", "1"]


def test_coxeter_polynomial_admission_reports_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations, "MAX_COXETER_POLYNOMIAL_WORK", 1)
    with pytest.raises(OperationResourceAdmissionError):
        operations.coxeter_polynomial(((2, -1), (-1, 2)))
