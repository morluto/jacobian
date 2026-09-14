"""Catalog invocation contract for rational discrete antiderivatives."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._discrete_antiderivative_tools import (
    RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION,
)


def test_catalog_invocation_returns_the_declared_typed_result() -> None:
    catalog = Catalog.open()
    invocation = RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples[0]
    result = invoke_operation(
        RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.operation_id,
        invocation.input,
        catalog,
    )
    assert set(result.output) == {
        "source",
        "variable",
        "antiderivative",
        "reconstructed_difference",
    }
