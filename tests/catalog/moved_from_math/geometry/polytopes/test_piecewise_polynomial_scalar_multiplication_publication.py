"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from
``tests/math/geometry/polytopes/test_piecewise_polynomial_scalar_multiplication.py``.
The preamble below is carried over so the moved test resolves every name it uses.
"""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes.complexes._models import (
    PiecewisePolynomialResult,
)


def test_catalog_example_executes_through_the_public_typed_operation():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.scalar_multiply.compute")
    assert operation is not None and operation.examples

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert (
        validated.pieces[0].polynomial.polynomial.terms[0].coefficient.as_fraction()
        == 2
    )
