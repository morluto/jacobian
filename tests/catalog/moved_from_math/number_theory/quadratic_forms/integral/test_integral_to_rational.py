"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/quadratic_forms/integral/test_integral_to_rational.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalCoordinateVector,
)


def _rational_vector(axis: tuple[str, ...], values: tuple[int, ...]):
    return RationalCoordinateVector(
        axis=axis,
        coordinates=tuple(
            CanonicalRational.from_integer_ratio(value, 1) for value in values
        ),
    )


def test_catalog_operation_is_discoverable_and_its_example_runs() -> None:
    operation_id = "quadratic_form.integral.rational_extension.compute"
    tool = next(item for item in BUILTIN_TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.target.cross_terms[0].coefficient.as_fraction() == 3
