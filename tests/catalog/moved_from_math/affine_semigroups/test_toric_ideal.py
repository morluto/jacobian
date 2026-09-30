"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/affine_semigroups/test_toric_ideal.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.affine_semigroups.graver_models import (
    IntegerConfigurationToricIdealRequest,
)
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration
from jacobian.math.polynomials.values import RationalPolynomialIdeal


def _configuration(weights: tuple[int, ...], labels: tuple[str, ...] | None = None):
    names = labels or tuple(f"x{i + 1}" for i in range(len(weights)))
    return AffineConfiguration(
        row_labels=("degree",),
        generator_labels=names,
        entries=(weights,),
    )


def _exponent_coefficients(ideal: RationalPolynomialIdeal):
    return tuple(
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in generator.polynomial.terms
        )
        for generator in ideal.generators
    )


def test_catalog_example_runs_and_json_roundtrips() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "integer_configuration.toric_ideal.compute"
    )
    request = IntegerConfigurationToricIdealRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(request)

    assert isinstance(result, RationalPolynomialIdeal)
    decoded = RationalPolynomialIdeal.model_validate_json(result.model_dump_json())
    assert decoded == result
