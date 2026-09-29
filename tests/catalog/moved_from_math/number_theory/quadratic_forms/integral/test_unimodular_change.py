"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/quadratic_forms/integral/test_unimodular_change.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticForm,
)


def _polynomial_value(form: IntegralQuadraticForm, coordinates: tuple[int, ...]) -> int:
    result = sum(
        coefficient * coordinate**2
        for coefficient, coordinate in zip(
            form.diagonal_coefficients, coordinates, strict=True
        )
    )
    return result + sum(
        term.coefficient * coordinates[term.left] * coordinates[term.right]
        for term in form.cross_terms
    )


def _matrix_vector(matrix: IntegerMatrix, vector: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(
            int(entry) * coordinate
            for entry, coordinate in zip(row, vector, strict=True)
        )
        for row in matrix.entries
    )


def _request(
    form: IntegralQuadraticForm,
    matrix: tuple[tuple[int, ...], ...],
    target_axis: tuple[str, ...],
) -> tuple[IntegralQuadraticForm, IntegerMatrix, tuple[str, ...]]:
    return (
        form,
        IntegerMatrix(
            row_count=len(matrix),
            column_count=len(matrix),
            entries=matrix,
        ),
        target_axis,
    )


def test_advertised_tool_composes_after_json_roundtrip() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "quadratic_form.unimodular_change.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    transported = result.model_validate_json(result.model_dump_json())
    assert transported.target == result.target
    assert transported.inverse == result.inverse
    assert transported.target.diagonal_coefficients == (1, 6)
    assert transported.target.cross_terms[0].coefficient == 5
