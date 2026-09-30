"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/extensions/test_affine_realization.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic.extensions._models import (
    FiniteLatticeExtension,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
)

_S3_TABLE = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 4, 5, 2, 3),
    (2, 5, 0, 4, 3, 1),
    (3, 4, 5, 0, 1, 2),
    (4, 3, 1, 2, 5, 0),
    (5, 2, 3, 1, 0, 4),
)
_S3_ACTION = (
    ((1, 0), (0, 1)),
    ((-1, 1), (0, 1)),
    ((1, 0), (1, -1)),
    ((0, -1), (-1, 0)),
    ((0, -1), (1, -1)),
    ((-1, 1), (-1, 0)),
)


def _s3_extension() -> FiniteLatticeExtension:
    return FiniteLatticeExtension(
        multiplication_table=_S3_TABLE,
        action_matrices=_S3_ACTION,
        factor_set=_S3_COCYCLE,
    )


def _translation_extension() -> FiniteLatticeExtension:
    return FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(((1, 0), (0, 1)),),
        factor_set=(((0, 0),),),
    )


def _shift(result, element: int) -> tuple[Fraction, ...]:
    return tuple(
        value.as_fraction() for value in result.section_maps[element].section_shift
    )


def _matvec(
    matrix: tuple[tuple[int, ...], ...], vector: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(len(vector))),
            Fraction(),
        )
        for row in range(len(matrix))
    )


def test_affine_realization_round_trips_and_manifest_examples_execute() -> None:
    result = affine_section_realization(_s3_extension())
    assert type(result).model_validate_json(result.model_dump_json()) == result

    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.extension.affine_realization.compute"
    )
    assert len(tool.examples) == 2
    for example in tool.examples:
        request = tool.request_type.model_validate_json(
            json.dumps(example.input), strict=True
        )
        realized = tool.run(request)
        assert realized.source == request


_S3_COCYCLE = (
    ((0, 0),) * 6,
    ((0, 0), (0, 0), (-1, 0), (0, 0), (-1, 0), (0, 0)),
    ((0, 0), (1, 0), (2, 1), (1, 0), (1, 0), (1, 0)),
    ((0, 0), (0, 0), (0, -1), (0, 0), (0, 0), (-1, 0)),
    ((0, 0), (0, 0), (0, 1), (-1, 0), (0, 0), (0, 0)),
    ((0, 0), (-1, 0), (-1, -1), (0, 0), (0, 0), (0, 0)),
)
