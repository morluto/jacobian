"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/affine_semigroups/test_group_lattice.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

import json

from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form as sympy_column_hnf

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.affine_semigroups.group_lattice_models import (
    AffineGroupLatticeRequest,
)
from jacobian.math.affine_semigroups.semigroup import AffineConfiguration


def _configuration(entries: tuple[tuple[int, ...], ...]) -> AffineConfiguration:
    return AffineConfiguration(
        row_labels=tuple(f"r{i}" for i in range(len(entries))),
        generator_labels=tuple(f"g{i}" for i in range(len(entries[0]))),
        entries=entries,
    )


def _sympy_image_hnf(configuration: AffineConfiguration) -> tuple[tuple[int, ...], ...]:
    matrix = Matrix([[int(value) for value in row] for row in configuration.entries])
    column_basis = sympy_column_hnf(matrix)
    return tuple(
        tuple(int(column_basis[row, column]) for row in range(column_basis.rows))
        for column in range(column_basis.cols)
    )


def test_group_lattice_catalog_example_executes() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_semigroup.group_lattice.compute"
    )
    request = AffineGroupLatticeRequest.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.lattice.basis.entries == ((2, 0), (0, 2))
