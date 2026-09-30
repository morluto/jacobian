"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/combinatorics/matroids/test_matroids.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from jacobian.math.combinatorics.matroids import (
    LinearMatroid,
)
from jacobian.math.combinatorics.matroids._models import (
    MatroidClosureRequest,
    MatroidClosureResult,
)
from jacobian.math.combinatorics.matroids.operations import closure_result
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def compute_closure(request: MatroidClosureRequest) -> MatroidClosureResult:
    return closure_result(request.matroid, request.subset)


class _PrimeFieldMatrixPayload(TypedDict):
    prime: int
    entries: list[list[int]]
    columns: int


class _MatroidPayload(TypedDict):
    matrix: _PrimeFieldMatrixPayload


class _ClosureRequestPayload(TypedDict):
    matroid: _MatroidPayload
    subset: list[int]


def _matroid(prime: int, rows: Sequence[Sequence[int]], columns: int) -> LinearMatroid:

    entries = tuple(tuple(row[j] for j in range(columns)) for row in rows)
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)
    )


def _identity_rows(r: int) -> list[tuple[int, ...]]:
    return [tuple(1 if i == j else 0 for j in range(r)) for i in range(r)]


class TestCatalogAdmission:
    def test_duplicate_rank_operation_not_registered(self) -> None:
        """prime_field.matrix.rank covers full-ground-set rank; no duplicate."""
        from jacobian.catalog.builtins import BUILTIN_TOOLS

        ids = [t.operation_id for t in BUILTIN_TOOLS]
        assert "matroid.rank.compute" not in ids
        assert "matroid.closure.compute" in ids
