"""Tests for linear matroid operations over a prime field."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids import (
    LinearMatroid,
    matroid_closure,
    matroid_rank,
)
from jacobian.math.combinatorics.matroids._models import (
    MatroidClosureRequest,
    MatroidClosureResult,
)
from jacobian.math.combinatorics.matroids.operations import closure_result


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
    from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

    entries = tuple(tuple(row[j] for j in range(columns)) for row in rows)
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)
    )


def _identity_rows(r: int) -> list[tuple[int, ...]]:
    return [tuple(1 if i == j else 0 for j in range(r)) for i in range(r)]


class TestLinearMatroidRepresentation:
    def test_canonical_matrix_representation(self) -> None:
        """The ground set indexes columns of the canonical matrix value."""
        from jacobian.math.matrices.finite_fields.linear_algebra import (
            PrimeFieldMatrix,
        )

        m = _matroid(5, _identity_rows(2), 2)
        assert isinstance(m.matrix, PrimeFieldMatrix)
        assert m.ground_size == 2

    def test_oversized_prime_rejected_before_construction(self) -> None:
        """A ~3,000-digit prime is bounded before primality work."""
        with pytest.raises(ValidationError) as exc_info:
            LinearMatroid.model_validate(
                {"matrix": {"prime": 2**9941 - 1, "entries": [], "columns": 0}}
            )
        assert exc_info.value.errors()[0]["type"] == "matroid.field_prime.bound"

    def test_matroid_accepts_shared_matrix_column_carrier(self) -> None:
        from jacobian.math.matrices.finite_fields.linear_algebra import (
            PrimeFieldMatrix,
        )

        boundary_matrix = PrimeFieldMatrix(
            prime=5,
            entries=((0,) * 256,),
            columns=256,
        )
        assert LinearMatroid(matrix=boundary_matrix).ground_size == 256
        matrix_payload: _PrimeFieldMatrixPayload = {
            "prime": 5,
            "entries": [[0] * 256],
            "columns": 256,
        }
        payload: _ClosureRequestPayload = {
            "matroid": {"matrix": matrix_payload},
            "subset": [],
        }
        assert MatroidClosureRequest.model_validate(payload).matroid.ground_size == 256

    def test_shared_carrier_does_not_widen_matroid_row_envelope(self) -> None:
        """The matrix carrier may scale without widening matroid witness work."""
        from jacobian.math.matrices.finite_fields.linear_algebra import (
            PrimeFieldMatrix,
        )

        matrix = PrimeFieldMatrix(
            prime=2,
            entries=((0,),) * 257,
            columns=1,
        )
        with pytest.raises(ValidationError) as exc_info:
            LinearMatroid(matrix=matrix)
        assert exc_info.value.errors()[0]["type"] == "matroid.representation_rows.bound"

    def test_empty_matroid_admitted(self) -> None:
        """The empty ground set with a preserved row axis is representable."""
        m = LinearMatroid.model_validate(
            {
                "matrix": {
                    "prime": 5,
                    "entries": [[], []],
                    "columns": 0,
                }
            }
        )
        assert m.ground_size == 0
        assert matroid_rank(m) == 0
        closure, rank = matroid_closure(m, [])
        assert closure == () and rank == 0


class TestClosure:
    def test_closure_of_basis_is_everything_in_its_span(self) -> None:
        """U(2,3): the closure of a basis is the whole ground set."""
        m = _matroid(5, _identity_rows(2) and [(1, 0, 1), (0, 1, 1)], 3)
        closure, rank = matroid_closure(m, [0, 1])
        assert rank == 2
        assert closure == (0, 1, 2)

    def test_wire_closure_round_trips(self) -> None:
        m = _matroid(5, [(1, 0, 1), (0, 1, 1)], 3)
        request = MatroidClosureRequest(matroid=m, subset=(0,))
        result = compute_closure(request)
        assert result.closure == (0,)
        assert result.rank == 1
        payload = result.model_dump(mode="json")
        assert type(result).model_validate(payload) == result

    def test_subset_indices_validated(self) -> None:
        m = _matroid(5, [(1, 0), (0, 1)], 2)
        request = MatroidClosureRequest(matroid=m, subset=(2,))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_closure(request)
        assert exc_info.value.errors()[0]["type"] == "matroid.subset.invalid"

    def test_native_closure_validates_indices(self) -> None:
        """The native entry point applies the wire subset admission."""
        m = _matroid(5, [(1, 0), (0, 1)], 2)
        with pytest.raises(ValueError, match=r"0\.\.n-1"):
            matroid_closure(m, [-1])
        with pytest.raises(ValueError, match=r"0\.\.n-1"):
            matroid_closure(m, [2])
        with pytest.raises(ValueError, match="distinct"):
            matroid_closure(m, [0, 0])

    def test_seven_by_105_closure_uses_result_sensitive_rank_work(self) -> None:
        rows = tuple(
            tuple(int(column % 7 == row) for column in range(105)) for row in range(7)
        )
        matroid = _matroid(1_000_003, rows, 105)

        closure, rank = matroid_closure(matroid, (0, 1, 2, 3, 4))

        assert rank == 5
        assert closure == tuple(
            column for column in range(105) if column % 7 in {0, 1, 2, 3, 4}
        )

    def test_dense_large_seed_rejects_rank_work_before_kernel(self) -> None:
        matroid = _matroid(2, _identity_rows(256), 256)

        with pytest.raises(OperationDomainValidationError, match="work bound"):
            matroid_closure(matroid, tuple(range(128)))


class TestCatalogAdmission:
    def test_duplicate_rank_operation_not_registered(self) -> None:
        """prime_field.matrix.rank covers full-ground-set rank; no duplicate."""
        from jacobian.catalog.builtins import BUILTIN_TOOLS

        ids = [t.operation_id for t in BUILTIN_TOOLS]
        assert "matroid.rank.compute" not in ids
        assert "matroid.closure.compute" in ids
