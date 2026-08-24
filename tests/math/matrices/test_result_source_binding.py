"""Regression tests binding rational RREF, rank, and nullspace to their source."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices._operation_models import (
    MatrixRankRequest,
    MatrixRankResult,
    NullspaceResult,
    RationalMatrixRequest,
    RrefResult,
)
from jacobian.math.matrices._operations import (
    compute_nullspace,
    compute_rank,
    compute_rref,
)
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION, RationalMatrix


def _matrix(rows: list[list[str]]) -> RationalMatrix:
    return RationalMatrix.model_validate(
        {
            "matrix_schema_version": "1",
            "domain": "QQ",
            "entries": [
                [
                    {
                        "num": value.split("/")[0],
                        "den": value.split("/")[1] if "/" in value else "1",
                    }
                    for value in row
                ]
                for row in rows
            ],
        }
    )


def test_producer_results_replay_across_shapes() -> None:
    """Zero, rectangular, rank-deficient, and full-rank sources stay bound."""

    shapes = (
        _matrix([["0", "0"], ["0", "0"]]),
        _matrix([["1", "2", "3"], ["4", "5", "6"]]),
        _matrix([["1", "2"], ["2", "4"], ["3", "6"]]),
        _matrix([["1/2", "0"], ["0", "1/3"]]),
    )
    for matrix in shapes:
        request = RationalMatrixRequest(matrix=matrix)
        rref = compute_rref(request)
        assert rref.matrix == matrix
        assert (
            rref.reduced_matrix.entries
            == compute_rref(RationalMatrixRequest(matrix=matrix)).reduced_matrix.entries
        )

        rank_request = MatrixRankRequest(matrix=matrix)
        rank = compute_rank(rank_request)
        assert rank.matrix == matrix
        assert rank.rank == len(rank.pivot_columns)

        nullspace = compute_nullspace(request)
        assert nullspace.matrix == matrix
        assert nullspace.rank + nullspace.nullity == nullspace.ambient_dimension
        assert len(nullspace.basis_vectors) == nullspace.nullity


@pytest.mark.parametrize(
    "rows",
    (
        [["1", "2", "3"], ["4", "5", "6"]],
        [["0", "0", "0"], ["1", "0", "0"]],
        [["1/2", "1/3"], ["1/5", "1/7"], ["1", "1"]],
    ),
)
def test_serialized_results_round_trip(rows: list[list[str]]) -> None:
    matrix = _matrix(rows)
    dumped_rref = compute_rref(RationalMatrixRequest(matrix=matrix)).model_dump()
    assert (
        RrefResult.model_validate(dumped_rref).reduced_matrix
        == compute_rref(RationalMatrixRequest(matrix=matrix)).reduced_matrix
    )
    dumped_rank = compute_rank(MatrixRankRequest(matrix=matrix)).model_dump()
    assert (
        MatrixRankResult.model_validate(dumped_rank).rank
        == compute_rank(MatrixRankRequest(matrix=matrix)).rank
    )
    dumped_null = compute_nullspace(RationalMatrixRequest(matrix=matrix)).model_dump()
    assert (
        NullspaceResult.model_validate(dumped_null).nullity
        == compute_nullspace(RationalMatrixRequest(matrix=matrix)).nullity
    )


def _mutable(dumped: dict) -> dict:
    """JSON round-trip so nested tuple payloads become mutable lists."""
    import json

    return json.loads(json.dumps(dumped))


def test_rref_result_rejects_mutations() -> None:
    matrix = _matrix([["1", "2"], ["2", "4"]])
    result = compute_rref(RationalMatrixRequest(matrix=matrix))
    dumped = result.model_dump()

    foreign_source = copy.deepcopy(_mutable(dumped))
    foreign_source["matrix"]["entries"] = [
        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
    ]
    with pytest.raises(ValidationError):
        RrefResult.model_validate(foreign_source)

    forged_form = copy.deepcopy(_mutable(dumped))
    forged_form["reduced_matrix"]["entries"][0][1] = {"num": "9", "den": "1"}
    with pytest.raises(ValidationError, match="unique RREF"):
        RrefResult.model_validate(forged_form)

    forged_pivots = copy.deepcopy(_mutable(dumped))
    forged_pivots["pivot_columns"] = [1]
    forged_pivots["free_columns"] = [0]
    with pytest.raises(ValidationError, match="unique RREF"):
        RrefResult.model_validate(forged_pivots)

    broken_partition = copy.deepcopy(_mutable(dumped))
    broken_partition["free_columns"] = []
    with pytest.raises(ValidationError, match="partition"):
        RrefResult.model_validate(broken_partition)


def test_rank_result_rejects_mutations() -> None:
    matrix = _matrix([["1", "2"], ["2", "4"]])
    result = compute_rank(MatrixRankRequest(matrix=matrix))
    dumped = result.model_dump()

    forged_rank = copy.deepcopy(_mutable(dumped))
    forged_rank["rank"] = 32
    with pytest.raises(ValidationError, match="pivot column count"):
        MatrixRankResult.model_validate(forged_rank)

    forged_pivots = copy.deepcopy(_mutable(dumped))
    forged_pivots["pivot_columns"] = [1]
    with pytest.raises(ValidationError, match="replay against the source"):
        MatrixRankResult.model_validate(forged_pivots)

    foreign_source = copy.deepcopy(_mutable(dumped))
    foreign_source["matrix"]["entries"] = [
        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
    ]
    with pytest.raises(ValidationError, match="replay against the source"):
        MatrixRankResult.model_validate(foreign_source)


def test_nullspace_result_rejects_mutations() -> None:
    matrix = _matrix([["1", "2"], ["2", "4"]])
    result = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    dumped = result.model_dump()
    assert result.nullity == 1
    assert result.basis_vectors[0][-2:] == (
        __import__("jacobian")._exact.CanonicalRational(num="-2", den="1"),
        __import__("jacobian")._exact.CanonicalRational(num="1", den="1"),
    )

    outside_vector = copy.deepcopy(_mutable(dumped))
    outside_vector["basis_vectors"] = [
        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}]
    ]
    with pytest.raises(ValidationError, match="A v = 0"):
        NullspaceResult.model_validate(outside_vector)

    non_fundamental = copy.deepcopy(_mutable(dumped))
    scaled = copy.deepcopy(non_fundamental["basis_vectors"][0])
    scaled[1] = {"num": "2", "den": "1"}
    scaled[0] = {"num": "-4", "den": "1"}
    non_fundamental["basis_vectors"] = [scaled]
    with pytest.raises(ValidationError, match="fundamental basis vector"):
        NullspaceResult.model_validate(non_fundamental)

    forged_dimension = copy.deepcopy(_mutable(dumped))
    forged_dimension["ambient_dimension"] = 3
    with pytest.raises(ValidationError, match="ambient dimension must equal"):
        NullspaceResult.model_validate(forged_dimension)

    forged_rank = copy.deepcopy(_mutable(dumped))
    forged_rank["rank"] = 2
    forged_rank["nullity"] = 0
    forged_rank["basis_vectors"] = []
    forged_rank["free_columns"] = []
    with pytest.raises(ValidationError, match="exact rank of the source"):
        NullspaceResult.model_validate(forged_rank)


def test_producer_to_serialized_interoperability() -> None:
    """Serialized RREF agrees with independently produced rank and nullspace."""

    matrix = _matrix([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
    rref = RrefResult.model_validate(
        compute_rref(RationalMatrixRequest(matrix=matrix)).model_dump()
    )
    rank = compute_rank(MatrixRankRequest(matrix=matrix))
    nullspace = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    assert rref.rank == rank.rank == nullspace.rank == 2
    assert list(rref.pivot_columns) == list(rank.pivot_columns)
    assert list(rref.free_columns) == list(nullspace.free_columns)


def _identity_entries(size: int) -> tuple[tuple[CanonicalRational, ...], ...]:
    one = CanonicalRational(num="1", den="1")
    zero = CanonicalRational(num="0", den="1")
    return tuple(
        tuple(one if index == column else zero for column in range(size))
        for index in range(size)
    )


def test_request_admission_rejects_matrices_above_the_computation_dimension() -> None:
    oversized = RationalMatrix(entries=_identity_entries(MAX_MATRIX_DIMENSION + 1))
    with pytest.raises(ValidationError, match="rows and columns"):
        RationalMatrixRequest(matrix=oversized)
    with pytest.raises(ValidationError, match="rows and columns"):
        MatrixRankRequest(matrix=oversized)


def test_request_admission_rejects_one_oversized_axis() -> None:
    tall = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num=str(column + 1), den="1") for column in range(2)
            )
            for _ in range(MAX_MATRIX_DIMENSION + 1)
        )
    )
    with pytest.raises(ValidationError, match="rows and columns"):
        RationalMatrixRequest(matrix=tall)
    with pytest.raises(ValidationError, match="rows and columns"):
        MatrixRankRequest(matrix=tall)


def test_request_admission_keeps_the_boundary_computation_dimension() -> None:
    boundary = RationalMatrix(entries=_identity_entries(MAX_MATRIX_DIMENSION))
    request = MatrixRankRequest(matrix=boundary)
    rank = compute_rank(request)
    assert rank.rank == MAX_MATRIX_DIMENSION
    assert len(rank.pivot_columns) == MAX_MATRIX_DIMENSION
    assert rank.matrix == boundary
