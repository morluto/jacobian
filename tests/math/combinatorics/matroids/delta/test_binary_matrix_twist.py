"""Binary principal-minor families and exact delta-matroid twists."""

from __future__ import annotations

import itertools
import json
from math import prod

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixRequest,
    BinaryMatrixResult,
    BinaryMatrixTwistRequest,
    BinarySymmetricMatrix,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import binary_matrix_twist


def _determinant_over_gf2_by_permutations(
    matrix: tuple[tuple[int, ...], ...], indices: tuple[int, ...]
) -> int:
    """Independent Leibniz oracle; signs disappear in characteristic two."""

    return (
        sum(
            prod(
                matrix[indices[row]][indices[permutation[row]]]
                for row in range(len(indices))
            )
            for permutation in itertools.permutations(range(len(indices)))
        )
        % 2
    )


def _symmetric_binary_matrix(
    size: int, upper_triangle_mask: int
) -> tuple[tuple[int, ...], ...]:
    entries = [[0] * size for _ in range(size)]
    bit = 0
    for row in range(size):
        for column in range(row, size):
            value = (upper_triangle_mask >> bit) & 1
            bit += 1
            entries[row][column] = value
            entries[column][row] = value
    return tuple(tuple(row) for row in entries)


def _feasible_masks_by_independent_determinants(
    matrix: tuple[tuple[int, ...], ...], size: int
) -> set[int]:
    feasible: set[int] = set()
    for mask in range(1 << size):
        indices = tuple(index for index in range(size) if mask >> index & 1)
        if _determinant_over_gf2_by_permutations(matrix, indices):
            feasible.add(mask)
    return feasible


def test_every_small_symmetric_matrix_and_twist_matches_independent_oracle() -> None:
    for size in range(4):
        matrix_count = 1 << (size * (size + 1) // 2)
        for matrix_mask in range(matrix_count):
            entries = _symmetric_binary_matrix(size, matrix_mask)
            ground = tuple(f"e{index}" for index in range(size))
            feasible = _feasible_masks_by_independent_determinants(entries, size)
            for twist_mask in range(1 << size):
                subset = tuple(
                    index for index in range(size) if twist_mask >> index & 1
                )
                result = binary_matrix_twist(
                    BinarySymmetricMatrix(ground=ground, entries=entries), subset
                )
                expected = {mask ^ twist_mask for mask in feasible}
                actual = {
                    sum(1 << index for index in row)
                    for row in result.delta_matroid.feasible
                }

                assert result.matrix.entries == entries
                assert result.matrix.ground == ground
                assert result.twist == subset
                assert actual == expected


def test_eight_axis_identity_matrix_reaches_the_admitted_output_boundary() -> None:
    size = 8
    matrix = BinarySymmetricMatrix(
        ground=tuple(f"e{index}" for index in range(size)),
        entries=tuple(
            tuple(int(row == column) for column in range(size)) for row in range(size)
        ),
    )
    result = binary_matrix_twist(matrix, tuple(range(size)))

    assert len(result.delta_matroid.feasible) == 256
    assert sum(map(len, result.delta_matroid.feasible)) == 1_024
    assert result.twist == tuple(range(size))
    assert result.delta_matroid.feasible == tuple(
        sorted(
            tuple(index for index in range(size) if mask >> index & 1)
            for mask in range(1 << size)
        )
    )


def test_matrix_axes_are_bounded_by_the_schema_and_during_json_parsing() -> None:
    schema = BinarySymmetricMatrix.model_json_schema()
    assert schema["properties"]["entries"]["maxItems"] == 8
    assert schema["properties"]["entries"]["items"]["maxItems"] == 8
    for request_type in (BinaryMatrixRequest, BinaryMatrixTwistRequest):
        request_schema = request_type.model_json_schema()
        matrix_schema = request_schema["$defs"]["BinarySymmetricMatrix"]
        assert matrix_schema["properties"]["entries"]["maxItems"] == 8
        assert matrix_schema["properties"]["entries"]["items"]["maxItems"] == 8

    ground = [f"e{index}" for index in range(8)]
    matrix = [[int(row == column) for column in range(8)] for row in range(8)]
    assert BinarySymmetricMatrix.model_validate_json(
        json.dumps({"ground": ground, "entries": matrix})
    ).entries == tuple(tuple(row) for row in matrix)
    assert BinaryMatrixTwistRequest.model_validate_json(
        json.dumps({"matrix": {"ground": ground, "entries": matrix}})
    ).matrix.entries == tuple(tuple(row) for row in matrix)

    with pytest.raises(ValidationError, match="at most 8 items"):
        BinaryMatrixTwistRequest.model_validate_json(
            json.dumps(
                {
                    "matrix": {
                        "ground": ground,
                        "entries": [*matrix, matrix[0]],
                    }
                }
            )
        )
    with pytest.raises(ValidationError, match="at most 8 items"):
        BinaryMatrixTwistRequest.model_validate_json(
            json.dumps(
                {
                    "matrix": {
                        "ground": ground,
                        "entries": [*matrix[:-1], [*matrix[-1], 0]],
                    }
                }
            )
        )


def test_zero_matrix_twist_example_composes_through_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("delta_matroid.binary.from_matrix_twist.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )

    decoded = BinaryMatrixResult.model_validate_json(json.dumps(result.output))
    assert decoded.matrix.entries == ((0, 0), (0, 0))
    assert decoded.twist == (0, 1)
    assert decoded.delta_matroid.feasible == ((0, 1),)


def test_request_schema_publishes_exact_preflight_envelopes() -> None:
    assert BinaryMatrixTwistRequest.model_json_schema()["admission_limits"] == {
        "max_ground_elements": 8,
        "max_principal_minor_elimination_work": 250_000,
        "max_feasible_rows": 256,
        "max_feasible_set_memberships": 1_024,
        "max_twist_transport_work_units": 20_736,
        "max_output_cells": 1_352,
        "max_output_ground_label_utf8_bytes": 4_096,
    }


@pytest.mark.parametrize("entry", [True, "1", 1.0])
def test_native_twist_rejects_forged_coerced_matrix_entries(entry: object) -> None:
    matrix = BinarySymmetricMatrix.model_construct(
        ground=("e0",), entries=((entry,),)
    )
    with pytest.raises(Exception, match="canonical symmetric binary matrix"):
        binary_matrix_twist(matrix)


def test_result_twist_rejects_coerced_non_integer_indices() -> None:
    payload = {
        "matrix": {"ground": ["e0", "e1"], "entries": [[0, 0], [0, 0]]},
        "delta_matroid": {"ground": ["e0", "e1"], "feasible": [[0, 1]]},
        "twist": [True],
    }
    with pytest.raises(ValidationError):
        BinaryMatrixResult.model_validate_json(json.dumps(payload))
