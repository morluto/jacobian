"""Highest-coroot profiles checked by independent Cartan-matrix arithmetic."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.root_systems import HighestCorootsResult, highest_coroots
from jacobian.math.groups.root_systems._models import CartanMatrix, CartanMatrixRequest
from jacobian.math.groups.root_systems._tools import TOOLS


def _independent_positive_roots(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    rank = len(matrix)
    roots = {tuple(int(i == j) for j in range(rank)) for i in range(rank)}
    pending = list(roots)
    while pending:
        root = pending.pop()
        for index in range(rank):
            reflected = list(root)
            reflected[index] -= sum(root[j] * matrix[index][j] for j in range(rank))
            image = tuple(reflected)
            if any(value < 0 for value in image) or not any(image):
                continue
            assert max(image) <= 6
            if image not in roots:
                roots.add(image)
                pending.append(image)
    return tuple(sorted(roots))


def _matrix_coroots(
    matrix: tuple[tuple[int, ...], ...],
    symmetrizer: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    bilinear = tuple(
        tuple(Fraction(symmetrizer[i] * matrix[i][j]) for j in range(len(matrix)))
        for i in range(len(matrix))
    )
    result = []
    for root in _independent_positive_roots(matrix):
        squared_length = sum(
            root[i] * bilinear[i][j] * root[j]
            for i in range(len(matrix))
            for j in range(len(matrix))
        )
        coroot = tuple(
            Fraction(2 * symmetrizer[i] * root[i], 1) / squared_length
            for i in range(len(matrix))
        )
        assert all(value.denominator == 1 for value in coroot)
        result.append((root, tuple(int(value) for value in coroot)))
    return tuple(result)


@pytest.mark.parametrize(
    (
        "matrix",
        "symmetrizer",
        "factors",
        "expected_comarks",
        "expected_preimage",
        "expected_highest_coroots",
    ),
    [
        (
            ((2, -3), (-1, 2)),
            (1, 3),
            ((0, 1),),
            ((2, 3),),
            ((2, 1),),
            ((2, 3),),
        ),
        (
            ((2, 0, 0), (0, 2, -2), (0, -1, 2)),
            (1, 1, 2),
            ((0,), (1, 2)),
            ((1,), (1, 2)),
            ((1, 0, 0), (0, 1, 1)),
            ((1, 0, 0), (0, 1, 2)),
        ),
    ],
)
def test_highest_coroots_match_independent_matrix_oracle(
    matrix: tuple[tuple[int, ...], ...],
    symmetrizer: tuple[int, ...],
    factors: tuple[tuple[int, ...], ...],
    expected_comarks: tuple[tuple[int, ...], ...],
    expected_preimage: tuple[tuple[int, ...], ...],
    expected_highest_coroots: tuple[tuple[int, ...], ...],
) -> None:
    oracle_pairs = _matrix_coroots(matrix, symmetrizer)
    result = highest_coroots(CartanMatrix.model_validate(matrix))
    assert (
        tuple(component.simple_root_indices for component in result.components)
        == factors
    )
    assert (
        tuple(component.comarks for component in result.components) == expected_comarks
    )
    assert (
        tuple(
            component.coroot_preimage_root_coefficients
            for component in result.components
        )
        == expected_preimage
    )
    assert (
        tuple(component.highest_coroot_coefficients for component in result.components)
        == expected_highest_coroots
    )

    source_pairs = result.positive_coroots.positive_root_coroot_pairs
    assert (
        tuple(
            (pair.root_coefficients, pair.coroot_coefficients) for pair in source_pairs
        )
        == oracle_pairs
    )
    for component in result.components:
        selected = source_pairs[component.highest_positive_root_coroot_pair_index]
        assert selected.root_coefficients == component.coroot_preimage_root_coefficients
        assert selected.coroot_coefficients == component.highest_coroot_coefficients
    assert HighestCorootsResult.model_validate_json(result.model_dump_json()) == result


def test_highest_coroot_preflight_precedes_root_and_coroot_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.groups.root_systems._highest_coroot_operations as operations

    monkeypatch.setattr(operations, "MAX_HIGHEST_COROOT_WORK", 0)
    monkeypatch.setattr(
        operations,
        "_cartan_datum_from_admitted",
        lambda _cartan: pytest.fail("root/coroot values expanded before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError):
        operations.highest_coroots(CartanMatrix.model_validate(((2, -1), (-1, 2))))


def test_highest_coroot_manifest_example_runs() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.highest_coroots.compute"
    )
    request = CartanMatrixRequest.model_validate_json(
        json.dumps(operation.examples[0].input), strict=True
    )
    result = operation.run(request)
    assert result.components[0].comarks == (2, 3)
