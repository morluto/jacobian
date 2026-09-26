"""Correctness and admission tests for Weyl action on root-lattice vectors."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    MAX_REFLECTION_REPRESENTABLE,
    WeylVectorActionRequest,
)
from jacobian.math.groups.root_systems.operations import weyl_word_act_on_root_vector

Matrix = tuple[tuple[int, ...], ...]
A2: Matrix = ((2, -1), (-1, 2))
B2: Matrix = ((2, -2), (-1, 2))
G2: Matrix = ((2, -3), (-1, 2))


def _reflect(vector: tuple[int, ...], index: int, matrix: Matrix) -> tuple[int, ...]:
    pairing = sum(
        vector[column] * matrix[index][column] for column in range(len(matrix))
    )
    image = list(vector)
    image[index] -= pairing
    return tuple(image)


def _replay(
    vector: tuple[int, ...], word: tuple[int, ...], matrix: Matrix
) -> tuple[int, ...]:
    for index in word:
        vector = _reflect(vector, index, matrix)
    return vector


@pytest.mark.parametrize(
    ("matrix", "word", "vector"),
    (
        (A2, (), (2, -3)),
        (A2, (0,), (3, -2)),
        (A2, (0, 1, 0), (3, -2)),
        (B2, (0, 1, 0, 1), (1, 2)),
        (G2, (1, 0, 1, 0, 1, 0), (2, -1)),
    ),
)
def test_vector_action_matches_independent_reflection_formula(
    matrix: Matrix, word: tuple[int, ...], vector: tuple[int, ...]
) -> None:
    result = weyl_word_act_on_root_vector(matrix, word, vector)
    assert result.matrix.entries == matrix
    assert result.word == word
    assert result.vector == vector
    assert result.image == _replay(vector, word, matrix)


def test_braid_equivalent_words_have_same_action() -> None:
    first = weyl_word_act_on_root_vector(A2, (0, 1, 0), (5, -2))
    second = weyl_word_act_on_root_vector(A2, (1, 0, 1), (5, -2))
    assert first.image == second.image


def test_empty_word_preserves_maximally_representable_vector() -> None:
    value = MAX_REFLECTION_REPRESENTABLE
    result = weyl_word_act_on_root_vector(((2,),), (), (value,))
    assert result.image == (value,)


def test_image_growth_bound_rejects_before_reflection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.groups.root_systems.operations as operations

    def forbidden(*_args: object, **_kwargs: object) -> list[int]:
        pytest.fail("reflection kernel ran before image-growth admission")

    monkeypatch.setattr(operations, "_apply_reflection", forbidden)
    large = MAX_REFLECTION_REPRESENTABLE // 12 + 1
    with pytest.raises(OperationDomainValidationError) as caught:
        operations.weyl_word_act_on_root_vector(A2, (0,), (large, 0))
    assert caught.value.errors()[0]["type"] == "root_system.weyl_vector_output_bound"


def test_request_and_result_retain_the_same_root_axis() -> None:
    request = WeylVectorActionRequest.model_validate(
        {
            "matrix": {
                "matrix": {
                    "domain": "ZZ",
                    "row_count": 2,
                    "column_count": 2,
                    "entries": [[2, -1], [-1, 2]],
                },
                "simple_root_axis": [0, 1],
            },
            "word": [0, 1],
            "vector": [1, 0],
        }
    )
    assert weyl_word_act_on_root_vector(
        request.matrix, request.word, request.vector
    ).image == _replay(request.vector, request.word, A2)
