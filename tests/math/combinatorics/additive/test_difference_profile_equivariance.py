"""Justified affine/permutation laws retain differences and their source labels."""

from collections import defaultdict

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.additive._models import (
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
)
from jacobian.math.combinatorics.additive.operations import (
    ordered_difference_profile,
    verify_ordered_difference_profile,
)

_Vector = tuple[int, int]
_Matrix = tuple[_Vector, _Vector]
_Pair = tuple[int, int]
_Table = dict[tuple[int, ...], frozenset[_Pair]]
_SOURCE: tuple[_Vector, ...] = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1))


def _linear(matrix: _Matrix, vector: tuple[int, ...]) -> _Vector:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    )


def _oracle(source: tuple[_Vector, ...]) -> _Table:
    groups: dict[tuple[int, ...], set[_Pair]] = defaultdict(set)
    for left, first in enumerate(source):
        for right, second in enumerate(source):
            if left != right:
                groups[(first[0] - second[0], first[1] - second[1])].add((left, right))
    return {difference: frozenset(pairs) for difference, pairs in groups.items()}


def _table(result: OrderedDifferenceProfileResult) -> _Table:
    assert all(entry.multiplicity == len(entry.pairs) for entry in result.entries)
    assert (
        sum(entry.multiplicity for entry in result.entries)
        == result.total_ordered_pairs
    )
    return {
        entry.difference.as_int_tuple(): frozenset(
            (pair.left_index, pair.right_index) for pair in entry.pairs
        )
        for entry in result.entries
    }


def _profile(source: tuple[_Vector, ...]) -> OrderedDifferenceProfileResult:
    request = OrderedDifferenceProfileRequest.model_validate(
        {"vectors": {"vectors": [{"coordinates": vector} for vector in source]}}
    )
    result = ordered_difference_profile(request.vectors)
    assert tuple(vector.as_int_tuple() for vector in result.vectors.vectors) == source
    assert _table(result) == _oracle(source)
    decoded = OrderedDifferenceProfileResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert verify_ordered_difference_profile(decoded)
    return result


@pytest.mark.parametrize(
    ("matrix", "translation"),
    [
        (((1, 0), (0, 1)), (7, -3)),
        (((1, 1), (0, 1)), (-2, 4)),
        (((2, 0), (0, 2)), (0, 0)),
        (((0, -1), (1, 0)), (3, 5)),
    ],
    ids=["translation", "shear", "scaling", "rotation"],
)
def test_invertible_affine_maps_transport_differences_and_preserve_multiplicity(
    matrix: _Matrix, translation: _Vector
) -> None:
    assert matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] != 0
    base = _profile(_SOURCE)
    transformed_source = tuple(
        (image[0] + translation[0], image[1] + translation[1])
        for vector in _SOURCE
        for image in (_linear(matrix, vector),)
    )
    transformed = _profile(transformed_source)
    expected = {
        _linear(matrix, difference): pairs for difference, pairs in _table(base).items()
    }
    assert _table(transformed) == expected
    assert transformed.support_size == base.support_size
    assert transformed.max_multiplicity == base.max_multiplicity
    assert transformed.total_ordered_pairs == len(_SOURCE) * (len(_SOURCE) - 1)
    if matrix == ((2, 0), (0, 2)):
        # Scaling transports differences; treating them as unchanged is false.
        assert _table(transformed) != _table(base)


def test_source_permutation_transports_indices_and_rejects_stale_labels() -> None:
    new_to_old = (2, 4, 0, 3, 1)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    base = _profile(_SOURCE)
    permuted = _profile(tuple(_SOURCE[old] for old in new_to_old))
    expected = {
        difference: frozenset(
            (old_to_new[left], old_to_new[right]) for left, right in pairs
        )
        for difference, pairs in _table(base).items()
    }
    assert _table(permuted) == expected
    assert _table(permuted) != _table(base)
    assert sorted(entry.multiplicity for entry in permuted.entries) == sorted(
        entry.multiplicity for entry in base.entries
    )
    # Correct multiplicities alone do not certify source-indexed witnesses.
    stale = base.model_copy(update={"vectors": permuted.vectors})
    assert not verify_ordered_difference_profile(stale)


def test_noninjective_projection_leaves_the_distinct_vector_set_domain() -> None:
    projected = tuple((vector[0], 0) for vector in _SOURCE)
    assert len(set(projected)) < len(_SOURCE)
    with pytest.raises(ValidationError) as raised:
        _profile(projected)
    assert raised.value.errors()[0]["type"] == (
        "additive_combinatorics.require_uniform_distinct_bounded"
    )
