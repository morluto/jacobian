"""Exact frame and vector-family contract tests."""

import pytest
from pydantic import ValidationError

from jacobian.math.frames._models import (
    CoherenceRequest,
    FiniteFrameRequest,
    VectorFamilyRequest,
)
from jacobian.math.frames._operations import (
    compute_coherence,
    compute_frame_potential,
    compute_gram,
)


def test_gram_accepts_nonspanning_vector_family() -> None:
    assert compute_gram(VectorFamilyRequest(vectors=[[1, 0], [2, 0]])).gram == (
        (1, 2),
        (2, 4),
    )


def test_frame_requires_full_ambient_span() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteFrameRequest(vectors=[[1, 0], [2, 0]])
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_coherence_rejects_zero_vector() -> None:
    with pytest.raises(ValidationError) as error:
        CoherenceRequest(vectors=[[0, 0], [1, 0], [0, 1]])
    assert error.value.errors()[0]["type"] == "frames.zero_vector"


def test_coherence_is_exact_and_carries_canonical_maximizer() -> None:
    result = compute_coherence(CoherenceRequest(vectors=[[1, 1], [1, 0], [0, 1]]))
    assert result.coherence_squared.as_integer_ratio() == (1, 2)
    assert result.maximizing_pair == (0, 2)


def test_potential_remains_exact_above_json_safe_integer() -> None:
    repeated = [1000] * 16
    final = [1000] * 15 + [999]
    vectors = (
        [repeated] * 5 + [final] + [[int(i == j) for j in range(16)] for i in range(16)]
    )
    result = compute_frame_potential(FiniteFrameRequest(vectors=vectors))
    expected = sum(
        sum(a * b for a, b in zip(left, right, strict=True)) ** 2
        for left in result.vectors
        for right in result.vectors
    )
    assert result.potential == str(expected)
