"""Exact frame and vector-family contract tests."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    FiniteFrameRequest,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames._tools import _coherence, _frame_potential, _gram


def test_gram_accepts_nonspanning_vector_family() -> None:
    assert _gram(
        VectorFamilyRequest.model_validate({"vectors": [[1, 0], [2, 0]]})
    ).gram == (
        (1, 2),
        (2, 4),
    )


def test_frame_requires_full_ambient_span() -> None:
    request = FiniteFrameRequest.model_validate({"vectors": [[1, 0], [2, 0]]})
    with pytest.raises(OperationDomainValidationError) as error:
        _frame_potential(request)
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_coherence_rejects_zero_vector() -> None:
    request = CoherenceRequest.model_validate({"vectors": [[0, 0], [1, 0], [0, 1]]})
    with pytest.raises(OperationDomainValidationError) as error:
        _coherence(request)
    assert error.value.errors()[0]["type"] == "frames.zero_vector"


def test_coherence_is_exact_and_carries_canonical_maximizer() -> None:
    result = _coherence(
        CoherenceRequest.model_validate({"vectors": [[1, 1], [1, 0], [0, 1]]})
    )
    assert result.coherence_squared.as_integer_ratio() == (1, 2)
    assert result.maximizing_pair == (0, 2)


def test_potential_remains_exact_above_json_safe_integer() -> None:
    repeated = [1000] * 16
    final = [1000] * 15 + [999]
    vectors = (
        [repeated] * 5 + [final] + [[int(i == j) for j in range(16)] for i in range(16)]
    )
    result = _frame_potential(FiniteFrameRequest.model_validate({"vectors": vectors}))
    expected = sum(
        sum(a * b for a, b in zip(left, right, strict=True)) ** 2
        for left in result.vectors
        for right in result.vectors
    )
    assert result.potential == str(expected)
