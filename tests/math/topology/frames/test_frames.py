"""Exact frame and vector-family contract tests."""

from fractions import Fraction

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    FiniteFrameRequest,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames._tools import _coherence, _frame_potential, _gram
from jacobian.math.topology.frames.operations import gram
from jacobian.math.topology.frames.values import MAX_DIM, MAX_VECTORS, VectorFamily


def _repeated_standard_basis(
    *, dimension: int, repeats: int
) -> tuple[tuple[int, ...], ...]:
    basis = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    return basis * repeats


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


def test_flint_gram_reconstructs_dot_products_and_quadratic_form() -> None:
    vectors = tuple(
        tuple(((row * 17 + column * 31) % 11) - 5 for column in range(128))
        for row in range(256)
    )
    result = gram(VectorFamily(vectors=vectors))

    assert result.gram[17][203] == sum(
        left * right for left, right in zip(vectors[17], vectors[203], strict=True)
    )
    coefficients = (2, -3, 1)
    indices = (4, 91, 177)
    quadratic = sum(
        coefficients[i] * result.gram[indices[i]][indices[j]] * coefficients[j]
        for i in range(len(indices))
        for j in range(len(indices))
    )
    combined = tuple(
        sum(coefficients[i] * vectors[indices[i]][column] for i in range(len(indices)))
        for column in range(len(vectors[0]))
    )
    assert quadratic == sum(entry * entry for entry in combined) >= 0


def test_result_sensitive_operations_diverge_at_full_carrier_boundary() -> None:
    vectors = _repeated_standard_basis(dimension=MAX_DIM, repeats=2)
    family = VectorFamily(vectors=vectors)

    assert len(vectors) == MAX_VECTORS
    gram_result = gram(family)
    potential = _frame_potential(FiniteFrameRequest(vectors=vectors))
    coherence = _coherence(CoherenceRequest(vectors=vectors))
    assert len(gram_result.gram) == MAX_VECTORS
    assert gram_result.gram[0][MAX_DIM] == 1
    assert potential.potential == str(2 * MAX_VECTORS)
    assert coherence.coherence_squared.as_integer_ratio() == (1, 1)
    assert coherence.maximizing_pair == (MAX_DIM - 1, MAX_VECTORS - 1)
    assert (
        len(encode_strict_json(gram_result.model_dump(mode="json")))
        <= CanonicalLimits().max_output_bytes
    )
    assert (
        len(encode_strict_json(potential.model_dump(mode="json")))
        <= CanonicalLimits().max_output_bytes
    )


def test_dense_high_height_gram_is_rejected_before_backend_expansion() -> None:
    dimension = MAX_DIM
    basis = tuple(
        tuple(1_000 if row == column else 999 for column in range(dimension))
        for row in range(dimension)
    )
    vectors = basis * 2
    family = VectorFamily(vectors=vectors)

    with pytest.raises(OperationDomainValidationError) as error:
        gram(family)
    assert error.value.errors()[0]["type"] == "frames.result_byte_budget"

    potential = _frame_potential(FiniteFrameRequest(vectors=vectors))
    diagonal = 1_000**2 + (dimension - 1) * 999**2
    off_diagonal = 2 * 1_000 * 999 + (dimension - 2) * 999**2
    expected = 4 * dimension * (diagonal**2 + (dimension - 1) * off_diagonal**2)
    assert potential.potential == str(expected)


def test_flint_rank_rejects_nonspanning_family_above_previous_boundary() -> None:
    vector = (1,) * 32
    request = FiniteFrameRequest(vectors=(vector,) * 64)

    with pytest.raises(OperationDomainValidationError) as error:
        _frame_potential(request)
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_coherence_maximizer_matches_complete_exact_profile() -> None:
    vectors = _repeated_standard_basis(dimension=32, repeats=2)
    result = _coherence(CoherenceRequest(vectors=vectors))
    gram_result = _gram(VectorFamilyRequest(vectors=vectors)).gram
    candidates = (
        (
            Fraction(
                gram_result[left][right] ** 2,
                gram_result[left][left] * gram_result[right][right],
            ),
            (left, right),
        )
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )

    maximum, pair = max(candidates)
    assert result.coherence_squared.as_integer_ratio() == (
        maximum.numerator,
        maximum.denominator,
    )
    assert result.maximizing_pair == pair
