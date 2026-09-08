"""Native finite-frame API and wire/native parity tests."""

from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames import VectorFamily, coherence, frame_potential, gram
from jacobian.math.topology.frames._tools import _coherence, _frame_potential, _gram
from jacobian.math.topology.frames.values import MAX_VECTOR_CELLS


def test_native_gram_and_potential_match_wire_adapters() -> None:
    family = VectorFamily(dimension=2, vectors=((1, 1), (1, 0), (0, 1)))

    assert gram(family).gram == ((2, 1, 1), (1, 1, 0), (1, 0, 1))
    assert (
        frame_potential(family).potential
        == _frame_potential(
            VectorFamily(dimension=family.dimension, vectors=family.vectors)
        ).potential
    )


def test_native_coherence_matches_wire_adapter() -> None:
    family = VectorFamily(dimension=2, vectors=((1, 1), (1, 0), (0, 1)))

    native = coherence(family)
    wire = _coherence(VectorFamily(dimension=family.dimension, vectors=family.vectors))

    assert native.model_dump() == wire.model_dump()


@pytest.mark.parametrize("operation", [coherence, frame_potential])
def test_native_frame_operations_keep_semantic_admission(
    operation: Callable[[VectorFamily], object],
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        operation(VectorFamily(dimension=2, vectors=((1, 0), (2, 0))))
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


@pytest.mark.parametrize("operation", [coherence, frame_potential])
def test_frame_operations_reject_undercomplete_families_before_rank(
    operation: Callable[[VectorFamily], object],
) -> None:
    family = VectorFamily(
        dimension=64,
        vectors=tuple(
            tuple(1 if index == coordinate else 0 for coordinate in range(64))
            for index in range(32)
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        operation(family)

    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_native_and_catalog_dense_gram_and_potential_are_exact() -> None:
    family = VectorFamily(
        dimension=2,
        vectors=((1_000, 999), (999, 1_000)) * 2,
    )
    result = gram(family)
    diagonal, off_diagonal = 1_998_001, 1_998_000
    assert result.gram == (
        (diagonal, off_diagonal, diagonal, off_diagonal),
        (off_diagonal, diagonal, off_diagonal, diagonal),
        (diagonal, off_diagonal, diagonal, off_diagonal),
        (off_diagonal, diagonal, off_diagonal, diagonal),
    )
    assert (
        _gram(VectorFamily(dimension=family.dimension, vectors=family.vectors))
        == result
    )
    assert frame_potential(family).potential == 8 * (diagonal**2 + off_diagonal**2)


@pytest.mark.scale
def test_native_and_catalog_gram_return_the_same_large_exact_matrix() -> None:
    dimension = 512
    basis = tuple(
        tuple(1_000 if row == column else 999 for column in range(dimension))
        for row in range(dimension)
    )
    vectors = basis * 2
    family = VectorFamily(dimension=len(vectors[0]), vectors=vectors)
    diagonal = 1_000**2 + (dimension - 1) * 999**2
    off_diagonal = 2 * 1_000 * 999 + (dimension - 2) * 999**2

    assert len(vectors) == MAX_VECTOR_CELLS // dimension
    result = gram(family)
    catalog_result = _gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    assert catalog_result == result
    assert result.gram[0][0] == diagonal
    assert result.gram[0][1] == off_diagonal
    assert result.gram[0][dimension] == diagonal
    assert result.gram[1][dimension] == off_diagonal


def test_empty_gram_retains_ambient_dimension() -> None:
    from jacobian.math.topology.frames._models import GramResult

    source = VectorFamily(dimension=7, vectors=())
    result = gram(source)
    assert result.dimension == 7
    assert result.gram.row_count == result.gram.column_count == 0
    assert GramResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValueError, match="at least as many"):
        frame_potential(source)


def test_zero_dimensional_frame_potential() -> None:
    result = frame_potential(VectorFamily(dimension=0, vectors=()))
    assert result.dimension == 0
    assert result.potential == 0


def test_zero_dimensional_vectors_have_zero_gram_but_no_coherence() -> None:
    from jacobian.math.topology.frames.operations import coherence, verify_gram

    source = VectorFamily(dimension=0, vectors=((), ()))
    result = gram(source)
    assert result.gram.entries == ((0, 0), (0, 0))
    assert verify_gram(result)
    assert frame_potential(source).potential == 0
    with pytest.raises(ValueError, match="nonzero"):
        coherence(source)


def test_empty_zero_dimensional_coherence_has_no_maximizer() -> None:
    from jacobian.math.topology.frames.operations import coherence

    result = coherence(VectorFamily(dimension=0, vectors=()))
    assert result.coherence_squared.num == 0
    assert result.maximizing_pair is None
