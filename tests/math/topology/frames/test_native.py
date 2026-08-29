"""Native finite-frame API and wire/native parity tests."""

from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames import VectorFamily, coherence, frame_potential, gram
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    FiniteFrameRequest,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames._tools import _coherence, _frame_potential, _gram
from jacobian.math.topology.frames.operations import _gram_result, _gram_result_bytes
from jacobian.math.topology.frames.values import MAX_VECTOR_CELLS


def test_native_gram_and_potential_match_wire_adapters() -> None:
    family = VectorFamily(vectors=((1, 1), (1, 0), (0, 1)))

    assert gram(family).gram == ((2, 1, 1), (1, 1, 0), (1, 0, 1))
    assert (
        frame_potential(family).potential
        == _frame_potential(FiniteFrameRequest(vectors=family.vectors)).potential
    )


def test_native_coherence_matches_wire_adapter() -> None:
    family = VectorFamily(vectors=((1, 1), (1, 0), (0, 1)))

    native = coherence(family)
    wire = _coherence(CoherenceRequest(vectors=family.vectors))

    assert native.model_dump() == wire.model_dump()


@pytest.mark.parametrize("operation", [coherence, frame_potential])
def test_native_frame_operations_keep_semantic_admission(
    operation: Callable[[VectorFamily], object],
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        operation(VectorFamily(vectors=((1, 0), (2, 0))))
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


@pytest.mark.parametrize("operation", [coherence, frame_potential])
def test_frame_operations_reject_undercomplete_families_before_rank(
    operation: Callable[[VectorFamily], object],
) -> None:
    family = VectorFamily(
        vectors=tuple(
            tuple(1 if index == coordinate else 0 for coordinate in range(64))
            for index in range(32)
        )
    )

    with pytest.raises(OperationDomainValidationError) as error:
        operation(family)

    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_native_gram_returns_exact_matrix_beyond_mcp_byte_cap() -> None:
    """MCP output size is a transport-only constraint; native gram stays exact."""
    dimension = 512
    basis = tuple(
        tuple(1_000 if row == column else 999 for column in range(dimension))
        for row in range(dimension)
    )
    vectors = basis * 2
    family = VectorFamily(vectors=vectors)
    diagonal = 1_000**2 + (dimension - 1) * 999**2
    off_diagonal = 2 * 1_000 * 999 + (dimension - 2) * 999**2

    assert len(vectors) == MAX_VECTOR_CELLS // dimension
    assert _gram_result_bytes(_gram_result(family)) > 10_485_760
    with pytest.raises(OperationDomainValidationError) as error:
        _gram(VectorFamilyRequest(vectors=vectors))
    assert error.value.errors()[0]["type"] == "frames.result_byte_budget"

    result = gram(family)
    assert result.gram[0][0] == diagonal
    assert result.gram[0][1] == off_diagonal
    assert result.gram[0][dimension] == diagonal
    assert result.gram[1][dimension] == off_diagonal
