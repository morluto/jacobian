"""Native finite-frame API and wire/native parity tests."""

from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames import VectorFamily, coherence, frame_potential, gram
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    FiniteFrameRequest,
)
from jacobian.math.topology.frames._operations import (
    compute_coherence,
    compute_frame_potential,
)


def test_native_gram_and_potential_match_wire_adapters() -> None:
    family = VectorFamily(vectors=((1, 1), (1, 0), (0, 1)))

    assert gram(family).gram == ((2, 1, 1), (1, 1, 0), (1, 0, 1))
    assert (
        frame_potential(family).potential
        == compute_frame_potential(FiniteFrameRequest(vectors=family.vectors)).potential
    )


def test_native_coherence_matches_wire_adapter() -> None:
    family = VectorFamily(vectors=((1, 1), (1, 0), (0, 1)))

    native = coherence(family)
    wire = compute_coherence(CoherenceRequest(vectors=family.vectors))

    assert native.model_dump() == wire.model_dump()


@pytest.mark.parametrize("operation", [coherence, frame_potential])
def test_native_frame_operations_keep_semantic_admission(
    operation: Callable[[VectorFamily], object],
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        operation(VectorFamily(vectors=((1, 0), (2, 0))))
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"
