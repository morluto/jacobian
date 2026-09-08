"""Additive profile verification distinguishes failure from disagreement."""

import pytest

from jacobian.math.combinatorics.additive import operations
from jacobian.math.combinatorics.additive._models import FiniteIntegerSet


@pytest.mark.parametrize(
    "operation",
    [
        "representation_profile",
        "additive_energy",
        "sumset_cardinality",
        "direct_sum_predicate",
    ],
)
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("backend"),
        ValueError("backend"),
        TypeError("backend"),
        MemoryError("backend"),
        TimeoutError("backend"),
    ],
)
def test_verifier_preserves_execution_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str, error: Exception
) -> None:
    source = FiniteIntegerSet(elements=(0, 1))
    args = (
        (3, source, source) if operation == "direct_sum_predicate" else (source, source)
    )
    claim = getattr(operations, operation)(*args)
    verifier = getattr(operations, "verify_" + operation)
    assert verifier(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(operations, operation, fail)
    with pytest.raises(type(error), match="backend"):
        verifier(claim)


@pytest.mark.parametrize(
    "operation",
    [
        "representation_profile",
        "additive_energy",
        "sumset_cardinality",
        "direct_sum_predicate",
    ],
)
def test_verifier_preserves_cartesian_resource_refusal(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.combinatorics.additive import _models

    source = FiniteIntegerSet(elements=(0, 1))
    args = (
        (3, source, source) if operation == "direct_sum_predicate" else (source, source)
    )
    claim = getattr(operations, operation)(*args)
    monkeypatch.setattr(_models, "_MAX_CARTESIAN_PAIR_COUNT", 0)
    with pytest.raises(OperationResourceAdmissionError):
        getattr(operations, "verify_" + operation)(claim)


def test_ordered_difference_verifier_preserves_incomplete_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.combinatorics.additive._models import (
        IntegerVector,
        IntegerVectorSet,
    )

    source = IntegerVectorSet(
        vectors=(IntegerVector(coordinates=(0,)), IntegerVector(coordinates=(1,)))
    )
    claim = operations.ordered_difference_profile(source)
    assert operations.verify_ordered_difference_profile(claim)
    monkeypatch.setattr(operations, "MAX_ORDERED_DIFFERENCE_COORDINATE_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError):
        operations.verify_ordered_difference_profile(claim)
