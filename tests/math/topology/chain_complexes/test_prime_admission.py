"""Typed semantic primality never hides operational failures or changes fields."""

from collections.abc import Callable
from typing import NoReturn

import pytest

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes import _filtered_operations as filtered
from jacobian.math.topology.chain_complexes import operations
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    HomologyGroupValue,
)


def _source(prime: int) -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=prime,
        degree_min=0,
        degree_max=0,
        basis_sizes=(1,),
        differential_matrices=(),
    )


@pytest.mark.parametrize("prime", [2, 3, 251])
def test_prime_fields_keep_exact_homology_and_source(prime: int) -> None:
    source = _source(prime)
    result = operations.homology_groups(source)
    assert result.complex == source
    assert len(result.homology_groups) == 1
    group = result.homology_groups[0]
    assert isinstance(group, HomologyGroupValue)
    assert (group.cycle_rank, group.boundary_rank, group.betti_number) == (1, 0, 1)
    assert operations.differential_squares_to_zero(source).is_valid


@pytest.mark.parametrize(
    "error",
    [
        ValueError("unexpected backend failure"),
        OperationBackendError(BackendFailureReason.INVALID_OUTPUT),
        OperationExecutionCancelledError("cancelled"),
        OperationExecutionTimeoutError("timed out"),
        OperationResourceAdmissionError(
            location=("complex",), code="test.resource", message="resource refusal"
        ),
    ],
)
def test_prime_admission_does_not_reclassify_operational_failures(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(operations, "require_prime_field_admission", fail)
    monkeypatch.setattr(filtered, "require_prime_field_admission", fail)
    source = _source(2)
    calls: tuple[Callable[[], object], ...] = (
        lambda: operations.homology_groups(source),
        lambda: operations.differential_squares_to_zero(source),
        lambda: filtered.associated_graded(
            source,
            (FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1,),)),)),),
        ),
    )
    for call in calls:
        with pytest.raises(type(error)) as caught:
            call()
        assert caught.value is error
