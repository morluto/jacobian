"""Operational non-completion does not refute an authored mathematical claim."""

from collections.abc import Callable
from typing import Any, NoReturn

import pytest

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.math.finite_fields import (
    analyze_collisions,
    analyze_permutation,
    element,
    fiber_partition,
    finite_field,
    finite_map_table,
    finite_polynomial,
    finite_polynomial_map,
    operations,
    paley_tournament,
)


@pytest.mark.parametrize(
    ("producer", "verifier", "boundary"),
    [
        (fiber_partition, operations.verify_fiber_partition, "_authenticate_map_table"),
        (analyze_collisions, operations.verify_collisions, "_authenticate_map_table"),
        (analyze_permutation, operations.verify_permutation, "_authenticate_map_table"),
    ],
)
@pytest.mark.parametrize("error", [RuntimeError, OperationExecutionTimeoutError])
def test_finite_map_verifiers_propagate_noncompletion(
    monkeypatch: pytest.MonkeyPatch,
    producer: Callable[..., Any],
    verifier: Callable[..., bool],
    boundary: str,
    error: type[Exception],
) -> None:
    field = finite_field(3, (0, 1))
    polynomial = finite_polynomial(field, (element(field, (0,)), element(field, (1,))))
    table = finite_map_table(finite_polynomial_map(polynomial))
    claim = producer(table)
    claim = type(claim).model_validate_json(claim.model_dump_json())
    assert verifier(claim)

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise error("injected operational failure")

    monkeypatch.setattr(operations, boundary, fail)
    with pytest.raises(error, match="injected operational failure"):
        verifier(claim)


def test_paley_verifier_propagates_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = paley_tournament(finite_field(3, (0, 1)))

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("injected operational failure")

    monkeypatch.setattr(operations, "paley_tournament", fail)
    with pytest.raises(RuntimeError, match="injected operational failure"):
        operations.verify_paley_tournament(claim)


def test_fiber_verifier_does_not_refute_an_unadmitted_table() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.finite_fields import FiniteMapTable

    field = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(field, (1,) + (0,) * 7)
    table = finite_map_table(finite_polynomial_map(finite_polynomial(field, (one,))))
    claim = fiber_partition(table)
    candidate_map = finite_polynomial_map(finite_polynomial(field, (one,) * 512))
    candidate = FiniteMapTable(map=candidate_map, entries=table.entries)
    unadmitted_claim = claim.model_copy(update={"table": candidate})
    decoded = type(claim).model_validate_json(unadmitted_claim.model_dump_json())
    with pytest.raises(OperationResourceAdmissionError):
        operations.verify_fiber_partition(decoded)
