"""Verification failures are not mathematical counterexamples."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraphs import operations
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    EdgeWeight,
    FiniteHypergraph,
    HypergraphIndependenceBudget,
    HypergraphIndependenceResult,
)


def _source() -> FiniteHypergraph:
    return FiniteHypergraph(vertices=("a", "b"), edges=(("e", ("a", "b")),))


@pytest.mark.parametrize(
    "operation",
    [
        "independence_number",
        "minimum_transversal",
        "maximum_edge_matching",
        "maximum_weight_packing",
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
def test_verifier_propagates_computation_failures(
    monkeypatch: pytest.MonkeyPatch, operation: str, error: Exception
) -> None:
    producer = getattr(operations, operation)
    arguments = (_source(),)
    if operation == "maximum_weight_packing":
        claim = producer(
            *arguments,
            (EdgeWeight(edge_id="e", weight=CanonicalRational(num=1, den=1)),),
        )
    else:
        claim = producer(*arguments)
    verifier = getattr(
        operations,
        "verify_weighted_packing"
        if operation == "maximum_weight_packing"
        else "verify_" + operation,
    )
    assert verifier(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(operations, operation, fail)
    with pytest.raises(type(error), match="backend"):
        verifier(claim)


def test_exact_independence_verification_reports_actual_incomplete_search() -> None:
    source = FiniteHypergraph(
        vertices=tuple(str(i) for i in range(5)),
        edges=tuple(
            (str(i), tuple(sorted((str(i), str((i + 1) % 5))))) for i in range(5)
        ),
    )
    exact = operations.independence_number(
        source, HypergraphIndependenceBudget(max_solver_calls=2)
    )
    assert exact.independence_number == 2
    payload = exact.model_dump(mode="json")
    payload["resource_budget"]["max_solver_calls"] = 1
    payload["solver_calls"] = 0
    claim = HypergraphIndependenceResult.model_validate(payload)
    with pytest.raises(OperationResourceAdmissionError, match="could not establish"):
        operations.verify_independence_number(claim)
