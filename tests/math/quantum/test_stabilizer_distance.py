"""General stabilizer distance against independent Pauli enumeration."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    stabilizer_exact_distance,
)


def _span(rows: tuple[tuple[int, ...], ...]) -> set[tuple[int, ...]]:
    width = len(rows[0]) if rows else 0
    return {
        tuple(
            sum(c * row[j] for c, row in zip(coefficients, rows, strict=True)) % 2
            for j in range(width)
        )
        for coefficients in product((0, 1), repeat=len(rows))
    }


def _oracle(rows: tuple[tuple[int, ...], ...], n: int):
    stabilizers = _span(rows)
    logicals = []
    for local in product((0, 1, 2, 3), repeat=n):
        flat = tuple(value & 1 for value in local) + tuple(
            (value >> 1) & 1 for value in local
        )
        if not any(local) or any(
            sum(flat[j] * row[n + j] + flat[n + j] * row[j] for j in range(n)) % 2
            for row in rows
        ):
            continue
        if flat not in stabilizers:
            logicals.append((sum(value != 0 for value in local), local))
    minimum = min(weight for weight, _ in logicals)
    return minimum, {local for weight, local in logicals if weight == minimum}


@pytest.mark.parametrize(
    ("n", "checks"),
    (
        (2, ((1, 0, 1, 0),)),
        (2, ((1, 0, 1, 1),)),
        (3, ((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 0))),
        (3, ((1, 1, 0, 1, 1, 0), (0, 0, 1, 0, 0, 1))),
    ),
)
def test_general_distance_matches_independent_pauli_oracle(n, checks) -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
    rows = tuple(
        PhaseFreeQubitPauli(register=register, x_bits=row[:n], z_bits=row[n:])
        for row in checks
    )
    value = CheckSpaceValue(register=register, basis=rows)
    distance, oracle_minima = _oracle(checks, n)
    result = stabilizer_exact_distance(value)
    assert result.check_space == value
    assert result.logical_qubits == n - len(checks)
    assert result.distance == distance
    assert result.representative is not None
    representative = result.representative
    encoded = tuple(
        x + 2 * z
        for x, z in zip(representative.x_bits, representative.z_bits, strict=True)
    )
    assert sum(v != 0 for v in encoded) == distance
    assert encoded in oracle_minima


def test_stabilizer_distance_k_zero_has_no_distance() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    check = PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,))
    result = stabilizer_exact_distance(
        CheckSpaceValue(register=register, basis=(check,))
    )
    assert result.logical_qubits == 0
    assert result.distance is result.representative is None


def test_stabilizer_distance_rejects_search_beyond_ten_qubits() -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(11)))
    value = CheckSpaceValue(register=register, basis=())
    with pytest.raises(OperationResourceAdmissionError, match="complete mixed-Pauli"):
        stabilizer_exact_distance(value)


def test_catalog_example_runs_and_returns_source_bound_result() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.distance.compute")
    assert operation is not None
    output = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    ).output
    assert output["logical_qubits"] == 1
    assert output["distance"] == 1
    assert (
        output["representative"]["qubit_register"]
        == output["check_space"]["qubit_register"]
    )
