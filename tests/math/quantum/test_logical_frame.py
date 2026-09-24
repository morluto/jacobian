from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.quantum import (
    CheckSpaceValue,
    LogicalPauliFrame,
    PhaseFreeQubitPauli,
    QubitRegister,
    stabilizer_logical_frame,
)


def _span(rows: tuple[int, ...] | list[int]) -> frozenset[int]:
    values = {0}
    for row in rows:
        values |= {value ^ row for value in tuple(values)}
    return frozenset(values)


def _pairing(left: int, right: int, n: int) -> int:
    mask = (1 << n) - 1
    left_x, left_z = left & mask, left >> n
    right_x, right_z = right & mask, right >> n
    return ((left_x & right_z).bit_count() + (left_z & right_x).bit_count()) % 2


def _pauli(register: QubitRegister, value: int) -> PhaseFreeQubitPauli:
    n = len(register.qubit_ids)
    return PhaseFreeQubitPauli(
        register=register,
        x_bits=tuple((value >> i) & 1 for i in range(n)),
        z_bits=tuple((value >> (n + i)) & 1 for i in range(n)),
    )


def _bits(row: PhaseFreeQubitPauli) -> int:
    n = len(row.x_bits)
    return sum(bit << i for i, bit in enumerate(row.x_bits)) | sum(
        bit << (n + i) for i, bit in enumerate(row.z_bits)
    )


def test_logical_frames_exhaust_all_two_qubit_isotropic_subspaces() -> None:
    n = 2
    spaces: set[frozenset[int]] = {frozenset({0})}
    for vector in range(1, 1 << (2 * n)):
        spaces.add(_span((vector,)))
    for left, right in combinations(range(1, 1 << (2 * n)), 2):
        if _pairing(left, right, n) == 0:
            spaces.add(_span((left, right)))

    register = QubitRegister(qubit_ids=("q0", "q1"))
    for span in spaces:
        basis = tuple(_pauli(register, value) for value in sorted(span - {0}))
        # The full span is accepted as a redundant generator family too; its
        # canonical result must be independent of that presentation.
        source = CheckSpaceValue(register=register, basis=basis)
        frame = stabilizer_logical_frame(source)
        assert isinstance(frame, LogicalPauliFrame)
        stabilizer = _span([_bits(row) for row in frame.check_space.basis])
        orthogonal = frozenset(
            candidate
            for candidate in range(1 << (2 * n))
            if all(_pairing(candidate, row, n) == 0 for row in stabilizer)
        )
        x_rows = tuple(_bits(row) for row in frame.x_logical_basis)
        z_rows = tuple(_bits(row) for row in frame.z_logical_basis)
        assert frame.logical_qubits == n - len(frame.check_space.basis)
        assert len(x_rows) == len(z_rows) == frame.logical_qubits
        assert all(row in orthogonal for row in (*x_rows, *z_rows))
        assert all(
            _pairing(left, right, n) == int(i == j)
            for i, left in enumerate(x_rows)
            for j, right in enumerate(z_rows)
        )
        assert all(
            _pairing(left, right, n) == 0
            for rows in (x_rows, z_rows)
            for i, left in enumerate(rows)
            for right in rows[i + 1 :]
        )
        assert _span((*stabilizer, *x_rows, *z_rows)) == orthogonal
        assert all(row not in stabilizer for row in (*x_rows, *z_rows))


def test_catalog_publishes_generic_mixed_pauli_logical_frame() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.logical_frame.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    frame = LogicalPauliFrame.model_validate(result.output)
    assert frame.logical_qubits == 1
    assert frame.x_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert frame.z_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert LogicalPauliFrame.model_validate(frame.model_dump(mode="json")) == frame

    outside_normalizer = frame.model_dump(mode="json")
    outside_normalizer["x_logical_basis"][0]["x_bits"] = [1, 0]
    outside_normalizer["x_logical_basis"][0]["z_bits"] = [0, 0]
    with pytest.raises(ValidationError, match="S-perp"):
        LogicalPauliFrame.model_validate(outside_normalizer)

    wrong_pairing = frame.model_dump(mode="json")
    wrong_pairing["z_logical_basis"][0] = wrong_pairing["x_logical_basis"][0].copy()
    with pytest.raises(ValidationError, match="canonical symplectic pairings"):
        LogicalPauliFrame.model_validate(wrong_pairing)
