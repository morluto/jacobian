"""Native exact binary stabilizer check-space canonicalization."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBITS,
    BinaryPauliRow,
    CanonicalCheckRow,
    CheckSpaceCanonicalizeResult,
    NonCommutingWitness,
)


def _reject(location: str, code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _symplectic_pairing(
    first: tuple[int, ...], second: tuple[int, ...], qubits: int
) -> int:
    """Binary symplectic pairing ``x.z' + z.x' mod 2`` of two ``(x|z)`` rows."""

    total = 0
    for q in range(qubits):
        total += first[q] * second[qubits + q] + first[qubits + q] * second[q]
    return total % 2


def _gf2_rref(rows: list[list[int]], width: int) -> tuple[list[list[int]], list[int]]:
    """Reduced row-echelon form over GF(2) with pivot columns."""

    basis = [row[:] for row in rows]
    pivots: list[int] = []
    target = 0
    for column in range(width):
        pivot = next(
            (r for r in range(target, len(basis)) if basis[r][column] == 1), None
        )
        if pivot is None:
            continue
        basis[target], basis[pivot] = basis[pivot], basis[target]
        for r in range(len(basis)):
            if r != target and basis[r][column] == 1:
                basis[r] = [
                    (a + b) % 2 for a, b in zip(basis[r], basis[target], strict=True)
                ]
        pivots.append(column)
        target += 1
    return basis[:target], pivots


def _admit_canonicalize(
    qubit_ids: object, generators: object
) -> tuple[tuple[str, ...], tuple[BinaryPauliRow, ...]]:
    """Enforce the shared envelope for native and catalog calls."""

    if not isinstance(qubit_ids, (tuple, list)) or not qubit_ids:
        _reject(
            "qubit_ids",
            "stabilizer.check_space.register_not_a_qubit_family",
            "check-space register must be a nonempty qubit family",
        )
    assert isinstance(qubit_ids, (tuple, list))
    if not isinstance(generators, (tuple, list)) or not generators:
        _reject(
            "generators",
            "stabilizer.check_space.generators_not_a_row_family",
            "check-space generators must be a nonempty Pauli row family",
        )
    assert isinstance(generators, (tuple, list))
    for row in generators:
        if not isinstance(row, BinaryPauliRow):
            _reject(
                "generators",
                "stabilizer.check_space.generator_not_a_pauli_row",
                "every check-space generator must be a phase-free Pauli row",
            )
    rows = tuple(generators)
    ids = tuple(qubit_ids)
    if len(set(ids)) != len(ids):
        _reject(
            "qubit_ids",
            "stabilizer.check_space.qubit_ids_not_unique",
            "qubit IDs must be unique",
        )
    width = len(ids)
    if width > MAX_QUBITS:
        raise OperationResourceAdmissionError(
            location=("qubit_ids",),
            code="stabilizer.check_space.register_over_envelope",
            message=f"qubit register exceeds the {MAX_QUBITS}-qubit envelope",
        )
    if len(rows) > MAX_CHECK_ROWS:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="stabilizer.check_space.rows_over_envelope",
            message=f"generator rows exceed the {MAX_CHECK_ROWS}-row envelope",
        )
    for row in rows:
        assert isinstance(row, BinaryPauliRow)
        if len(row.x_bits) != width or len(row.z_bits) != width:
            _reject(
                "generators",
                "stabilizer.check_space.register_binding_mismatch",
                "every generator bit row must match the register length",
            )
    return tuple(str(q) for q in ids), rows


def canonicalize_check_space(
    qubit_ids: tuple[str, ...] | list[str],
    generators: tuple[BinaryPauliRow, ...] | list[BinaryPauliRow],
) -> CheckSpaceCanonicalizeResult:
    """Canonicalize a binary check matrix to RREF or witness non-isotropy.

    Phase-free Paulis commute iff their symplectic pairing is zero, so the
    kernel first replays the complete pairwise pairing table: the first
    pair with pairing one becomes the explicit ``NOT_ISOTROPIC`` witness.
    Otherwise GF(2) elimination yields the canonical RREF basis, rank, and
    ``ISOTROPIC_CHECK_SPACE`` decision. The RREF is independent of input
    row order; zero rows contribute no pivot and reduce the rank.
    """

    ids, rows = _admit_canonicalize(qubit_ids, generators)
    qubits = len(ids)
    flat = [(*row.x_bits, *row.z_bits) for row in rows]
    table_rows = [[int(bit) for bit in row] for row in flat]
    for i in range(len(table_rows)):
        for j in range(i + 1, len(table_rows)):
            if _symplectic_pairing(tuple(table_rows[i]), tuple(table_rows[j]), qubits):
                first, second = rows[i].row_id, rows[j].row_id
                ordered = (min(first, second), max(first, second))
                return CheckSpaceCanonicalizeResult._from_kernel(
                    status="NOT_ISOTROPIC",
                    qubit_ids=ids,
                    rank=0,
                    basis=(),
                    witness=NonCommutingWitness(
                        first_row_id=ordered[0],
                        second_row_id=ordered[1],
                        pairing=1,
                    ),
                )
    # Replay isotropy independently of the witness search above.
    for i in range(len(table_rows)):
        for j in range(i + 1, len(table_rows)):
            assert (
                _symplectic_pairing(tuple(table_rows[i]), tuple(table_rows[j]), qubits)
                == 0
            )
    basis_rows, pivots = _gf2_rref(table_rows, 2 * qubits)
    basis = tuple(
        CanonicalCheckRow(
            pivot=pivot,
            x_bits=tuple(row[:qubits]),
            z_bits=tuple(row[qubits:]),
        )
        for row, pivot in zip(basis_rows, pivots, strict=True)
    )
    return CheckSpaceCanonicalizeResult._from_kernel(
        status="ISOTROPIC_CHECK_SPACE",
        qubit_ids=ids,
        rank=len(basis),
        basis=basis,
        witness=None,
    )


__all__ = ["canonicalize_check_space"]
