"""Native exact binary stabilizer check-space canonicalization."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBIT_LABEL_LENGTH,
    MAX_QUBITS,
    BinaryPauliRow,
    CanonicalCheckRow,
    CheckSpaceCanonicalizeResult,
    CheckSpaceValue,
    ExactQubitPauli,
    NonCommutingWitness,
    NormalizerResult,
    PauliInverseResult,
    PauliPairingResult,
    PauliProductResult,
    PhaseFreeQubitPauli,
    QubitRegister,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
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
    if not isinstance(generators, (tuple, list)) or not generators:
        _reject(
            "generators",
            "stabilizer.check_space.generators_not_a_row_family",
            "check-space generators must be a nonempty Pauli row family",
        )
    validated_rows: list[BinaryPauliRow] = []
    for row in generators:
        if not isinstance(row, BinaryPauliRow):
            _reject(
                "generators",
                "stabilizer.check_space.generator_not_a_pauli_row",
                "every check-space generator must be a phase-free Pauli row",
            )
        validated_rows.append(row)
    rows = tuple(validated_rows)
    ids = tuple(qubit_ids)
    if any(
        not isinstance(qubit_id, str)
        or not qubit_id
        or len(qubit_id) > MAX_QUBIT_LABEL_LENGTH
        or any(0xD800 <= ord(character) <= 0xDFFF for character in qubit_id)
        for qubit_id in ids
    ):
        _reject(
            "qubit_ids",
            "stabilizer.check_space.qubit_id_not_strict_string",
            "qubit IDs must be nonempty Unicode scalar strings",
        )
    if len(set(ids)) != len(ids):
        _reject(
            "qubit_ids",
            "stabilizer.check_space.qubit_ids_not_unique",
            "qubit IDs must be unique",
        )
    row_ids = tuple(row.row_id for row in rows)
    if any(
        not isinstance(row_id, str)
        or not row_id
        or len(row_id) > MAX_QUBIT_LABEL_LENGTH
        or any(0xD800 <= ord(character) <= 0xDFFF for character in row_id)
        for row_id in row_ids
    ):
        _reject(
            "generators",
            "stabilizer.check_space.generator_row_id_not_strict_string",
            "generator row IDs must be nonempty Unicode scalar strings",
        )
    if tuple(sorted(row_ids)) != row_ids or len(set(row_ids)) != len(row_ids):
        _reject(
            "generators",
            "stabilizer.check_space.generator_row_ids",
            "generator row IDs must be unique and strictly ordered",
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
        if len(row.x_bits) != width or len(row.z_bits) != width:
            _reject(
                "generators",
                "stabilizer.check_space.register_binding_mismatch",
                "every generator bit row must match the register length",
            )
    return ids, rows


def _admit_register(register: object, location: str) -> QubitRegister:
    if not isinstance(register, QubitRegister):
        _reject(
            location, "quantum.pauli.invalid_register", "Pauli register is malformed"
        )
    ids = getattr(register, "qubit_ids", None)
    if (
        not isinstance(ids, tuple)
        or not 1 <= len(ids) <= MAX_QUBITS
        or any(
            type(value) is not str
            or not value
            or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            for value in ids
        )
        or len(set(ids)) != len(ids)
    ):
        _reject(
            location, "quantum.pauli.invalid_register", "Pauli register is malformed"
        )
    if any(len(value) > MAX_QUBIT_LABEL_LENGTH for value in ids):
        _reject(
            location,
            "quantum.pauli.invalid_register",
            "Pauli register labels exceed the envelope",
        )
    return register


def _admit_phase_free(value: object, location: str = "pauli") -> PhaseFreeQubitPauli:
    if not isinstance(value, PhaseFreeQubitPauli):
        _reject(
            location,
            "quantum.pauli.not_a_pauli",
            "Pauli operation requires typed exact Pauli values",
        )
    register = _admit_register(getattr(value, "qubit_register", None), location)
    x_bits = getattr(value, "x_bits", None)
    z_bits = getattr(value, "z_bits", None)
    width = len(register.qubit_ids)
    if (
        not isinstance(x_bits, tuple)
        or not isinstance(z_bits, tuple)
        or len(x_bits) != width
        or len(z_bits) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in (*x_bits, *z_bits))
    ):
        _reject(
            location,
            "quantum.pauli.invalid_bits",
            "Pauli coordinates must be binary rows on their register",
        )
    return value


def _admit_exact(value: object, location: str = "pauli") -> ExactQubitPauli:
    if not isinstance(value, ExactQubitPauli):
        _reject(
            location,
            "quantum.pauli.not_a_exact_pauli",
            "operation requires an exact phase-lifted Pauli",
        )
    _admit_phase_free(getattr(value, "phase_free", None), location)
    phase = getattr(value, "phase", None)
    if type(phase) is not int or not 0 <= phase <= 3:
        _reject(
            location,
            "quantum.pauli.invalid_phase",
            "Pauli phase must be an integer modulo four",
        )
    return value


def _admit_pauli_pair(left: PhaseFreeQubitPauli, right: PhaseFreeQubitPauli) -> None:
    _admit_phase_free(left, "left")
    _admit_phase_free(right, "right")
    if left.qubit_register != right.qubit_register:
        _reject(
            "right",
            "quantum.pauli.register_mismatch",
            "Paulis must use the identical ordered qubit register",
        )


def pauli_pairing(
    left: PhaseFreeQubitPauli, right: PhaseFreeQubitPauli
) -> PauliPairingResult:
    _admit_pauli_pair(left, right)
    pairing = (
        sum(
            x * z2 + z * x2
            for x, z, x2, z2 in zip(
                left.x_bits, left.z_bits, right.x_bits, right.z_bits, strict=True
            )
        )
        % 2
    )
    return PauliPairingResult(
        left=left, right=right, pairing=pairing, commute=pairing == 0
    )


def pauli_multiply(left: ExactQubitPauli, right: ExactQubitPauli) -> PauliProductResult:
    _admit_exact(left, "left")
    _admit_exact(right, "right")
    _admit_pauli_pair(left.phase_free, right.phase_free)
    phase = (
        left.phase
        + right.phase
        + 2
        * sum(
            z * x
            for z, x in zip(
                left.phase_free.z_bits, right.phase_free.x_bits, strict=True
            )
        )
    ) % 4
    product = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=left.register,
            x_bits=tuple(
                (x + y) % 2
                for x, y in zip(
                    left.phase_free.x_bits, right.phase_free.x_bits, strict=True
                )
            ),
            z_bits=tuple(
                (z + w) % 2
                for z, w in zip(
                    left.phase_free.z_bits, right.phase_free.z_bits, strict=True
                )
            ),
        ),
        phase=phase,
    )
    return PauliProductResult(left=left, right=right, product=product)


def pauli_inverse(value: ExactQubitPauli) -> PauliInverseResult:
    _admit_exact(value)
    phase = (
        -value.phase
        - 2
        * sum(
            x * z
            for x, z in zip(
                value.phase_free.x_bits, value.phase_free.z_bits, strict=True
            )
        )
    ) % 4
    inverse = ExactQubitPauli(
        phase_free=value.phase_free,
        phase=phase,
    )
    return PauliInverseResult(source=value, inverse=inverse)


def _gf2_nullspace(rows: list[list[int]], width: int) -> tuple[tuple[int, ...], ...]:
    rref, pivots = _gf2_rref(rows, width)
    pivot_set = set(pivots)
    free = [column for column in range(width) if column not in pivot_set]
    vectors: list[tuple[int, ...]] = []
    for free_column in free:
        vector = [0] * width
        vector[free_column] = 1
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = rref[row_index][free_column]
        vectors.append(tuple(vector))
    return tuple(vectors)


def stabilizer_normalizer(check_space: CheckSpaceValue) -> NormalizerResult:
    if not isinstance(check_space, CheckSpaceValue):
        _reject(
            "check_space",
            "quantum.stabilizer.not_a_check_space",
            "normalizer requires a typed register-bound check space",
        )
    register = _admit_register(
        getattr(check_space, "qubit_register", None), "check_space"
    )
    basis_value = getattr(check_space, "basis", None)
    if not isinstance(basis_value, tuple) or len(basis_value) > MAX_CHECK_ROWS:
        _reject(
            "check_space",
            "quantum.stabilizer.invalid_basis",
            "check-space basis is malformed",
        )
    basis = tuple(basis_value)
    for row in basis:
        _admit_phase_free(row, "check_space")
        if row.qubit_register != register:
            _reject(
                "check_space",
                "quantum.stabilizer.parent_mismatch",
                "all check rows must use the declared register",
            )
    for i, left in enumerate(basis):
        for right in basis[i + 1 :]:
            if pauli_pairing(left, right).pairing:
                _reject(
                    "check_space",
                    "quantum.stabilizer.not_isotropic",
                    "check space must be symplectically isotropic",
                )
    # Canonicalize the supplied row space before computing its orthogonal.
    flat = [[*row.x_bits, *row.z_bits] for row in basis]
    canonical_rows, _ = _gf2_rref(flat, 2 * len(register.qubit_ids))
    canonical_basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=tuple(row[: len(register.qubit_ids)]),
            z_bits=tuple(row[len(register.qubit_ids) :]),
        )
        for row in canonical_rows
    )
    constraints = [[*row.z_bits, *row.x_bits] for row in canonical_basis]
    orthogonal_flat = _gf2_nullspace(constraints, 2 * len(register.qubit_ids))
    orthogonal_basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=tuple(row[: len(register.qubit_ids)]),
            z_bits=tuple(row[len(register.qubit_ids) :]),
        )
        for row in orthogonal_flat
    )
    return NormalizerResult._from_kernel(
        check_space=CheckSpaceValue(register=register, basis=canonical_basis),
        orthogonal_basis=orthogonal_basis,
    )


def canonicalize_check_space(
    qubit_ids: tuple[str, ...] | list[str],
    generators: tuple[BinaryPauliRow, ...] | list[BinaryPauliRow],
) -> CheckSpaceCanonicalizeResult:
    """Canonicalize a binary check matrix to RREF or witness non-isotropy.

    Phase-free Paulis commute iff their symplectic pairing is zero, so the
    kernel first replays the complete pairwise pairing table: the first
    pair with pairing one becomes the explicit ``NOT_ISOTROPIC`` witness.
    Otherwise GF(2) elimination yields the canonical RREF basis, rank, and
    ``ISOTROPIC_CHECK_SPACE`` decision. The RREF depends only on the GF(2)
    row space; zero rows contribute no pivot and reduce the rank.
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
            if (
                _symplectic_pairing(tuple(table_rows[i]), tuple(table_rows[j]), qubits)
                != 0
            ):
                raise RuntimeError("isotropic check-space replay disagreed with search")
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


__all__ = [
    "canonicalize_check_space",
    "pauli_inverse",
    "pauli_multiply",
    "pauli_pairing",
    "stabilizer_normalizer",
]
