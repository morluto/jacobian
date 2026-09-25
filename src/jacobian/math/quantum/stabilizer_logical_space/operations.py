"""Construct the exact binary symplectic quotient of a stabilizer space."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.quantum._models import (
    MAX_CHECK_ROWS,
    MAX_QUBITS,
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.operations import stabilizer_normalizer
from jacobian.math.quantum.stabilizer_logical_space._models import (
    LogicalPauliSpace,
    _ambient_axis,
    _check_axis,
    _normalizer_axis,
    _quotient_axis,
)

MAX_LOGICAL_SPACE_WORK = 4_000_000
MAX_LOGICAL_SPACE_RESULT_BYTES = 3_000_000
_GF2 = FiniteFieldPresentation(
    characteristic=2,
    modulus_coefficients=(0, 1),
    generator="a",
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _admit_check_space(value: object) -> tuple[CheckSpaceValue, int, int, int]:
    if not isinstance(value, CheckSpaceValue):
        _reject(
            "check_space",
            "quantum.stabilizer.logical_pauli_space.invalid_check_space",
            "input must be a typed register-bound check space",
        )
    register = getattr(value, "qubit_register", None)
    ids = getattr(register, "qubit_ids", None)
    if (
        not isinstance(register, QubitRegister)
        or not isinstance(ids, tuple)
        or not 1 <= len(ids) <= MAX_QUBITS
        or any(
            type(qubit) is not str
            or not qubit
            or len(qubit) > 64
            or any(0xD800 <= ord(character) <= 0xDFFF for character in qubit)
            for qubit in ids
        )
        or len(set(ids)) != len(ids)
    ):
        _reject(
            "check_space.register",
            "quantum.stabilizer.logical_pauli_space.invalid_register",
            "check-space register must have unique bounded Unicode scalar labels",
        )
    rows = getattr(value, "basis", None)
    if not isinstance(rows, tuple) or len(rows) > MAX_CHECK_ROWS:
        _reject(
            "check_space.basis",
            "quantum.stabilizer.logical_pauli_space.invalid_basis",
            "check-space basis exceeds its row envelope",
        )
    width = len(ids)
    for index, row in enumerate(rows):
        if (
            not isinstance(row, PhaseFreeQubitPauli)
            or row.qubit_register != register
            or type(row.x_bits) is not tuple
            or type(row.z_bits) is not tuple
            or len(row.x_bits) != width
            or len(row.z_bits) != width
            or any(
                type(bit) is not int or bit not in (0, 1)
                for bit in (*row.x_bits, *row.z_bits)
            )
        ):
            _reject(
                f"check_space.basis[{index}]",
                "quantum.stabilizer.logical_pauli_space.invalid_row",
                "check rows must be binary Paulis on the declared ordered register",
            )
    label_bytes = sum(len(qubit.encode("utf-8")) for qubit in ids)
    return value, width, len(rows), label_bytes


def _work_and_output_bounds(
    qubits: int, check_rows: int, label_bytes: int
) -> tuple[int, int]:
    ambient_dimension = 2 * qubits
    normalizer_dimension = ambient_dimension
    quotient_dimension = ambient_dimension
    pair_count = check_rows * (check_rows - 1) // 2
    matrix_cells = (
        normalizer_dimension * check_rows
        + ambient_dimension * normalizer_dimension
        + quotient_dimension * normalizer_dimension
        + normalizer_dimension * quotient_dimension
        + quotient_dimension * quotient_dimension
    )
    work = (
        pair_count * ambient_dimension
        + 6 * ambient_dimension * ambient_dimension * max(1, check_rows)
        + 6 * ambient_dimension * ambient_dimension * normalizer_dimension
        + 8 * matrix_cells
        + 32 * ambient_dimension * check_rows
    )
    # Only check rows repeat the full register value in JSON; the normalizer
    # and quotient bases are represented by their axis-bound maps.
    pauli_rows = check_rows + 1
    repeated_register_bytes = label_bytes + 40 * qubits + 256
    axis_bytes = (
        2 * label_bytes
        + 24
        * (ambient_dimension + check_rows + normalizer_dimension + quotient_dimension)
        + 512
    )
    output_bytes = (
        pauli_rows * repeated_register_bytes
        + 10 * axis_bytes
        + 12 * matrix_cells
        + 4096
    )
    return work, output_bytes


def _int_row(row: PhaseFreeQubitPauli) -> int:
    bits = (*row.x_bits, *row.z_bits)
    return sum(bit << position for position, bit in enumerate(bits))


def _vector_from_int(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> position) & 1 for position in range(width))


def _insert_row(basis: dict[int, int], value: int) -> bool:
    """Add a row to an incremental reduced GF(2) basis if it is independent."""
    reduced = value
    for pivot in sorted(basis):
        if (reduced >> pivot) & 1:
            reduced ^= basis[pivot]
    if not reduced:
        return False
    pivot = (reduced & -reduced).bit_length() - 1
    for old_pivot, old_row in tuple(basis.items()):
        if (old_row >> pivot) & 1:
            basis[old_pivot] = old_row ^ reduced
    basis[pivot] = reduced
    return True


def _rref_with_transform(
    rows: tuple[tuple[int, ...], ...], width: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return RREF, its row-operation matrix, and pivots over GF(2)."""
    reduced = [list(row) for row in rows]
    size = len(reduced)
    transform = [[int(row == column) for column in range(size)] for row in range(size)]
    pivots: list[int] = []
    target = 0
    for column in range(width):
        pivot = next((row for row in range(target, size) if reduced[row][column]), None)
        if pivot is None:
            continue
        reduced[target], reduced[pivot] = reduced[pivot], reduced[target]
        transform[target], transform[pivot] = transform[pivot], transform[target]
        for row in range(size):
            if row != target and reduced[row][column]:
                reduced[row] = [
                    left ^ right
                    for left, right in zip(reduced[row], reduced[target], strict=True)
                ]
                transform[row] = [
                    left ^ right
                    for left, right in zip(
                        transform[row], transform[target], strict=True
                    )
                ]
        pivots.append(column)
        target += 1
    return (
        tuple(tuple(row) for row in reduced[:target]),
        tuple(tuple(row) for row in transform[:target]),
        tuple(pivots),
    )


def _coordinate_in_row_basis(
    vector: tuple[int, ...],
    transform: tuple[tuple[int, ...], ...],
    pivots: tuple[int, ...],
) -> tuple[int, ...]:
    reduced_coordinates = tuple(vector[pivot] for pivot in pivots)
    return tuple(
        sum(
            reduced_coordinates[row] * transform[row][column]
            for row in range(len(reduced_coordinates))
        )
        % 2
        for column in range(len(reduced_coordinates))
    )


def _linear_map(
    source: Axis,
    target: Axis,
    rows: tuple[tuple[int, ...], ...],
) -> FiniteLinearMap:
    return FiniteLinearMap(
        source_axis=source,
        target_axis=target,
        matrix=PrimeFieldMatrix(
            prime=2,
            entries=rows,
            columns=len(source.labels),
        ),
    )


def logical_pauli_space(check_space: CheckSpaceValue) -> LogicalPauliSpace:
    """Construct the canonical coordinate data of ``S^perp / S``.

    Quotient representatives are selected deterministically by extending the
    canonical check basis with rows of the canonical normalizer basis in order.
    """
    source, qubits, raw_check_count, label_bytes = _admit_check_space(check_space)
    work_bound, output_bound = _work_and_output_bounds(
        qubits, raw_check_count, label_bytes
    )
    if work_bound > MAX_LOGICAL_SPACE_WORK:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.stabilizer.logical_pauli_space.work_over_envelope",
            message="logical quotient construction exceeds its GF(2) work envelope",
        )
    if output_bound > MAX_LOGICAL_SPACE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.stabilizer.logical_pauli_space.output_over_envelope",
            message="logical quotient maps exceed their exact output envelope",
        )

    normalizer = stabilizer_normalizer(source)
    canonical_space = normalizer.check_space
    register = canonical_space.qubit_register
    qubits = len(register.qubit_ids)
    ambient_dimension = 2 * qubits
    check_rows = tuple(_int_row(row) for row in canonical_space.basis)
    normalizer_rows = tuple(_int_row(row) for row in normalizer.orthogonal_basis)
    normalizer_dimension = len(normalizer_rows)

    # The normalizer basis is the canonical right-nullspace basis: its free
    # ambient columns are identity coordinates. Recover those columns from the
    # retained basis so the serialized value remains independently usable.
    coordinate_positions: list[int] = []
    for basis_index in range(normalizer_dimension):
        unit = 1 << basis_index
        position = next(
            (
                column
                for column in range(ambient_dimension)
                if sum(
                    ((row >> column) & 1) << index
                    for index, row in enumerate(normalizer_rows)
                )
                == unit
            ),
            None,
        )
        if position is None:
            _reject(
                "check_space",
                "quantum.stabilizer.logical_pauli_space.normalizer_coordinates",
                "normalizer basis must retain independent ambient coordinate selectors",
            )
        coordinate_positions.append(position)
    positions = tuple(coordinate_positions)

    # Extend S by a canonical subset of the nullspace basis. These rows are a
    # section of the quotient map and form the returned ambient representatives.
    span: dict[int, int] = {}
    for row in check_rows:
        _insert_row(span, row)
    selected_indices: list[int] = []
    quotient_rows: list[int] = []
    for index, row in enumerate(normalizer_rows):
        if _insert_row(span, row):
            selected_indices.append(index)
            quotient_rows.append(row)
    quotient_dimension = len(quotient_rows)
    if quotient_dimension != normalizer.logical_dimension:
        _reject(
            "check_space",
            "quantum.stabilizer.logical_pauli_space.dimension_mismatch",
            "the quotient complement must have dimension dim(S-perp)-dim(S)",
        )

    # Coordinates in the row basis S followed by quotient representatives give
    # both the projection N -> N/S and its chosen linear section.
    spanning_rows = tuple(
        _vector_from_int(row, ambient_dimension)
        for row in (*check_rows, *quotient_rows)
    )
    _, transform, pivots = _rref_with_transform(spanning_rows, ambient_dimension)
    if len(pivots) != normalizer_dimension:
        _reject(
            "check_space",
            "quantum.stabilizer.logical_pauli_space.basis_failure",
            "checks and quotient representatives must form a basis of the normalizer",
        )
    projection_columns = tuple(
        _coordinate_in_row_basis(
            _vector_from_int(row, ambient_dimension), transform, pivots
        )[len(check_rows) :]
        for row in normalizer_rows
    )
    projection_matrix = tuple(
        tuple(projection_columns[column][row] for column in range(normalizer_dimension))
        for row in range(quotient_dimension)
    )
    inclusion_columns = tuple(
        tuple((row >> position) & 1 for position in positions) for row in check_rows
    )
    inclusion_matrix = tuple(
        tuple(inclusion_columns[column][row] for column in range(len(check_rows)))
        for row in range(normalizer_dimension)
    )
    lift_matrix = tuple(
        tuple(int(row == selected) for selected in selected_indices)
        for row in range(normalizer_dimension)
    )

    normalizer_axis = _normalizer_axis(normalizer_dimension)
    quotient_axis = _quotient_axis(quotient_dimension)
    ambient_axis = _ambient_axis(canonical_space)
    check_axis = _check_axis(canonical_space)
    normalizer_embedding = tuple(
        tuple((row >> ambient) & 1 for row in normalizer_rows)
        for ambient in range(ambient_dimension)
    )

    def pairing(left: int, right: int) -> int:
        return (
            sum(
                ((left >> index) & 1) * ((right >> (qubits + index)) & 1)
                + ((left >> (qubits + index)) & 1) * ((right >> index) & 1)
                for index in range(qubits)
            )
            % 2
        )

    form_values = tuple(
        tuple(
            FiniteFieldElement(presentation=_GF2, coordinates=(pairing(left, right),))
            for right in quotient_rows
        )
        for left in quotient_rows
    )
    induced_form = AxisBoundMatrix(
        presentation=_GF2,
        row_axis=quotient_axis,
        column_axis=quotient_axis,
        entries=form_values,
    )

    return LogicalPauliSpace._from_kernel(
        check_space=canonical_space,
        stabilizer_inclusion=_linear_map(check_axis, normalizer_axis, inclusion_matrix),
        normalizer_embedding=_linear_map(
            normalizer_axis, ambient_axis, normalizer_embedding
        ),
        quotient_projection=_linear_map(
            normalizer_axis, quotient_axis, projection_matrix
        ),
        quotient_lift=_linear_map(quotient_axis, normalizer_axis, lift_matrix),
        induced_symplectic_form=induced_form,
    )


__all__ = ["logical_pauli_space"]
