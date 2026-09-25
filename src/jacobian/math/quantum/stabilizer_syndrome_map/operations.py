"""Bounded construction of the exact binary stabilizer syndrome map."""

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
    CheckSpaceValue,
    NormalizerResult,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.operations import stabilizer_normalizer
from jacobian.math.quantum.stabilizer_syndrome_map._models import (
    MAX_SYNDROME_MAP_CHECK_ROWS,
    MAX_SYNDROME_MAP_RESULT_BYTES,
    MAX_SYNDROME_MAP_WORK,
    SyndromeMapResult,
)


def _domain_error(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _require_source_shape(check_space: object) -> tuple[QubitRegister, int, int]:
    if not isinstance(check_space, CheckSpaceValue):
        _domain_error(
            "check_space",
            "quantum.syndrome_map.not_a_check_space",
            "syndrome map requires a typed register-bound check space",
        )
    register = getattr(check_space, "qubit_register", None)
    if not isinstance(register, QubitRegister):
        _domain_error(
            "check_space",
            "quantum.syndrome_map.invalid_register",
            "check-space register is malformed",
        )
    ids = getattr(register, "qubit_ids", None)
    if (
        not isinstance(ids, tuple)
        or not 1 <= len(ids) <= MAX_QUBITS
        or any(
            type(label) is not str
            or not label
            or len(label) > MAX_QUBIT_LABEL_LENGTH
            or any(0xD800 <= ord(character) <= 0xDFFF for character in label)
            for label in ids
        )
        or len(set(ids)) != len(ids)
    ):
        _domain_error(
            "check_space",
            "quantum.syndrome_map.invalid_register",
            "check-space register is outside the bounded canonical envelope",
        )
    rows = getattr(check_space, "basis", None)
    if (
        not isinstance(rows, tuple)
        or len(rows) > MAX_SYNDROME_MAP_CHECK_ROWS
        or len(rows) > MAX_CHECK_ROWS
    ):
        _domain_error(
            "check_space",
            "quantum.syndrome_map.input_rows_exceeded",
            f"syndrome map accepts at most {MAX_SYNDROME_MAP_CHECK_ROWS} check rows",
        )
    n = len(ids)
    for row in rows:
        if not isinstance(row, PhaseFreeQubitPauli):
            _domain_error(
                "check_space",
                "quantum.syndrome_map.invalid_row",
                "check-space rows must be typed phase-free Paulis",
            )
        if row.qubit_register != register:
            _domain_error(
                "check_space",
                "quantum.syndrome_map.parent_mismatch",
                "all check rows must use the declared ordered register",
            )
        if (
            not isinstance(row.x_bits, tuple)
            or not isinstance(row.z_bits, tuple)
            or len(row.x_bits) != n
            or len(row.z_bits) != n
            or any(
                type(bit) is not int or bit not in (0, 1)
                for bit in (*row.x_bits, *row.z_bits)
            )
        ):
            _domain_error(
                "check_space",
                "quantum.syndrome_map.invalid_row",
                "check rows must be binary vectors on the full register axis",
            )
    return register, n, len(rows)


def _result_bytes_upper_bound(
    check_space: CheckSpaceValue, n: int, input_rows: int
) -> int:
    """Conservatively bound JSON output before canonical/kernel expansion."""

    register_bytes = len(check_space.qubit_register.model_dump_json().encode("utf-8"))
    source_bytes = len(check_space.model_dump_json().encode("utf-8"))
    width = 2 * n
    # A serialized PhaseFreeQubitPauli repeats the parent register and has 2n
    # one-digit bits. Include generous fixed syntax/key overhead per row.
    pauli_bytes = register_bytes + (2 * n) + 256
    canonical_rows = min(input_rows, width)
    kernel_rows = width
    # The normalizer carries its check-space parent plus a basis of S-perp.
    normalizer_bytes = (
        register_bytes
        + canonical_rows * (pauli_bytes + 1)
        + kernel_rows * (pauli_bytes + 1)
        + 4_096
    )
    matrix_bytes = input_rows * width + input_rows * 4 + 256
    scalar_metadata_bytes = 2_048
    return source_bytes + normalizer_bytes + matrix_bytes + scalar_metadata_bytes


def _estimated_work(n: int, rows: int) -> int:
    width = 2 * n
    isotropy_pairings = rows * (rows - 1) // 2 * n
    canonical_rref = rows * width * width
    kernel_rref = rows * width * width
    kernel_output = width * width
    map_output = rows * width
    return isotropy_pairings + canonical_rref + kernel_rref + kernel_output + map_output


def _admit_work(n: int, rows: int, result_bytes: int) -> None:
    width = 2 * n
    if width > 2 * MAX_QUBITS:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.syndrome_map.fiber_exponent_exceeded",
            message="syndrome fiber cardinality exponent exceeds its exact bound",
        )
    work = _estimated_work(n, rows)
    if work > MAX_SYNDROME_MAP_WORK:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.syndrome_map.work_exceeded",
            message=(
                f"syndrome map and kernel work exceeds the "
                f"{MAX_SYNDROME_MAP_WORK}-unit envelope"
            ),
        )
    if result_bytes > MAX_SYNDROME_MAP_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("check_space",),
            code="quantum.syndrome_map.output_exceeded",
            message=(
                "syndrome map, kernel, and source exceed the "
                f"{MAX_SYNDROME_MAP_RESULT_BYTES}-byte compact JSON envelope"
            ),
        )


def syndrome_map(check_space: CheckSpaceValue) -> SyndromeMapResult:
    """Return ``e |-> (<g_i,e>)`` into ``S*`` on the canonical check axis.

    The map domain uses flattened phase-free coordinates ``[x|z]``. Its row
    for check ``g=(x_g|z_g)`` is ``[z_g|x_g]`` because the qubit symplectic
    pairing is ``x_g·z_e + z_g·x_e`` over GF(2). The returned normalizer basis
    is exactly the kernel ``S^perp``.
    """

    _, n, input_rows = _require_source_shape(check_space)
    estimated_bytes = _result_bytes_upper_bound(check_space, n, input_rows)
    _admit_work(n, input_rows, estimated_bytes)

    normalizer: NormalizerResult = stabilizer_normalizer(check_space)
    matrix = tuple((*row.z_bits, *row.x_bits) for row in normalizer.check_space.basis)
    return SyndromeMapResult._from_kernel(
        source_check_space=check_space,
        normalizer=normalizer,
        matrix=matrix,
    )


__all__ = ["syndrome_map"]
