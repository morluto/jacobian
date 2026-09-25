"""Exact linear syndrome maps on bounded binary stabilizer spaces."""

import json
from itertools import product

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    stabilizer_syndrome,
)
from jacobian.math.quantum.stabilizer_syndrome_map import (
    SyndromeMapRequest,
    SyndromeMapResult,
    syndrome_map,
)
from jacobian.math.quantum.stabilizer_syndrome_map import operations as map_operations
from jacobian.math.quantum.stabilizer_syndrome_map._models import (
    MAX_SYNDROME_MAP_RESULT_BYTES,
    MAX_SYNDROME_MAP_WORK,
)
from jacobian.math.quantum.stabilizer_syndrome_map._tools import TOOLS
from jacobian.math.quantum.stabilizer_syndrome_map.operations import (
    _estimated_work,
    _result_bytes_upper_bound,
)


def _pauli(
    register: QubitRegister, x: tuple[int, ...], z: tuple[int, ...]
) -> PhaseFreeQubitPauli:
    return PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z)


def _pairing(
    left_x: tuple[int, ...],
    left_z: tuple[int, ...],
    right_x: tuple[int, ...],
    right_z: tuple[int, ...],
) -> int:
    return (
        sum(
            x * right_z_bit + z * right_x_bit
            for x, z, right_z_bit, right_x_bit in zip(
                left_x, left_z, right_z, right_x, strict=True
            )
        )
        % 2
    )


def _apply_matrix(
    matrix: tuple[tuple[int, ...], ...], coordinates: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        sum(
            coefficient * bit for coefficient, bit in zip(row, coordinates, strict=True)
        )
        % 2
        for row in matrix
    )


def test_map_rows_match_direct_symplectic_oracle_and_existing_syndrome() -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    zz = _pauli(register, (0, 0), (1, 1))
    source = CheckSpaceValue(register=register, basis=(zz, zz))
    result = syndrome_map(source)

    assert result.source_check_space == source
    assert result.normalizer.check_space.basis == (zz,)
    assert result.matrix == ((1, 1, 0, 0),)
    assert result.field_order == 2
    assert result.rank == result.codomain_dimension == 1
    assert result.domain_dimension == 4
    assert result.kernel_dimension == result.fiber_dimension == 3
    assert result.fiber_cardinality == "8"

    for coordinates in product((0, 1), repeat=4):
        error = _pauli(register, coordinates[:2], coordinates[2:])
        expected = (_pairing((0, 0), (1, 1), coordinates[:2], coordinates[2:]),)
        assert _apply_matrix(result.matrix, coordinates) == expected
        assert stabilizer_syndrome(source, error).syndrome == expected


def test_empty_map_and_lagrangian_kernel_dimensions() -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    empty = syndrome_map(CheckSpaceValue(register=register, basis=()))
    assert empty.matrix == ()
    assert empty.rank == empty.codomain_dimension == 0
    assert empty.domain_dimension == empty.kernel_dimension == 4
    assert empty.fiber_dimension == 4
    assert empty.fiber_cardinality == "16"
    assert len(empty.normalizer.orthogonal_basis) == 4

    checks = (
        _pauli(register, (1, 1), (0, 0)),
        _pauli(register, (0, 0), (1, 1)),
    )
    lagrangian = syndrome_map(CheckSpaceValue(register=register, basis=checks))
    assert lagrangian.rank == 2
    assert lagrangian.kernel_dimension == lagrangian.fiber_dimension == 2
    assert lagrangian.fiber_cardinality == "4"
    assert (
        lagrangian.normalizer.orthogonal_basis
        == lagrangian.normalizer.check_space.basis
    )


def test_map_result_roundtrips_and_catalog_publishes_operation() -> None:
    register = QubitRegister(qubit_ids=("q",))
    check = _pauli(register, (0,), (1,))
    result = syndrome_map(CheckSpaceValue(register=register, basis=(check,)))

    assert SyndromeMapResult.model_validate_json(result.model_dump_json()) == result
    assert len(result.model_dump_json().encode("utf-8")) <= _result_bytes_upper_bound(
        result.source_check_space,
        len(register.qubit_ids),
        len(result.source_check_space.basis),
    )
    assert (
        SyndromeMapRequest.model_validate_json(
            SyndromeMapRequest(check_space=result.source_check_space).model_dump_json()
        ).check_space
        == result.source_check_space
    )
    assert len(TOOLS) == 1
    assert TOOLS[0].operation_id == "quantum.stabilizer.syndrome_map.compute"
    assert TOOLS[0] in BUILTIN_TOOLS
    assert TOOLS[0].result_type is SyndromeMapResult


def test_worst_label_and_kernel_output_fixture_stays_bounded() -> None:
    ids = tuple("😀" * 63 + chr(ord("A") + index) for index in range(32))
    register = QubitRegister(qubit_ids=ids)
    zero = _pauli(register, (0,) * 32, (0,) * 32)
    source = CheckSpaceValue(register=register, basis=(zero,) * 64)

    result = syndrome_map(source)
    output_bytes = len(result.model_dump_json().encode("utf-8"))
    output_bound = _result_bytes_upper_bound(source, 32, 64)

    assert result.rank == 0
    assert result.kernel_dimension == 64
    assert output_bytes <= output_bound <= MAX_SYNDROME_MAP_RESULT_BYTES


def test_work_admission_precedes_normalizer_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    row = _pauli(register, (0, 0), (1, 1))
    source = CheckSpaceValue(register=register, basis=(row, row))
    monkeypatch.setattr(map_operations, "MAX_SYNDROME_MAP_WORK", 0)

    def fail_if_expanded(_space: CheckSpaceValue) -> None:
        raise AssertionError("normalizer kernel ran before syndrome-map admission")

    monkeypatch.setattr(map_operations, "stabilizer_normalizer", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError, match="work exceeds"):
        map_operations.syndrome_map(source)


def test_forged_oversized_source_rejects_before_normalizer_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = QubitRegister(qubit_ids=("q",))
    zero = _pauli(register, (0,), (0,))
    source = CheckSpaceValue.model_construct(
        qubit_register=register, basis=(zero,) * 65
    )

    def fail_if_expanded(_space: CheckSpaceValue) -> None:
        raise AssertionError("oversized basis reached the normalizer")

    monkeypatch.setattr(map_operations, "stabilizer_normalizer", fail_if_expanded)
    with pytest.raises(OperationDomainValidationError, match="at most 64 check rows"):
        map_operations.syndrome_map(source)


def test_work_limit_accepts_exact_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    row = _pauli(register, (0, 0), (1, 1))
    source = CheckSpaceValue(register=register, basis=(row, row))
    exact_work = _estimated_work(2, 2)
    monkeypatch.setattr(map_operations, "MAX_SYNDROME_MAP_WORK", exact_work)

    assert syndrome_map(source).rank == 1


def test_output_admission_precedes_normalizer_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    source = CheckSpaceValue(register=register, basis=())
    monkeypatch.setattr(map_operations, "MAX_SYNDROME_MAP_RESULT_BYTES", 0)

    def fail_if_expanded(_space: CheckSpaceValue) -> None:
        raise AssertionError("normalizer kernel ran before output admission")

    monkeypatch.setattr(map_operations, "stabilizer_normalizer", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError, match="compact JSON envelope"):
        map_operations.syndrome_map(source)


def test_output_byte_limit_accepts_exact_conservative_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    source = CheckSpaceValue(register=register, basis=())
    exact_bound = _result_bytes_upper_bound(source, 2, 0)
    monkeypatch.setattr(map_operations, "MAX_SYNDROME_MAP_RESULT_BYTES", exact_bound)

    assert syndrome_map(source).rank == 0
    monkeypatch.setattr(
        map_operations, "MAX_SYNDROME_MAP_RESULT_BYTES", exact_bound - 1
    )

    def fail_if_expanded(_space: CheckSpaceValue) -> None:
        raise AssertionError("normalizer kernel ran before output admission")

    monkeypatch.setattr(map_operations, "stabilizer_normalizer", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError, match="compact JSON envelope"):
        map_operations.syndrome_map(source)


def test_admission_limits_are_published() -> None:
    schema = SyndromeMapRequest.model_json_schema()
    limits = schema["admission_limits"]
    assert limits["max_qubits"] == 32
    assert limits["max_input_check_rows"] == 64
    assert limits["max_work_units"] == MAX_SYNDROME_MAP_WORK
    assert limits["max_result_compact_json_bytes"] == MAX_SYNDROME_MAP_RESULT_BYTES


def test_map_result_wire_shape_rejects_wrong_matrix_dimensions() -> None:
    register = QubitRegister(qubit_ids=("q",))
    check = _pauli(register, (0,), (1,))
    result = syndrome_map(CheckSpaceValue(register=register, basis=(check,)))
    payload = json.loads(result.model_dump_json())
    payload["matrix"] = [[1, 0, 0]]

    with pytest.raises(ValueError, match="matrix and dimensions must match"):
        SyndromeMapResult.model_validate(payload)
