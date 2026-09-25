from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.stabilizer_logical_space import (
    LogicalPauliSpace,
    logical_pauli_space,
)
from jacobian.math.quantum.stabilizer_logical_space._tools import TOOLS


def _bits(pauli: PhaseFreeQubitPauli) -> int:
    n = len(pauli.x_bits)
    return sum(bit << index for index, bit in enumerate(pauli.x_bits)) | sum(
        bit << (n + index) for index, bit in enumerate(pauli.z_bits)
    )


def _pauli(register: QubitRegister, value: int) -> PhaseFreeQubitPauli:
    n = len(register.qubit_ids)
    return PhaseFreeQubitPauli(
        register=register,
        x_bits=tuple((value >> index) & 1 for index in range(n)),
        z_bits=tuple((value >> (n + index)) & 1 for index in range(n)),
    )


def _span(rows: tuple[int, ...]) -> frozenset[int]:
    result = {0}
    for row in rows:
        result |= {value ^ row for value in tuple(result)}
    return frozenset(result)


def _pairing(left: int, right: int, n: int) -> int:
    mask = (1 << n) - 1
    left_x, left_z = left & mask, left >> n
    right_x, right_z = right & mask, right >> n
    return ((left_x & right_z).bit_count() + (left_z & right_x).bit_count()) % 2


def _apply_map(linear_map, vector: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(entry * coordinate for entry, coordinate in zip(row, vector, strict=True))
        % 2
        for row in linear_map.matrix.entries
    )


def _coordinates(vector: int, positions: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((vector >> position) & 1 for position in positions)


@pytest.mark.parametrize(
    ("n", "checks"),
    [
        (1, ()),
        (1, (0b10,)),  # A maximal check space has zero symplectic rank.
        (2, (0b0100,)),  # Z0 leaves one logical qubit.
        (2, (0b1001, 0b0110)),  # Bell checks have no logical quotient.
        (2, (0b0011, 0b1100)),  # A mixed, non-CSS presentation.
    ],
)
def test_quotient_maps_and_form_match_exhaustive_pauli_oracle(
    n: int, checks: tuple[int, ...]
) -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
    source = CheckSpaceValue(
        register=register,
        basis=tuple(_pauli(register, row) for row in checks),
    )
    space = logical_pauli_space(source)

    stabilizer = _span(tuple(_bits(row) for row in space.check_space.basis))
    normalizer = frozenset(
        value
        for value in range(1 << (2 * n))
        if all(_pairing(value, check, n) == 0 for check in stabilizer)
    )
    normalizer_basis = tuple(_bits(row) for row in space.normalizer_basis)
    quotient_basis = tuple(_bits(row) for row in space.quotient_basis)
    assert _span(normalizer_basis) == normalizer
    assert 1 << len(quotient_basis) == len(normalizer) // len(stabilizer)
    assert all(representative in normalizer for representative in quotient_basis)

    # Each ambient normalizer vector gets its canonical coordinates by reading
    # the retained coordinate-selector positions. The projection's kernel is S.
    projected_classes: set[frozenset[int]] = set()
    for value in normalizer:
        normalizer_coordinates = _coordinates(
            value, space.normalizer_coordinate_positions
        )
        assert _apply_map(space.normalizer_embedding, normalizer_coordinates) == tuple(
            (value >> index) & 1 for index in range(2 * n)
        )
        quotient_coordinates = _apply_map(
            space.quotient_projection, normalizer_coordinates
        )
        representative_coordinates = _apply_map(
            space.quotient_lift, quotient_coordinates
        )
        representative = 0
        for row, coefficient in zip(
            space.normalizer_basis, representative_coordinates, strict=True
        ):
            if coefficient:
                representative ^= _bits(row)
        assert _apply_map(
            space.normalizer_embedding, representative_coordinates
        ) == tuple((representative >> index) & 1 for index in range(2 * n))
        projected_classes.add(
            frozenset(representative ^ element for element in stabilizer)
        )

    oracle_classes = {
        frozenset(value ^ stabilizer_element for stabilizer_element in stabilizer)
        for value in normalizer
    }
    assert projected_classes == oracle_classes
    assert len(projected_classes) == 1 << (2 * space.logical_qubits)

    # The stored matrix is the exact form induced from ambient Pauli pairing.
    form = space.induced_symplectic_form.entries
    assert tuple(tuple(entry.coordinates[0] for entry in row) for row in form) == tuple(
        tuple(_pairing(left, right, n) for right in quotient_basis)
        for left in quotient_basis
    )
    assert all(form[i][i].is_zero for i in range(len(form)))

    # The quotient pairing must be nondegenerate, and changing either ambient
    # representative by an element of S must not change its pairing.
    quotient_vectors = tuple(range(1 << len(quotient_basis)))
    for left_coordinates in quotient_vectors:
        left = 0
        for index, row in enumerate(quotient_basis):
            if (left_coordinates >> index) & 1:
                left ^= row
        assert _pairing(left, left, n) == 0
        if left_coordinates:
            assert any(
                sum(
                    ((left_coordinates >> i) & 1)
                    * ((right_coordinates >> j) & 1)
                    * form[i][j].coordinates[0]
                    for i in range(len(quotient_basis))
                    for j in range(len(quotient_basis))
                )
                % 2
                for right_coordinates in quotient_vectors
                if right_coordinates
            )
        for right_coordinates in quotient_vectors:
            right = 0
            for index, row in enumerate(quotient_basis):
                if (right_coordinates >> index) & 1:
                    right ^= row
            assert all(
                _pairing(left ^ stabilizer_element, right ^ other_stabilizer, n)
                == _pairing(left, right, n)
                for stabilizer_element in stabilizer
                for other_stabilizer in stabilizer
            )

    # The representative section is a right inverse to quotient projection.
    for coordinates in itertools.product((0, 1), repeat=len(quotient_basis)):
        lifted = _apply_map(space.quotient_lift, coordinates)
        assert _apply_map(space.quotient_projection, lifted) == coordinates


def test_result_is_source_bound_and_roundtrips_as_a_consumer_value() -> None:
    register = QubitRegister(qubit_ids=("left", "right"))
    check = _pauli(register, 0b0100)
    source = CheckSpaceValue(register=register, basis=(check,))
    value = logical_pauli_space(source)
    decoded = LogicalPauliSpace.model_validate_json(value.model_dump_json())

    assert decoded == value
    assert decoded.check_space.qubit_register == register
    assert (
        decoded.quotient_projection.source_axis
        == decoded.normalizer_embedding.source_axis
    )
    assert (
        decoded.induced_symplectic_form.row_axis
        == decoded.quotient_projection.target_axis
    )
    assert decoded.normalizer_embedding.target_axis.labels == (
        "X:left",
        "X:right",
        "Z:left",
        "Z:right",
    )


def test_catalog_manifest_publishes_the_issue_operation_and_example() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert tool.operation_id == "quantum.stabilizer.logical_pauli_space.compute"
    request = CheckSpaceValue.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, LogicalPauliSpace)
    assert result.logical_qubits == 1


def test_nonisotropic_checks_are_not_a_logical_quotient() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    source = CheckSpaceValue(
        register=register,
        basis=(_pauli(register, 0b01), _pauli(register, 0b10)),
    )
    with pytest.raises(OperationDomainValidationError, match="isotropic"):
        logical_pauli_space(source)


def test_maximum_register_and_check_row_boundary_remain_accepted() -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(32)))
    zero = _pauli(register, 0)
    source = CheckSpaceValue(register=register, basis=(zero,) * 64)
    result = logical_pauli_space(source)
    assert result.check_space.basis == ()
    assert result.logical_qubits == 32
    assert len(result.quotient_basis) == 64


def test_structural_roundtrip_rejects_a_foreign_register_claim() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    value = logical_pauli_space(CheckSpaceValue(register=register, basis=()))
    payload = value.model_dump(mode="json")
    payload["normalizer_embedding"]["target_axis"]["labels"][0] = "X:other"
    with pytest.raises(ValidationError, match="source-bound binary coordinate axes"):
        LogicalPauliSpace.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("inclusion", "embedded stabilizer inclusion"),
        ("embedding", "orthogonal to every check"),
        ("kernel", "stabilizer inclusion must lie in the quotient projection kernel"),
        ("section", "projection composed with its section must be identity"),
        ("form", "induced quotient form must equal ambient Pauli pairings"),
        ("degenerate_form", "induced quotient symplectic form must be nondegenerate"),
    ],
)
def test_consumer_validation_rejects_forged_semantic_relations(
    corruption: str, message: str
) -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    source = CheckSpaceValue(register=register, basis=(_pauli(register, 0b0100),))
    payload = logical_pauli_space(source).model_dump(mode="json")

    if corruption == "inclusion":
        payload["stabilizer_inclusion"]["matrix"]["entries"][0][0] = 1
    elif corruption == "embedding":
        # The selector rows stay intact; add X0 to an existing normalizer column.
        payload["normalizer_embedding"]["matrix"]["entries"][0][0] = 1
    elif corruption == "kernel":
        payload["quotient_projection"]["matrix"]["entries"][0][1] = 1
    elif corruption == "section":
        payload["quotient_lift"]["matrix"]["entries"][0][0] = 0
    elif corruption == "form":
        payload["induced_symplectic_form"]["entries"][0][0]["coordinates"] = ["1"]
    else:
        for row in payload["induced_symplectic_form"]["entries"]:
            for entry in row:
                entry["coordinates"] = ["0"]

    with pytest.raises(ValidationError, match=message):
        LogicalPauliSpace.model_validate_json(json.dumps(payload))
