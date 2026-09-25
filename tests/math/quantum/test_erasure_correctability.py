"""Exact erasure criteria checked against exhaustive small Pauli spaces."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.math.quantum import (
    CheckSpaceValue,
    PhaseFreeQubitPauli,
    QubitRegister,
    StabilizerErasureCorrectabilityRequest,
    StabilizerErasureCorrectabilityResult,
    stabilizer_erasure_correctability,
)


def _span(rows: tuple[tuple[int, ...], ...], width: int) -> set[tuple[int, ...]]:
    return {
        tuple(
            sum(
                coefficient * row[column]
                for coefficient, row in zip(coefficients, rows, strict=True)
            )
            % 2
            for column in range(width)
        )
        for coefficients in product((0, 1), repeat=len(rows))
    }


def _oracle(rows: tuple[tuple[int, ...], ...], n: int, erased: tuple[int, ...]):
    stabilizers = _span(rows, 2 * n)
    supported_normalizer: list[tuple[int, ...]] = []
    supported_stabilizers: list[tuple[int, ...]] = []
    logical_witnesses: list[tuple[int, ...]] = []
    for local in product((0, 1, 2, 3), repeat=len(erased)):
        flat = [0] * (2 * n)
        for position, label in zip(erased, local, strict=True):
            flat[position] = label & 1
            flat[n + position] = (label >> 1) & 1
        vector = tuple(flat)
        if any(
            sum(vector[q] * row[n + q] + vector[n + q] * row[q] for q in range(n)) % 2
            for row in rows
        ):
            continue
        supported_normalizer.append(vector)
        if vector in stabilizers:
            supported_stabilizers.append(vector)
        elif any(vector):
            logical_witnesses.append(vector)
    return (
        len(supported_normalizer).bit_length() - 1,
        len(supported_stabilizers).bit_length() - 1,
        logical_witnesses,
    )


def _value(n: int, rows: tuple[tuple[int, ...], ...]) -> CheckSpaceValue:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
    basis = tuple(
        PhaseFreeQubitPauli(
            register=register,
            x_bits=row[:n],
            z_bits=row[n:],
        )
        for row in rows
    )
    return CheckSpaceValue(register=register, basis=basis)


@pytest.mark.parametrize(
    ("n", "rows"),
    (
        (2, ()),
        (2, ((1, 1, 0, 0), (0, 0, 1, 1))),  # Bell stabilizer, k=0.
        (3, ((1, 1, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0))),  # Bit-flip repetition.
        (3, ((1, 0, 0, 1, 0, 0),)),  # A mixed logical quotient.
    ),
)
def test_erasure_profile_matches_exhaustive_pauli_oracle(n, rows) -> None:
    value = _value(n, rows)
    for mask in range(1 << n):
        erased_positions = tuple(index for index in range(n) if mask & (1 << index))
        erased_ids = tuple(f"q{index}" for index in erased_positions)
        oracle_normalizer, oracle_stabilizer, logicals = _oracle(
            rows, n, erased_positions
        )
        result = stabilizer_erasure_correctability(
            StabilizerErasureCorrectabilityRequest(
                check_space=value, erased_qubit_ids=erased_ids
            )
        )
        assert result.supported_normalizer_dimension == oracle_normalizer
        assert result.supported_stabilizer_dimension == oracle_stabilizer
        assert (
            result.supported_logical_dimension == oracle_normalizer - oracle_stabilizer
        )
        assert result.correctable is (not logicals)
        if logicals:
            assert result.witness is not None
            encoded = tuple(
                x + 2 * z
                for x, z in zip(
                    result.witness.x_bits, result.witness.z_bits, strict=True
                )
            )
            assert encoded in {
                tuple(flat[q] + 2 * flat[n + q] for q in range(n)) for flat in logicals
            }
        else:
            assert result.witness is None
        assert (
            StabilizerErasureCorrectabilityResult.model_validate_json(
                result.model_dump_json()
            )
            == result
        )


def test_erasure_request_rejects_duplicate_or_foreign_axes() -> None:
    value = _value(2, ())
    with pytest.raises(ValueError):
        StabilizerErasureCorrectabilityRequest(
            check_space=value, erased_qubit_ids=("q0", "q0")
        )
    with pytest.raises(ValueError):
        StabilizerErasureCorrectabilityRequest(
            check_space=value, erased_qubit_ids=("other",)
        )


def test_nonisotropic_check_space_is_rejected() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    x = PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,))
    z = PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(1,))
    with pytest.raises(ValueError, match="isotropic"):
        stabilizer_erasure_correctability(
            StabilizerErasureCorrectabilityRequest(
                check_space=CheckSpaceValue(register=register, basis=(x, z)),
                erased_qubit_ids=("q0",),
            )
        )
