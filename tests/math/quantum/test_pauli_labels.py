"""Pauli label conversions replayed against tiny dense operators."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.quantum import (
    ExactQubitPauli,
    PauliFromLabelsRequest,
    PauliToLabelsRequest,
    PhaseFreeQubitPauli,
    QubitRegister,
    pauli_from_labels,
    pauli_to_labels,
)

_I = ((1, 0), (0, 1))
_X = ((0, 1), (1, 0))
_Y = ((0, -1j), (1j, 0))
_Z = ((1, 0), (0, -1))
_LOCAL = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _multiply(left, right):
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(len(right)))
            for j in range(len(right))
        )
        for i in range(len(left))
    )


def _tensor(left, right):
    return tuple(
        tuple(
            left[i][j] * right[k][ell]
            for j in range(len(left[0]))
            for ell in range(len(right[0]))
        )
        for i in range(len(left))
        for k in range(len(right))
    )


def _operator(labels: tuple[str, ...], phase: int):
    matrix = ((1 + 0j,),)
    for label in labels:
        matrix = _tensor(matrix, _LOCAL[label])
    return tuple(tuple((1j**phase) * entry for entry in row) for row in matrix)


def _stored_operator(pauli: ExactQubitPauli):
    base = ((1 + 0j,),)
    for x, z in zip(pauli.phase_free.x_bits, pauli.phase_free.z_bits, strict=True):
        local = _multiply(_X if x else _I, _Z if z else _I)
        base = _tensor(base, local)
    return tuple(tuple((1j**pauli.phase) * entry for entry in row) for row in base)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_label_round_trips_match_independent_dense_operator(width: int) -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(width)))
    for labels in product(("I", "X", "Y", "Z"), repeat=width):
        for scalar_phase in range(4):
            request = PauliFromLabelsRequest(
                register=register, labels=labels, phase=scalar_phase
            )
            converted = pauli_from_labels(request)
            y_count = labels.count("Y")
            expected_x = tuple(int(label in ("X", "Y")) for label in labels)
            expected_z = tuple(int(label in ("Z", "Y")) for label in labels)
            assert converted.pauli.phase_free.x_bits == expected_x
            assert converted.pauli.phase_free.z_bits == expected_z
            assert converted.pauli.phase == (scalar_phase + y_count) % 4
            assert _stored_operator(converted.pauli) == _operator(labels, scalar_phase)

            decoded = pauli_to_labels(PauliToLabelsRequest(pauli=converted.pauli))
            assert decoded.labels == labels
            assert decoded.phase == scalar_phase
            assert _operator(decoded.labels, decoded.phase) == _operator(
                labels, scalar_phase
            )


def test_label_conversion_catalog_round_trip_preserves_y_scalar() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    from_labels = catalog.operation("quantum.pauli.qubit.from_labels.compute")
    to_labels = catalog.operation("quantum.pauli.qubit.to_labels.compute")
    assert from_labels is not None and to_labels is not None
    constructed = invoke_operation(
        from_labels.operation_id,
        {"register": {"qubit_ids": ["a", "b"]}, "labels": ["Y", "Z"], "phase": 2},
        catalog,
    )
    decoded = invoke_operation(
        to_labels.operation_id, {"pauli": constructed.output["pauli"]}, catalog
    ).output
    assert decoded["labels"] == ["Y", "Z"]
    assert decoded["phase"] == 2


def test_labels_must_cover_register_and_use_named_pauli_symbols() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    with pytest.raises(ValidationError):
        PauliFromLabelsRequest(register=register, labels=("X",))
    with pytest.raises(ValidationError):
        PauliFromLabelsRequest(register=register, labels=("X", "A"))


def test_scalar_phase_encoding_is_not_reinterpreted_as_y_phase() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    pauli = ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(1,)),
        phase=3,
    )
    decoded = pauli_to_labels(PauliToLabelsRequest(pauli=pauli))
    assert decoded.labels == ("Y",)
    assert decoded.phase == 2
    assert decoded.phase == (pauli.phase - 1) % 4
