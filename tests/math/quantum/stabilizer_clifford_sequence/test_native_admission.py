"""Native admission must classify forged typed values, not raise on them."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    PhaseFreeQubitPauli,
    QubitRegister,
)
from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    CliffordGate,
    StabilizerCliffordSequence,
)
from jacobian.math.quantum.stabilizer_clifford_sequence.operations import (
    _admit_register,
    _admitted_pauli_bits,
    apply_stabilizer_clifford_sequence,
)


def _register() -> QubitRegister:
    return QubitRegister(qubit_ids=("q0", "q1"))


def _pauli() -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(
            register=_register(), x_bits=(0, 1), z_bits=(1, 0)
        ),
        phase=1,
    )


def test_a_forged_register_is_refused_rather_than_raising_attribute_error() -> None:
    """A native caller can build a typed register without its fields.

    ``isinstance`` succeeds for a ``model_construct``-ed register, so reading
    ``qubit_ids`` directly raised ``AttributeError`` from inside the admission
    path instead of the declared ``OperationDomainValidationError``.
    """
    forged = QubitRegister.model_construct()
    with pytest.raises(OperationDomainValidationError) as refusal:
        _admit_register(forged, "sequence.register")
    assert (
        refusal.value.errors()[0]["type"]
        == "quantum.stabilizer_clifford_sequence.invalid_register"
    )
    # Negative control: a real register is still admitted.
    assert _admit_register(_register(), "sequence.register") is not None


def test_a_forged_pauli_missing_its_bits_is_refused_not_raised() -> None:
    """Missing Pauli coordinates are a malformed value, not an implementation bug.

    ``_admitted_pauli_bits`` read ``x_bits`` and ``z_bits`` directly, so a
    ``model_construct``-ed ``PhaseFreeQubitPauli`` that kept its register but
    omitted the coordinates escaped as ``AttributeError`` after the type,
    parent, and phase checks all passed.
    """
    forged_phase_free = PhaseFreeQubitPauli.model_construct(register=_register())
    assert _admitted_pauli_bits(forged_phase_free, 2) is False
    # Negative control: a canonical Pauli is still admitted.
    assert _admitted_pauli_bits(_pauli().phase_free, 2) is True

    forged = ExactQubitPauli.model_construct(phase_free=forged_phase_free, phase=1)
    group = ExactStabilizerGroup.model_construct(
        qubit_register=_register(), generators=(forged,)
    )
    sequence = StabilizerCliffordSequence(
        register=_register(), gates=(CliffordGate(gate="H", qubits=("q0",)),)
    )
    with pytest.raises(OperationDomainValidationError):
        apply_stabilizer_clifford_sequence(group, sequence)
