from __future__ import annotations

import pytest
import sympy as sp

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum._models import (
    ExactQubitPauli,
    ExactStabilizerGroup,
    PhaseFreeQubitPauli,
    QubitRegister,
    StabilizerCodeRequest,
    StabilizerStatePauliMeasurementRequest,
    StabilizerStatePauliMeasurementResult,
)
from jacobian.math.quantum._tools import TOOLS
from jacobian.math.quantum.operations import (
    stabilizer_code_compute,
    stabilizer_state_measure_pauli,
)


def _pauli(
    register: QubitRegister,
    x: tuple[int, ...],
    z: tuple[int, ...],
    phase: int = 0,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z),
        phase=phase,
    )


def _state(*generators: ExactQubitPauli):
    register = generators[0].register
    return stabilizer_code_compute(
        StabilizerCodeRequest(
            group=ExactStabilizerGroup(register=register, generators=generators),
            generator_eigenvalues=(1,) * len(generators),
        )
    )


def _pauli_matrix(pauli: ExactQubitPauli) -> sp.Matrix:
    """Independent exact computational-basis representation of the Pauli."""
    n = len(pauli.register.qubit_ids)
    dimension = 1 << n
    matrix = sp.zeros(dimension, dimension)
    flip_mask = sum(
        bit << (n - index - 1) for index, bit in enumerate(pauli.phase_free.x_bits)
    )
    for basis in range(dimension):
        z_parity = sum(
            bit * ((basis >> (n - index - 1)) & 1)
            for index, bit in enumerate(pauli.phase_free.z_bits)
        )
        matrix[basis ^ flip_mask, basis] = sp.I**pauli.phase * (-1) ** z_parity
    return matrix


def _state_projector(state) -> sp.Matrix:
    n = len(state.group.register.qubit_ids)
    identity = sp.eye(1 << n)
    projector = identity
    for generator in state.group.generators:
        projector = projector * (identity + _pauli_matrix(generator)) / 2
    return projector


def _assert_branch_projectors(result, source_state) -> None:
    source_projector = _state_projector(source_state)
    observable_matrix = _pauli_matrix(result.observable)
    for branch in (result.positive_branch, result.negative_branch):
        assert branch is not None
        spectral_projector = (
            sp.eye(source_projector.rows) + branch.outcome * observable_matrix
        ) / 2
        assert (
            spectral_projector * source_projector * spectral_projector
            == sp.Rational(1, 2) * _state_projector(branch.state)
        )


def test_measurement_is_publicly_declared_with_exact_json_result() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "quantum.stabilizer_state.measure_pauli.compute"
    )
    assert tool.request_type is StabilizerStatePauliMeasurementRequest
    register = QubitRegister(qubit_ids=("q0",))
    source = _state(_pauli(register, (0,), (1,)))
    request = StabilizerStatePauliMeasurementRequest(
        state=source,
        observable=_pauli(register, (1,), (0,)),
    )
    result = tool.run(request)
    decoded = StabilizerStatePauliMeasurementResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    _assert_branch_projectors(decoded, source)


@pytest.mark.parametrize(("observable_phase", "outcome"), [(0, 1), (2, -1)])
def test_commuting_measurement_returns_signed_deterministic_relation(
    observable_phase: int, outcome: int
) -> None:
    register = QubitRegister(qubit_ids=("q0",))
    source = _state(_pauli(register, (0,), (1,)))
    observable = _pauli(register, (0,), (1,), phase=observable_phase)
    result = stabilizer_state_measure_pauli(
        StabilizerStatePauliMeasurementRequest(state=source, observable=observable)
    )

    assert result.status == "DETERMINISTIC"
    assert result.deterministic_outcome == outcome
    assert result.relation_generator_bits == (1,)
    assert result.relation_phase == observable_phase
    assert result.deterministic_state == result.source_state
    assert _pauli_matrix(observable) * _state_projector(source) == (
        outcome * _state_projector(source)
    )


def test_measurement_on_bell_state_matches_exact_projector_oracle() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1"))
    source = _state(
        _pauli(register, (1, 1), (0, 0)),
        _pauli(register, (0, 0), (1, 1)),
    )
    result = stabilizer_state_measure_pauli(
        StabilizerStatePauliMeasurementRequest(
            state=source,
            observable=_pauli(register, (0, 0), (1, 0)),
        )
    )

    assert result.status == "UNIFORM_BINARY"
    assert result.positive_branch is not None
    assert result.negative_branch is not None
    _assert_branch_projectors(result, source)


def test_measurement_rejects_nonhermitian_observable() -> None:
    register = QubitRegister(qubit_ids=("q0",))
    with pytest.raises(OperationDomainValidationError):
        stabilizer_state_measure_pauli(
            StabilizerStatePauliMeasurementRequest(
                state=_state(_pauli(register, (0,), (1,))),
                observable=_pauli(register, (1,), (0,), phase=1),
            )
        )
