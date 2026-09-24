"""Exact family commutation matrices with a retained Pauli axis."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quantum import (
    PauliFamilyCommutationRequest,
    PauliFamilyEntry,
    PhaseFreeQubitPauli,
    QubitRegister,
    pauli_family_commutation_matrix,
)


def _value(
    register: QubitRegister, x: tuple[int, ...], z: tuple[int, ...]
) -> PhaseFreeQubitPauli:
    return PhaseFreeQubitPauli(register=register, x_bits=x, z_bits=z)


def test_family_matrix_matches_independent_local_label_oracle() -> None:
    register = QubitRegister(qubit_ids=("left", "right"))
    family = (
        PauliFamilyEntry(pauli_id="x-left", pauli=_value(register, (1, 0), (0, 0))),
        PauliFamilyEntry(pauli_id="z-left", pauli=_value(register, (0, 0), (1, 0))),
        PauliFamilyEntry(pauli_id="y-right", pauli=_value(register, (0, 1), (0, 1))),
        PauliFamilyEntry(pauli_id="xz", pauli=_value(register, (1, 0), (0, 1))),
    )
    request = PauliFamilyCommutationRequest(family=family)
    result = pauli_family_commutation_matrix(request)

    # At one qubit, distinct nonidentity Pauli labels anticommute. Tensor
    # products commute iff the number of locally anticommuting factors is even.
    labels = (
        ((1, 0), (0, 0)),
        ((0, 1), (0, 0)),
        ((0, 0), (1, 1)),
        ((1, 0), (0, 1)),
    )
    expected = []
    for first in labels:
        expected_row = []
        for second in labels:
            odd = (
                sum(
                    (
                        first[q] != (0, 0)
                        and second[q] != (0, 0)
                        and first[q] != second[q]
                    )
                    for q in (0, 1)
                )
                % 2
            )
            expected_row.append(odd)
        expected.append(tuple(expected_row))
    assert result.source == request
    assert result.commutation_matrix == tuple(expected)
    assert result.commutation_matrix == tuple(
        zip(*result.commutation_matrix, strict=True)
    )
    assert all(result.commutation_matrix[i][i] == 0 for i in range(4))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_family_rejects_same_shape_pauli_from_another_ordered_register() -> None:
    first = QubitRegister(qubit_ids=("q0", "q1"))
    other = QubitRegister(qubit_ids=("q1", "q0"))
    request = PauliFamilyCommutationRequest.model_construct(
        family=(
            PauliFamilyEntry(pauli_id="a", pauli=_value(first, (1, 0), (0, 0))),
            PauliFamilyEntry(pauli_id="b", pauli=_value(other, (1, 0), (0, 0))),
        )
    )
    with pytest.raises(
        OperationDomainValidationError, match="identical ordered register"
    ):
        pauli_family_commutation_matrix(request)


def test_family_size_is_bounded_at_the_domain_value() -> None:
    register = QubitRegister(qubit_ids=("q",))
    row = _value(register, (0,), (0,))
    with pytest.raises(ValueError):
        PauliFamilyCommutationRequest(
            family=tuple(
                PauliFamilyEntry(pauli_id=f"p{i}", pauli=row) for i in range(65)
            )
        )


def test_aggregate_output_is_admitted_before_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.quantum.operations as operations

    register = QubitRegister(qubit_ids=tuple(f"q{i:02}" + "x" * 61 for i in range(32)))
    row = _value(register, (0,) * 32, (0,) * 32)
    request = PauliFamilyCommutationRequest(
        family=tuple(PauliFamilyEntry(pauli_id=f"p{i}", pauli=row) for i in range(64))
    )

    def unexpected_pairing(*args: object, **kwargs: object) -> int:
        raise AssertionError("pairing ran before aggregate result admission")

    monkeypatch.setattr(operations, "_symplectic_pairing", unexpected_pairing)
    with pytest.raises(OperationResourceAdmissionError, match="result size"):
        pauli_family_commutation_matrix(request)


def test_maximum_admitted_family_and_register_complete() -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(32)))
    row = _value(register, (0,) * 32, (0,) * 32)
    request = PauliFamilyCommutationRequest(
        family=tuple(PauliFamilyEntry(pauli_id=f"p{i}", pauli=row) for i in range(64))
    )
    result = pauli_family_commutation_matrix(request)
    assert len(result.commutation_matrix) == 64
    assert all(not any(row) for row in result.commutation_matrix)
