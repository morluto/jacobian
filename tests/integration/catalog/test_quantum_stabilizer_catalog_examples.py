"""Stabilizer and Pauli catalog evidence, including consumer composition."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.quantum import (
    CheckSpaceValue,
    ExactQubitPauli,
    LogicalPauliFrame,
    PhaseFreeQubitPauli,
    QubitRegister,
    css_exact_distance,
    css_logical_pauli_frame,
)
from jacobian.math.quantum._models import ExactStabilizerGroup


def _pauli(
    register: QubitRegister,
    x_bits: tuple[int, ...],
    z_bits: tuple[int, ...],
    phase: int,
) -> ExactQubitPauli:
    return ExactQubitPauli(
        phase_free=PhaseFreeQubitPauli(register=register, x_bits=x_bits, z_bits=z_bits),
        phase=phase,
    )


def test_catalog_example_serializes_into_existing_stabilizer_consumers() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.quantum import (
        PhaseFreeQubitPauli,
        stabilizer_error_equivalence,
        stabilizer_syndrome,
    )

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.css_check_space.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["witness"] is None
    from jacobian.math.quantum import CSSCheckSpaceValue

    css_value = CSSCheckSpaceValue.model_validate(result.output["css_check_space"])
    check_space = CheckSpaceValue.model_validate(css_value.check_space.model_dump())
    error = PhaseFreeQubitPauli(
        register=check_space.qubit_register,
        x_bits=(1, 0, 0),
        z_bits=(0, 0, 0),
    )
    assert stabilizer_syndrome(check_space, error).syndrome == (0, 1)
    assert stabilizer_error_equivalence(
        check_space, error, error
    ).equivalent_mod_stabilizers


def test_catalog_css_logical_frame_and_steane_parameter_fixture() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.quantum import (
        CSSCheckSpaceValue,
        CSSLogicalPauliFrame,
        css_check_space,
    )

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.css_logical_frame.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    frame = CSSLogicalPauliFrame.model_validate(result.output)
    assert frame.logical_qubits == 2
    distance_operation = catalog.operation("quantum.stabilizer.css_distance.compute")
    assert distance_operation is not None
    distance_result = invoke_operation(
        distance_operation.operation_id,
        distance_operation.examples[0].input,
        catalog,
    )
    from jacobian.math.quantum import CSSDistanceResult

    distance = CSSDistanceResult.model_validate(distance_result.output)
    assert distance.logical_qubits == 2
    assert distance.x_distance == distance.z_distance == 2

    # The standard seven-qubit Hamming CSS checks have one logical qubit.
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(7)))
    hamming_rows = (
        (1, 0, 1, 0, 1, 0, 1),
        (0, 1, 1, 0, 0, 1, 1),
        (0, 0, 0, 1, 1, 1, 1),
    )
    css = css_check_space(register, hamming_rows, hamming_rows).css_check_space
    assert css is not None
    steane_frame = css_logical_pauli_frame(
        CSSCheckSpaceValue.model_validate(css.model_dump())
    )
    assert steane_frame.logical_qubits == 1
    assert (
        sum(
            x * z
            for x, z in zip(
                steane_frame.x_logical_basis[0].x_bits,
                steane_frame.z_logical_basis[0].z_bits,
                strict=True,
            )
        )
        % 2
        == 1
    )
    distance = css_exact_distance(steane_frame.css_check_space)
    assert distance.x_distance == distance.z_distance == 3
    assert distance.x_representative is not None
    assert distance.z_representative is not None
    assert distance.x_representative.weight == distance.z_representative.weight == 3


def test_exact_group_is_available_through_catalog_dispatch() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.exact_group.from_generators.compute"
    assert catalog.operation(operation_id) is not None
    operation_result = invoke_operation(
        operation_id,
        {
            "register": {"qubit_ids": ["q"]},
            "generators": [
                {
                    "phase_free": {
                        "register": {"qubit_ids": ["q"]},
                        "x_bits": [0],
                        "z_bits": [1],
                    },
                    "phase": 0,
                }
            ],
        },
        catalog,
    )
    result = ExactStabilizerGroup.model_validate(operation_result.output)
    assert result.generators == (
        _pauli(QubitRegister(qubit_ids=("q",)), (0,), (1,), 0),
    )


def test_catalog_publishes_generic_mixed_pauli_logical_frame() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.logical_frame.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    frame = LogicalPauliFrame.model_validate(result.output)
    assert frame.logical_qubits == 1
    assert frame.x_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert frame.z_logical_basis[0].qubit_register == frame.check_space.qubit_register
    assert LogicalPauliFrame.model_validate(frame.model_dump(mode="json")) == frame

    # Deserialization is structural only: the kernel establishes S-perp
    # membership and canonical pairings once, so model_validate must not replay
    # that GF(2) work. Structural violations are still rejected here.
    bad_register = frame.model_dump(mode="json")
    bad_register["x_logical_basis"][0]["qubit_register"] = {"qubit_ids": ["other"]}
    with pytest.raises(ValidationError, match="register"):
        LogicalPauliFrame.model_validate(bad_register)

    bad_dimension = frame.model_dump(mode="json")
    bad_dimension["x_logical_basis"] = []
    with pytest.raises(ValidationError, match="size k"):
        LogicalPauliFrame.model_validate(bad_dimension)


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


def test_code_value_is_published_and_roundtrips_through_catalog() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation_id = "quantum.stabilizer.code.compute"
    assert catalog.operation(operation_id) is not None
    raw = invoke_operation(
        operation_id,
        {
            "group": {
                "register": {"qubit_ids": ["q"]},
                "generators": [
                    {
                        "phase_free": {
                            "register": {"qubit_ids": ["q"]},
                            "x_bits": [0],
                            "z_bits": [1],
                        },
                        "phase": 0,
                    }
                ],
            },
            "generator_eigenvalues": [-1],
        },
        catalog,
    )
    from jacobian.math.quantum._models import StabilizerCodeValue

    value = StabilizerCodeValue.model_validate(raw.output)
    assert value.logical_qubits == 0
    assert value.group.generators[0].phase == 2


def test_catalog_example_runs_and_returns_source_bound_result() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    catalog = Catalog.open()
    operation = catalog.operation("quantum.stabilizer.distance.compute")
    assert operation is not None
    output = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    ).output
    assert output["logical_qubits"] == 1
    assert output["distance"] == 1
    assert (
        output["representative"]["qubit_register"]
        == output["check_space"]["qubit_register"]
    )
