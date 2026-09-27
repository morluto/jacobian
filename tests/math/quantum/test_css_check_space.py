"""CSS orthogonality and conversion against independent binary enumeration."""

from __future__ import annotations

import json
from itertools import combinations, product

import pytest

from jacobian.math.quantum import (
    QubitRegister,
    css_check_space,
    css_exact_distance,
    css_logical_pauli_frame,
)
from jacobian.math.quantum._models import CSSNonOrthogonalWitness


def _span(rows: tuple[tuple[int, ...], ...], width: int) -> set[tuple[int, ...]]:
    """Enumerate the source row span independently of the kernel RREF."""
    return {
        tuple(
            sum(
                coefficient * row[column]
                for coefficient, row in zip(mask, rows, strict=True)
            )
            % 2
            for column in range(width)
        )
        for mask in product((0, 1), repeat=len(rows))
    }


def _subsets(rows: tuple[tuple[int, ...], ...]):
    for size in range(len(rows) + 1):
        yield from combinations(rows, size)


@pytest.mark.parametrize("n", (1, 2))
def test_all_binary_check_families_match_independent_css_oracle(n: int) -> None:
    rows = tuple(tuple(bits) for bits in product((0, 1), repeat=n))
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
    for x_checks in _subsets(rows):
        for z_checks in _subsets(rows):
            orthogonal = all(
                sum(x * z for x, z in zip(xrow, zrow, strict=True)) % 2 == 0
                for xrow in x_checks
                for zrow in z_checks
            )
            result = css_check_space(register, x_checks, z_checks)
            assert (result.css_check_space is not None) is orthogonal
            if not orthogonal:
                witness = result.witness
                assert witness is not None
                assert (
                    sum(
                        x * z
                        for x, z in zip(
                            x_checks[witness.x_row],
                            z_checks[witness.z_row],
                            strict=True,
                        )
                    )
                    % 2
                    == 1
                )
                continue
            assert result.witness is None
            css_value = result.css_check_space
            assert css_value is not None
            x_basis = tuple(row.x_bits for row in css_value.x_check_basis)
            z_basis = tuple(row.z_bits for row in css_value.z_check_basis)
            assert _span(x_basis, n) == _span(x_checks, n)
            assert _span(z_basis, n) == _span(z_checks, n)
            combined_rows = tuple(
                (*row.x_bits, *row.z_bits) for row in css_value.check_space.basis
            )
            expected = {
                (*x, *z) for x in _span(x_checks, n) for z in _span(z_checks, n)
            }
            assert _span(combined_rows, 2 * n) == expected
            distance = css_exact_distance(css_value)
            if distance.logical_qubits == 0:
                assert distance.x_distance is distance.z_distance is None
                continue
            x_stabilizers = _span(x_basis, n)
            z_stabilizers = _span(z_basis, n)

            def oracle_distance(
                checks: tuple[tuple[int, ...], ...],
                stabilizers: set[tuple[int, ...]],
            ) -> tuple[int, tuple[int, ...]]:
                logicals = tuple(
                    vector
                    for vector in product((0, 1), repeat=n)
                    if all(
                        sum(a * b for a, b in zip(vector, check, strict=True)) % 2 == 0
                        for check in checks
                    )
                    and vector not in stabilizers
                )
                _, vector = min(
                    (
                        (
                            sum(vector),
                            tuple(i for i, bit in enumerate(vector) if bit),
                        ),
                        vector,
                    )
                    for vector in logicals
                )
                return sum(vector), vector

            expected_x_distance, expected_x = oracle_distance(z_checks, x_stabilizers)
            expected_z_distance, expected_z = oracle_distance(x_checks, z_stabilizers)
            assert distance.x_distance == expected_x_distance
            assert distance.z_distance == expected_z_distance
            assert distance.x_representative is not None
            assert distance.z_representative is not None
            assert distance.x_representative.x_bits == expected_x
            assert distance.z_representative.z_bits == expected_z


def test_catalog_example_serializes_into_existing_stabilizer_consumers() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation
    from jacobian.math.quantum import (
        CheckSpaceValue,
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


def test_css_logical_frames_exhaust_small_quotients_independently() -> None:
    for n in (1, 2):
        rows = tuple(tuple(bits) for bits in product((0, 1), repeat=n))
        register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
        for x_checks in _subsets(rows):
            for z_checks in _subsets(rows):
                if any(
                    sum(x * z for x, z in zip(a, b, strict=True)) % 2
                    for a in x_checks
                    for b in z_checks
                ):
                    continue
                css = css_check_space(register, x_checks, z_checks).css_check_space
                assert css is not None
                frame = css_logical_pauli_frame(css)
                x_stabilizers = _span(tuple(row.x_bits for row in css.x_check_basis), n)
                z_stabilizers = _span(tuple(row.z_bits for row in css.z_check_basis), n)
                x_normalizer = {
                    vector
                    for vector in product((0, 1), repeat=n)
                    if all(
                        sum(a * b for a, b in zip(vector, row, strict=True)) % 2 == 0
                        for row in z_checks
                    )
                }
                z_normalizer = {
                    vector
                    for vector in product((0, 1), repeat=n)
                    if all(
                        sum(a * b for a, b in zip(vector, row, strict=True)) % 2 == 0
                        for row in x_checks
                    )
                }

                def quotient_classes(
                    normalizer: set[tuple[int, ...]], stabilizers: set[tuple[int, ...]]
                ) -> set[tuple[int, ...]]:
                    return {
                        min(
                            tuple(
                                (a + b) % 2
                                for a, b in zip(vector, stabilizer, strict=True)
                            )
                            for stabilizer in stabilizers
                        )
                        for vector in normalizer
                    }

                expected_x = quotient_classes(x_normalizer, x_stabilizers)
                expected_z = quotient_classes(z_normalizer, z_stabilizers)
                x_basis = tuple(row.x_bits for row in frame.x_logical_basis)
                z_basis = tuple(row.z_bits for row in frame.z_logical_basis)
                generated_x = {
                    tuple(
                        sum(
                            coefficient * row[i]
                            for coefficient, row in zip(mask, x_basis, strict=True)
                        )
                        % 2
                        for i in range(n)
                    )
                    for mask in product((0, 1), repeat=frame.logical_qubits)
                }
                generated_z = {
                    tuple(
                        sum(
                            coefficient * row[i]
                            for coefficient, row in zip(mask, z_basis, strict=True)
                        )
                        % 2
                        for i in range(n)
                    )
                    for mask in product((0, 1), repeat=frame.logical_qubits)
                }
                assert {
                    min(
                        tuple((a + b) % 2 for a, b in zip(v, s, strict=True))
                        for s in x_stabilizers
                    )
                    for v in generated_x
                } == expected_x
                assert {
                    min(
                        tuple((a + b) % 2 for a, b in zip(v, s, strict=True))
                        for s in z_stabilizers
                    )
                    for v in generated_z
                } == expected_z
                assert len(expected_x) == len(expected_z) == 2**frame.logical_qubits
                assert all(
                    sum(x * z for x, z in zip(xrow, zrow, strict=True)) % 2
                    == int(i == j)
                    for i, xrow in enumerate(x_basis)
                    for j, zrow in enumerate(z_basis)
                )


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


def test_css_distance_search_envelope_admits_nineteen_and_rejects_twenty() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.quantum import css_exact_distance

    for n, x_rank, z_rank, expect_distance in ((19, 9, 9, True), (20, 10, 9, False)):
        register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(n)))
        x_checks = tuple(tuple(int(i == j) for i in range(n)) for j in range(x_rank))
        z_checks = tuple(
            tuple(int(i == j) for i in range(n)) for j in range(x_rank, x_rank + z_rank)
        )
        css = css_check_space(register, x_checks, z_checks).css_check_space
        assert css is not None
        if expect_distance:
            result = css_exact_distance(css)
            assert result.logical_qubits == 1
            assert result.x_distance == result.z_distance == 1
        else:
            with pytest.raises(OperationResourceAdmissionError):
                css_exact_distance(css)


def test_css_distance_keeps_asymmetric_x_and_z_conventions() -> None:
    register = QubitRegister(qubit_ids=("q0", "q1", "q2"))
    result = css_check_space(
        register,
        ((1, 1, 0), (0, 1, 1)),
        (),
    )
    assert result.css_check_space is not None
    distance = css_exact_distance(result.css_check_space)
    assert distance.x_distance == 1
    assert distance.z_distance == 3
    assert distance.x_representative is not None
    assert distance.z_representative is not None
    assert distance.x_representative.x_bits == (1, 0, 0)
    assert distance.z_representative.z_bits == (1, 1, 1)


def test_logical_frame_rejects_forged_register_role_and_combined_space() -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.quantum import (
        CheckSpaceValue,
        CSSCheckSpaceValue,
        PhaseFreeQubitPauli,
        css_logical_pauli_frame,
    )

    register = QubitRegister(qubit_ids=("q",))
    foreign = QubitRegister(qubit_ids=("other",))
    x = PhaseFreeQubitPauli(register=register, x_bits=(1,), z_bits=(0,))
    z = PhaseFreeQubitPauli(register=register, x_bits=(0,), z_bits=(1,))
    empty = CheckSpaceValue(register=register, basis=())
    bad_values = (
        CSSCheckSpaceValue.model_construct(
            qubit_register=register,
            x_check_basis=(
                PhaseFreeQubitPauli(register=foreign, x_bits=(1,), z_bits=(0,)),
            ),
            z_check_basis=(),
            check_space=CheckSpaceValue(register=register, basis=(x,)),
        ),
        CSSCheckSpaceValue.model_construct(
            qubit_register=register,
            x_check_basis=(z,),
            z_check_basis=(),
            check_space=CheckSpaceValue(register=register, basis=(z,)),
        ),
        CSSCheckSpaceValue.model_construct(
            qubit_register=register,
            x_check_basis=(x,),
            z_check_basis=(),
            check_space=empty,
        ),
    )
    for bad_value in bad_values:
        with pytest.raises(OperationDomainValidationError):
            css_logical_pauli_frame(bad_value)


def test_logical_frame_rejects_constructed_combined_check_space() -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.quantum import (
        CheckSpaceValue,
        CSSCheckSpaceValue,
        css_logical_pauli_frame,
    )

    register = QubitRegister(qubit_ids=("q",))
    with pytest.raises(OperationDomainValidationError):
        css_logical_pauli_frame(
            CSSCheckSpaceValue.model_construct(
                qubit_register=register,
                x_check_basis=(),
                z_check_basis=(),
                check_space=CheckSpaceValue.model_construct(),
            )
        )


def test_css_and_logical_frame_accept_register_and_row_envelopes() -> None:
    register = QubitRegister(qubit_ids=tuple(f"q{i}" for i in range(32)))
    checks = tuple(tuple(int(i == j) for i in range(32)) for j in range(32))
    css = css_check_space(register, checks, ()).css_check_space
    assert css is not None
    frame = css_logical_pauli_frame(css)
    assert frame.logical_qubits == 0
    assert len(frame.css_check_space.check_space.basis) == 32

    one_qubit = QubitRegister(qubit_ids=("q",))
    at_row_limit = css_check_space(one_qubit, ((0,),) * 64, ())
    assert at_row_limit.css_check_space is not None
    assert len(at_row_limit.css_check_space.x_check_basis) == 0


def test_witness_deserialization_stays_structural_after_kernel_admission() -> None:
    register = QubitRegister(qubit_ids=("a", "b"))
    witness = css_check_space(register, ((1, 0),), ((1, 1),)).witness
    assert witness is not None
    # The kernel established the odd pairing once while selecting the
    # obstruction, and the retained rows carry that exact witness.
    assert sum(x * z for x, z in zip(witness.x_bits, witness.z_bits, strict=True)) % 2
    dumped = witness.model_dump(by_alias=True)
    flipped = {**dumped, "z_bits": [bit ^ 1 for bit in dumped["z_bits"]]}
    assert (
        sum(x * z for x, z in zip(flipped["x_bits"], flipped["z_bits"], strict=True))
        % 2
        == 0
    )
    # Transport must not recompute the GF(2) inner product: a structurally
    # valid row pair deserializes without a mathematical replay.
    assert CSSNonOrthogonalWitness.model_validate(flipped).x_bits == witness.x_bits


def test_witness_retains_its_ordered_register_context() -> None:
    first = css_check_space(
        QubitRegister(qubit_ids=("q0", "q1")), ((1, 0),), ((1, 1),)
    ).witness
    second = css_check_space(
        QubitRegister(qubit_ids=("q1", "q0")), ((1, 0),), ((1, 1),)
    ).witness
    assert first is not None and second is not None
    assert first.qubit_register == QubitRegister(qubit_ids=("q0", "q1"))
    first_json = first.model_dump(mode="json", by_alias=True)
    assert first_json != second.model_dump(mode="json", by_alias=True)
    restored = CSSNonOrthogonalWitness.model_validate_json(json.dumps(first_json))
    assert restored == first
    # Persisted coordinates map back onto the ordered qubit IDs.
    assert restored.qubit_register.qubit_ids[0] == "q0"
    assert restored.x_bits[0] == 1
