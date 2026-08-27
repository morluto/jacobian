"""Exact contracts for labelled subsystem Hermitian matrix operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import product
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices import subsystems
from jacobian.math.matrices.subsystems._models import (
    MAX_KRONECKER_RESULT_COMPONENT_DIGITS,
    MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS,
    MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS,
    PsdOrderRequest,
    PsdOrderResult,
    SubsystemKroneckerProductRequest,
    SubsystemPartialTraceRequest,
    SubsystemPartialTraceResult,
)
from jacobian.math.matrices.subsystems._operations import (
    _verify_partial_trace_result,
    _verify_psd_order_result,
    compute_kronecker_product,
    compute_partial_trace,
    decide_psd_order,
)
from jacobian.math.matrices.subsystems.operations import (
    kronecker_product,
    partial_trace,
    psd_order,
)
from jacobian.math.matrices.subsystems.values import (
    FactorizedHermitianMatrix,
    MatrixSubsystem,
    partial_trace_measured_entries,
)
from jacobian.math.matrices.values import RationalMatrix


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _matrix(
    entries: list[list[Fraction | int]],
    factors: tuple[MatrixSubsystem, ...],
) -> FactorizedHermitianMatrix:
    return FactorizedHermitianMatrix(
        matrix=RationalMatrix(
            entries=tuple(
                tuple(CanonicalRational.from_fraction(Fraction(value)) for value in row)
                for row in entries
            )
        ),
        factors=factors,
    )


def _entries(matrix: FactorizedHermitianMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.matrix.entries
    )


def test_subsystem_native_public_api_is_explicit() -> None:
    assert tuple(subsystems.__all__) == (
        "FactorizedHermitianMatrix",
        "MatrixSubsystem",
        "kronecker_product",
        "partial_trace",
        "psd_order",
    )


def test_factorized_matrix_rejects_shape_symmetry_and_duplicate_labels() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    with pytest.raises(ValidationError):
        _matrix([[1]], (q,))
    with pytest.raises(ValidationError):
        _matrix([[1, 2], [3, 4]], (q,))
    with pytest.raises(ValidationError):
        _matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            (q, MatrixSubsystem(label="q", dimension=2)),
        )


def test_axis_bound_kronecker_product_concatenates_named_factors() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left = _matrix([[1, 0], [0, 2]], (q,))
    right = _matrix([[3, 0], [0, 4]], (r,))

    native = kronecker_product(left, right)
    wire = compute_kronecker_product(
        SubsystemKroneckerProductRequest(left=left, right=right)
    ).product
    assert native == wire
    assert native.factors == (q, r)
    assert _entries(native) == (
        (Fraction(3), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(6), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(8)),
    )
    reduced = compute_partial_trace(
        SubsystemPartialTraceRequest.model_validate(
            {
                "matrix": wire.model_dump(mode="json"),
                "traced_factor_labels": ["q"],
            }
        )
    ).reduced_matrix
    assert reduced.factors == (r,)
    assert _entries(reduced) == (
        (Fraction(9), Fraction(0)),
        (Fraction(0), Fraction(12)),
    )

    with pytest.raises(ValidationError):
        SubsystemKroneckerProductRequest(left=left, right=left)


def test_partial_trace_binds_the_yz_linearization_canary_to_factor_labels() -> None:
    y = MatrixSubsystem(label="Y", dimension=2)
    z = MatrixSubsystem(label="Z", dimension=2)
    diagonal: list[list[Fraction | int]] = [
        [Fraction(1, 5), 0, 0, 0],
        [0, Fraction(-1, 10), 0, 0],
        [0, 0, Fraction(3, 10), 0],
        [0, 0, 0, Fraction(-1, 5)],
    ]
    source = _matrix(diagonal, (y, z))
    reduced = partial_trace(source, ("Y",))
    wire = compute_partial_trace(
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("Y",))
    )

    assert reduced == wire.reduced_matrix
    assert reduced.factors == (z,)
    assert _entries(reduced) == (
        (Fraction(1, 2), Fraction(0)),
        (Fraction(0), Fraction(-3, 10)),
    )
    forged = wire.model_dump(mode="python")
    forged["reduced_matrix"] = _matrix([[1, 0], [0, 1]], (z,))
    assert not _verify_partial_trace_result(wire.__class__.model_validate(forged))

    differently_bound = _matrix(diagonal, (z, y))
    wrong = partial_trace(differently_bound, ("Y",))
    assert _entries(wrong) == (
        (Fraction(1, 10), Fraction(0)),
        (Fraction(0), Fraction(1, 10)),
    )
    assert wrong != reduced


def test_partial_trace_commutes_over_disjoint_named_factors_and_retains_scalar_context() -> (
    None
):
    a = MatrixSubsystem(label="a", dimension=2)
    b = MatrixSubsystem(label="b", dimension=2)
    c = MatrixSubsystem(label="c", dimension=2)
    source = _matrix(
        [[index + 1 if row == index else 0 for index in range(8)] for row in range(8)],
        (a, b, c),
    )
    combined = partial_trace(source, ("a", "c"))
    sequential = partial_trace(partial_trace(source, ("a",)), ("c",))
    assert combined == sequential
    assert combined.factors == (b,)

    total = partial_trace(source, ("a", "b", "c"))
    assert total.factors == ()
    assert _entries(total) == ((Fraction(36),),)


def test_partial_trace_rejects_unknown_and_repeated_factor_labels() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    source = _matrix([[1, 0], [0, 2]], (q,))
    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("r",))
    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("q", "q"))


def test_partial_trace_result_round_trips_structurally_and_verifies_forged_claims() -> (
    None
):
    q = MatrixSubsystem(label="q", dimension=2)
    base = compute_partial_trace(
        SubsystemPartialTraceRequest(
            matrix=_matrix([[1, 0], [0, 2]], (q,)), traced_factor_labels=("q",)
        )
    )
    assert (
        SubsystemPartialTraceResult.model_validate(base.model_dump(mode="python"))
        == base
    )

    forged = base.model_dump(mode="python")
    forged["reduced_matrix"] = _matrix([[1]], ())
    assert not _verify_partial_trace_result(
        SubsystemPartialTraceResult.model_validate(forged)
    )

    beyond_envelope = _matrix(
        [
            [Fraction(1, 10**4098 + 1), 0],
            [0, Fraction(1, 10**4098 + 1)],
        ],
        (q,),
    )
    forged = base.model_dump(mode="python")
    forged["source_matrix"] = beyond_envelope
    assert not _verify_partial_trace_result(
        SubsystemPartialTraceResult.model_validate(forged)
    )


def test_partial_trace_boundary_result_components_round_trip() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    boundary_denominator = 10**4097 + 9
    over_denominator = 10**4098 + 1
    boundary_source = _matrix(
        [
            [Fraction(1, boundary_denominator), 0],
            [0, Fraction(1, boundary_denominator)],
        ],
        (q,),
    )
    over_source = _matrix(
        [[Fraction(1, over_denominator), 0], [0, Fraction(1, over_denominator)]],
        (q,),
    )

    reduced = partial_trace(boundary_source, ("q",))
    assert len(reduced.matrix.entries[0][0].den) == 4098
    wire = compute_partial_trace(
        SubsystemPartialTraceRequest(
            matrix=boundary_source, traced_factor_labels=("q",)
        )
    )
    assert (
        SubsystemPartialTraceResult.model_validate(wire.model_dump(mode="python"))
        == wire
    )
    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(matrix=over_source, traced_factor_labels=("q",))


def test_kronecker_product_rejects_structural_bounds_before_operand_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.matrices.subsystems import _models

    heavy = Fraction(1, 10**300 + 9)

    def dense(order: int, factor: MatrixSubsystem) -> FactorizedHermitianMatrix:
        return _matrix(
            [
                [heavy if row == column else 0 for column in range(order)]
                for row in range(order)
            ],
            (factor,),
        )

    converted: list[object] = []
    real_entry_fractions = _models._entry_fractions

    def counted(matrix: object) -> object:
        converted.append(matrix)
        return real_entry_fractions(matrix)  # type: ignore[arg-type]

    monkeypatch.setattr(_models, "_entry_fractions", counted)

    wide_left = MatrixSubsystem(label="wide-left", dimension=5)
    wide_right = MatrixSubsystem(label="wide-right", dimension=4)
    with pytest.raises(ValidationError):
        SubsystemKroneckerProductRequest(
            left=dense(5, wide_left),
            right=dense(4, wide_right),
        )

    first = MatrixSubsystem(label="first", dimension=2)
    second = MatrixSubsystem(label="second", dimension=2)
    third = MatrixSubsystem(label="third", dimension=1)
    fourth = MatrixSubsystem(label="fourth", dimension=1)
    fifth = MatrixSubsystem(label="fifth", dimension=1)
    crowded_left = _matrix(
        [[heavy if row == column else 0 for column in range(4)] for row in range(4)],
        (first, second),
    )
    with pytest.raises(ValidationError):
        SubsystemKroneckerProductRequest(
            left=crowded_left,
            right=_matrix([[heavy]], (third, fourth, fifth)),
        )
    assert converted == []


def test_kronecker_product_stops_the_digit_scan_after_the_first_excess_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.matrices.subsystems import _models

    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    heavy = Fraction(1, 10**300 + 9)
    scanned: list[int] = []
    real_digits = _models._fraction_component_digits

    def counted(value: Fraction) -> tuple[int, int]:
        digits = real_digits(value)
        scanned.append(max(digits))
        return digits

    monkeypatch.setattr(_models, "_fraction_component_digits", counted)
    with pytest.raises(ValidationError):
        SubsystemKroneckerProductRequest(
            left=_matrix([[heavy, 0], [0, heavy]], (q,)),
            right=_matrix([[1, 0], [0, 1]], (r,)),
        )
    assert scanned == [301]


def test_partial_trace_rejects_genuinely_growing_contractions() -> None:
    factors = tuple(MatrixSubsystem(label=label, dimension=2) for label in "pqrs")
    source = _matrix(
        [
            [
                Fraction(1, 10 ** (1030 + row) + row) if row == column else 0
                for column in range(16)
            ]
            for row in range(16)
        ],
        factors,
    )

    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(
            matrix=source,
            traced_factor_labels=("p", "q", "r", "s"),
        )


def test_partial_trace_work_envelope_admits_folded_boundary_terms() -> None:
    q = MatrixSubsystem(label="q", dimension=4)
    admitted_denominator = 10**4095 + 3
    source = _matrix(
        [
            [
                Fraction(1, admitted_denominator) if row == column else 0
                for column in range(4)
            ]
            for row in range(4)
        ],
        (q,),
    )

    reduced = partial_trace(source, ("q",))
    assert _entries(reduced) == ((Fraction(4, admitted_denominator),),)


def test_partial_trace_readmits_its_emitted_shared_denominator_factor() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=8)
    shared_denominator = 10**4094 + 9
    source = _matrix(
        [
            [
                Fraction(1, shared_denominator) if row == column else 0
                for column in range(16)
            ]
            for row in range(16)
        ],
        (q, r),
    )

    reduced = partial_trace(source, ("q",))
    assert reduced.factors == (r,)
    assert _entries(reduced) == tuple(
        tuple(
            Fraction(2, shared_denominator) if row == column else Fraction(0)
            for column in range(8)
        )
        for row in range(8)
    )
    stepwise = partial_trace(reduced, ("r",))
    combined = partial_trace(source, ("q", "r"))
    assert stepwise == combined
    assert _entries(stepwise) == ((Fraction(16, shared_denominator),),)


def test_partial_trace_work_charges_only_contracted_terms() -> None:
    y = MatrixSubsystem(label="y", dimension=2)
    z = MatrixSubsystem(label="z", dimension=2)
    uncontracted = Fraction(1, 10**9000 + 9)
    source = _matrix(
        [
            [Fraction(1, 3), 0, 0, uncontracted],
            [0, Fraction(1, 5), 0, 0],
            [0, 0, Fraction(1, 7), 0],
            [uncontracted, 0, 0, Fraction(1, 11)],
        ],
        (y, z),
    )

    reduced = partial_trace(source, ("y",))
    assert _entries(reduced) == (
        (Fraction(1, 3) + Fraction(1, 7), Fraction(0)),
        (Fraction(0), Fraction(1, 5) + Fraction(1, 11)),
    )


def test_partial_trace_work_envelope_rejects_one_step_above_the_contracted_boundary() -> (
    None
):
    q = MatrixSubsystem(label="q", dimension=4)
    rejected_denominators = (
        10**4099 - 3,
        10**4099 - 5,
        10**4099 - 11,
        10**4099 - 13,
    )
    source = _matrix(
        [
            [
                Fraction(1, rejected_denominators[row]) if row == column else 0
                for column in range(4)
            ]
            for row in range(4)
        ],
        (q,),
    )

    _, peak = partial_trace_measured_entries(source, ("q",))
    assert peak > MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS

    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("q",))


def test_partial_trace_admits_cancelling_pairs_and_rereads_its_emitted_factor() -> None:
    a = MatrixSubsystem(label="a", dimension=1)
    r = MatrixSubsystem(label="r", dimension=16)
    denominators = tuple(10**2049 + offset for offset in (3, 5, 11, 13, 19, 21, 27, 29))
    diagonal = [
        entry
        for denominator in denominators
        for entry in (Fraction(1, denominator), Fraction(-1, denominator))
    ]
    assert len(str(denominators[0])) == 2050
    source = _matrix(
        [
            [diagonal[row] if row == column else 0 for column in range(16)]
            for row in range(16)
        ],
        (a, r),
    )

    reduced = partial_trace(source, ("a",))
    assert reduced.factors == (r,)
    assert _entries(reduced) == tuple(
        tuple(diagonal[row] if row == column else Fraction(0) for column in range(16))
        for row in range(16)
    )

    wire = compute_partial_trace(
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("a",))
    )
    replayed = SubsystemPartialTraceResult.model_validate(
        wire.model_dump(mode="python")
    )
    assert replayed == wire

    final = partial_trace(reduced, ("r",))
    assert final.factors == ()
    assert _entries(final) == ((Fraction(0),),)


def test_partial_trace_admits_folds_whose_cancellation_arrives_late() -> None:
    a = MatrixSubsystem(label="a", dimension=1)
    r = MatrixSubsystem(label="r", dimension=16)
    denominators = tuple(10**4097 + offset for offset in (1, 3, 5, 7, 9))
    diagonal = (
        [Fraction(1, denominator) for denominator in denominators]
        + [Fraction(-1, denominator) for denominator in denominators]
        + [Fraction(0)] * 6
    )
    source = _matrix(
        [
            [diagonal[row] if row == column else 0 for column in range(16)]
            for row in range(16)
        ],
        (a, r),
    )

    entries, peak = partial_trace_measured_entries(source, ("r",))
    assert entries == ((Fraction(0),),)
    assert peak == len(str(denominators[0]))

    emitted = partial_trace(source, ("a",))
    assert _entries(emitted) == tuple(
        tuple(diagonal[row] if row == column else Fraction(0) for column in range(16))
        for row in range(16)
    )

    stepwise = partial_trace(emitted, ("r",))
    combined = partial_trace(source, ("a", "r"))
    assert stepwise == combined
    assert _entries(stepwise) == ((Fraction(0),),)


def test_partial_trace_schema_describes_the_coupled_trace_envelopes() -> None:
    schema = SubsystemPartialTraceRequest.model_json_schema()
    description = schema["properties"]["matrix"]["description"]
    assert "work envelope" in description
    assert str(MAX_PARTIAL_TRACE_WORK_COMPONENT_DIGITS) in description
    assert str(MAX_PARTIAL_TRACE_RESULT_COMPONENT_DIGITS) in description


def test_traced_label_arrays_are_bounded_during_parsing() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    source = _matrix([[1, 0], [0, 2]], (q,))
    oversized = {
        "matrix": source.model_dump(mode="json"),
        "traced_factor_labels": ["q", "r", "s", "t", "u"],
    }

    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest.model_validate(oversized)

    request_schema = SubsystemPartialTraceRequest.model_json_schema()
    assert request_schema["properties"]["traced_factor_labels"]["maxItems"] == 4
    result_schema = SubsystemPartialTraceResult.model_json_schema()
    assert result_schema["properties"]["traced_factor_labels"]["maxItems"] == 4


def test_psd_order_is_source_bound_and_verifies_forged_claims() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    zero = _matrix([[0, 0], [0, 0]], (q,))
    positive = _matrix([[1, 0], [0, 2]], (q,))
    accepted = psd_order(zero, positive)
    assert accepted.is_less_or_equal is True
    assert accepted.inertia.n_negative == 0
    assert accepted.negative_witness is None

    indefinite = _matrix([[-1, 0], [0, 2]], (q,))
    rejected = decide_psd_order(PsdOrderRequest(left=zero, right=indefinite))
    assert rejected.is_less_or_equal is False
    assert rejected.inertia.n_positive == 1
    assert rejected.inertia.n_negative == 1
    assert rejected.negative_witness is not None
    assert rejected.negative_witness.quadratic_value == _q(-1)

    forged = rejected.model_dump(mode="python")
    forged["difference"] = positive
    assert not _verify_psd_order_result(PsdOrderResult.model_validate(forged))
    forged = rejected.model_dump(mode="python")
    forged["inertia"] = {"n_positive": 2, "n_negative": 0, "n_zero": 0}
    forged["is_less_or_equal"] = True
    forged["negative_witness"] = None
    assert not _verify_psd_order_result(PsdOrderResult.model_validate(forged))


def test_psd_order_rejects_same_shape_but_different_subsystem_identity() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left = _matrix([[0, 0], [0, 0]], (q,))
    right = _matrix([[1, 0], [0, 1]], (r,))
    with pytest.raises(ValidationError):
        PsdOrderRequest(left=left, right=right)


def test_psd_order_agrees_with_the_two_by_two_principal_minor_criterion() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    zero = _matrix([[0, 0], [0, 0]], (q,))
    for diagonal_left, off_diagonal, diagonal_right in product(range(-1, 2), repeat=3):
        candidate = _matrix(
            [[diagonal_left, off_diagonal], [off_diagonal, diagonal_right]],
            (q,),
        )
        result = psd_order(zero, candidate)
        expected = (
            diagonal_left >= 0
            and diagonal_right >= 0
            and diagonal_left * diagonal_right - off_diagonal**2 >= 0
        )
        assert result.is_less_or_equal is expected
        if result.negative_witness is not None:
            assert result.negative_witness.quadratic_value.as_fraction() < 0


def test_operation_input_digits_are_checked_before_exact_backend_work() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    first = Fraction(1, 10**257)
    second = Fraction(1, 10**257 + 21)
    source = _matrix(
        [[first, 0], [0, first]],
        (q,),
    )
    other = _matrix(
        [[second, 0], [0, second]],
        (q,),
    )
    with pytest.raises(ValidationError):
        PsdOrderRequest(left=source, right=other)


def test_psd_order_admits_identical_operands_before_witness_growth() -> None:
    q = MatrixSubsystem(label="q", dimension=4)
    r = MatrixSubsystem(label="r", dimension=4)
    left_value = Fraction(1, 10**127 + 159)
    right_value = Fraction(1, 10**127 + 197)

    def scaled(value: Fraction, factor: MatrixSubsystem) -> FactorizedHermitianMatrix:
        return _matrix(
            [
                [value if row == column else 0 for column in range(4)]
                for row in range(4)
            ],
            (factor,),
        )

    operand = compute_kronecker_product(
        SubsystemKroneckerProductRequest(
            left=scaled(left_value, q),
            right=scaled(right_value, r),
        )
    ).product
    assert len(operand.matrix.entries[0][0].den) == 255

    ordered = decide_psd_order(PsdOrderRequest(left=operand, right=operand))
    assert ordered.is_less_or_equal is True
    assert (ordered.inertia.n_positive, ordered.inertia.n_negative) == (0, 0)
    assert ordered.inertia.n_zero == 16
    assert ordered.negative_witness is None
    assert _entries(ordered.difference) == tuple(
        tuple(Fraction(0) for _ in range(16)) for _ in range(16)
    )
    assert PsdOrderResult.model_validate(ordered.model_dump(mode="python")) == ordered


def test_psd_order_admits_nearly_equal_operands_with_a_tiny_reduced_difference() -> (
    None
):
    q = MatrixSubsystem(label="q", dimension=16)
    shared_denominator = 10**254 + 19
    left = _matrix(
        [
            [
                Fraction(1, shared_denominator) if row == column else 0
                for column in range(16)
            ]
            for row in range(16)
        ],
        (q,),
    )
    right_entries = [
        [
            (
                Fraction(shared_denominator + 1, shared_denominator)
                if (row, column) == (0, 0)
                else (Fraction(1, shared_denominator) if row == column else 0)
            )
            for column in range(16)
        ]
        for row in range(16)
    ]
    right = _matrix(right_entries, (q,))
    assert len(left.matrix.entries[0][0].den) == 255

    ordered = decide_psd_order(PsdOrderRequest(left=left, right=right))
    assert ordered.is_less_or_equal is True
    expected_difference = tuple(
        tuple(
            Fraction(1) if (row, column) == (0, 0) else Fraction(0)
            for column in range(16)
        )
        for row in range(16)
    )
    assert _entries(ordered.difference) == expected_difference
    assert (ordered.inertia.n_positive, ordered.inertia.n_negative) == (1, 0)
    assert ordered.inertia.n_zero == 15
    assert ordered.negative_witness is None
    assert PsdOrderResult.model_validate(ordered.model_dump(mode="python")) == ordered


def _dense_equal_operand(digits: int) -> FactorizedHermitianMatrix:
    q = MatrixSubsystem(label="q", dimension=16)
    value = Fraction(10**digits + 3, 10**digits + 9)
    return _matrix([[value] * 16 for _ in range(16)], (q,))


def test_psd_order_rejects_results_beyond_the_canonical_output_limit() -> None:
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    operand = _dense_equal_operand(digits=10224)
    request_bytes = 2 * len(encode_strict_json(operand.model_dump(mode="json")))
    assert request_bytes <= CanonicalLimits().max_input_bytes

    with pytest.raises(ValidationError):
        PsdOrderRequest(left=operand, right=operand)


def test_psd_order_admits_dense_equal_operands_inside_the_output_budget() -> None:
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    operand = _dense_equal_operand(digits=10150)
    request_bytes = 2 * len(encode_strict_json(operand.model_dump(mode="json")))
    assert request_bytes <= CanonicalLimits().max_input_bytes

    ordered = decide_psd_order(PsdOrderRequest(left=operand, right=operand))
    assert ordered.is_less_or_equal is True
    encoded = encode_strict_json(
        {
            "operation_id": "matrix.subsystem.psd_order.decide",
            "runtime_ms": 1,
            "output": ordered.model_dump(mode="json"),
        }
    )
    assert len(encoded) <= CanonicalLimits().max_output_bytes


def _trace_source_with_unused_large_block(
    cross_digits: int, diagonal_digits: int
) -> FactorizedHermitianMatrix:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=8)
    cross = Fraction(10**cross_digits + 3, 10**cross_digits + 9)
    diagonal = Fraction(10**diagonal_digits + 3, 10**diagonal_digits + 9)
    entries = [[None] * 16 for _ in range(16)]
    for row in range(16):
        row_q, _ = divmod(row, 8)
        for column in range(16):
            column_q, _ = divmod(column, 8)
            if row_q == column_q:
                value = diagonal if row_q == 0 else -diagonal
            else:
                value = cross
            entries[row][column] = value
    return _matrix(entries, (q, r))


def test_partial_trace_rejects_results_beyond_the_canonical_output_limit() -> None:
    from jacobian.canonical import (
        CanonicalizationError,
        encode_strict_json,
    )

    source = _trace_source_with_unused_large_block(
        cross_digits=25000, diagonal_digits=16300
    )
    with pytest.raises(CanonicalizationError):
        encode_strict_json(source.model_dump(mode="json"))

    with pytest.raises(ValidationError):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("q",))


def test_partial_trace_admits_sources_inside_the_output_budget() -> None:
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    source = _trace_source_with_unused_large_block(
        cross_digits=20000, diagonal_digits=10000
    )

    wire = compute_partial_trace(
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("q",))
    )
    encoded = encode_strict_json(
        {
            "operation_id": "matrix.subsystem.partial_trace.compute",
            "runtime_ms": 1,
            "output": wire.model_dump(mode="json"),
        }
    )
    assert len(encoded) <= CanonicalLimits().max_output_bytes


def test_kronecker_product_schema_describes_the_exact_product_component_envelope() -> (
    None
):
    schema = SubsystemKroneckerProductRequest.model_json_schema()
    for side in ("left", "right"):
        description = schema["properties"][side]["description"]
        assert "exact product coefficients" in description
        assert str(MAX_KRONECKER_RESULT_COMPONENT_DIGITS) in description


def test_kronecker_product_admits_asymmetric_operand_digit_growth() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    heavy = Fraction(1, 10**199 + 3)
    wide = _matrix([[heavy, 0], [0, 1]], (q,))
    compact = _matrix([[1, 0], [0, 1]], (r,))

    product = kronecker_product(wide, compact)
    assert len(product.matrix.entries[0][0].den) == 200

    with pytest.raises(ValidationError):
        SubsystemKroneckerProductRequest(
            left=_matrix([[heavy, 0], [0, 1]], (r,)),
            right=wide,
        )


def test_psd_order_result_round_trip_does_not_replay_source_admission() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    zero = _matrix([[0, 0], [0, 0]], (q,))
    positive = _matrix([[1, 0], [0, 2]], (q,))
    accepted = psd_order(zero, positive)
    assert PsdOrderResult.model_validate(accepted.model_dump(mode="python")) == accepted

    large_first = Fraction(1, 10**257)
    large_second = Fraction(1, 10**257 + 21)
    wide_left = _matrix(
        [[large_first, 0], [0, large_first]],
        (q,),
    )
    wide_right = _matrix(
        [[large_second, 0], [0, large_second]],
        (q,),
    )
    forged = accepted.model_dump(mode="python")
    forged["left"] = wide_left
    forged["right"] = wide_right
    forged["difference"] = zero
    forged["inertia"] = {"n_positive": 0, "n_negative": 0, "n_zero": 2}
    structural = PsdOrderResult.model_validate(forged)
    assert not _verify_psd_order_result(structural)


def test_kronecker_product_replays_through_trace_and_psd_order_at_derived_bound() -> (
    None
):
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left_value = Fraction(1, 10**63 + 159)
    right_value = Fraction(1, 10**63 + 197)
    left = _matrix([[left_value, 0], [0, left_value]], (q,))
    right = _matrix([[right_value, 0], [0, right_value]], (r,))

    product_matrix = compute_kronecker_product(
        SubsystemKroneckerProductRequest(left=left, right=right)
    ).product
    assert len(product_matrix.matrix.entries[0][0].den) == 127

    traced = compute_partial_trace(
        SubsystemPartialTraceRequest(
            matrix=product_matrix,
            traced_factor_labels=("q",),
        )
    ).reduced_matrix
    assert _entries(traced) == (
        (2 * left_value * right_value, Fraction(0)),
        (Fraction(0), 2 * left_value * right_value),
    )
    ordered = decide_psd_order(
        PsdOrderRequest(left=product_matrix, right=product_matrix)
    )
    assert ordered.is_less_or_equal is True


def test_psd_order_rejects_a_large_product_before_witness_expansion() -> None:
    q = MatrixSubsystem(label="q", dimension=4)
    r = MatrixSubsystem(label="r", dimension=4)
    left_value = Fraction(1, 10**127 + 159)
    right_value = Fraction(1, 10**127 + 197)
    third_value = Fraction(1, 10**127 + 211)

    def scaled(value: Fraction, factor: MatrixSubsystem) -> FactorizedHermitianMatrix:
        return _matrix(
            [
                [value if row == column else 0 for column in range(4)]
                for row in range(4)
            ],
            (factor,),
        )

    left_product = compute_kronecker_product(
        SubsystemKroneckerProductRequest(
            left=scaled(left_value, q),
            right=scaled(right_value, r),
        )
    ).product
    right_product = compute_kronecker_product(
        SubsystemKroneckerProductRequest(
            left=scaled(left_value, q),
            right=scaled(third_value, r),
        )
    ).product
    assert len(left_product.matrix.entries[0][0].den) == 255

    with pytest.raises(ValidationError):
        PsdOrderRequest(left=left_product, right=right_product)


def test_partial_trace_readmits_its_emitted_values_across_sequential_traces() -> None:
    y = MatrixSubsystem(label="y", dimension=2)
    z = MatrixSubsystem(label="z", dimension=2)
    first = Fraction(1, 10**255 + 19)
    second = Fraction(1, 10**255 + 21)
    source = _matrix(
        [
            [first, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, second, 0],
            [0, 0, 0, 1],
        ],
        (y, z),
    )

    reduced = partial_trace(source, ("y",))
    assert len(reduced.matrix.entries[0][0].den) == 511
    stepwise = partial_trace(reduced, ("z",))
    combined = partial_trace(source, ("y", "z"))
    assert stepwise == combined
    assert stepwise.factors == ()
    assert _entries(stepwise) == ((first + second + 2,),)


def test_partial_trace_readmits_sequential_boundary_traces_of_one_coordinate() -> None:
    y = MatrixSubsystem(label="y", dimension=2)
    z = MatrixSubsystem(label="z", dimension=2)
    first = Fraction(1, 10**2047 + 19)
    second = Fraction(1, 10**2047 + 21)
    source = _matrix(
        [
            [first, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, second, 0],
            [0, 0, 0, 1],
        ],
        (y, z),
    )

    reduced = partial_trace(source, ("y",))
    assert len(reduced.matrix.entries[0][0].den) == 4095
    stepwise = partial_trace(reduced, ("z",))
    combined = partial_trace(source, ("y", "z"))
    assert stepwise == combined
    assert stepwise.factors == ()
    assert _entries(stepwise) == ((first + second + 2,),)


def test_kronecker_product_readmits_its_emitted_product_as_an_operand() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    s = MatrixSubsystem(label="s", dimension=2)
    first_value = Fraction(1, 10**64 + 7)
    second_value = Fraction(1, 10**64 + 9)
    left = _matrix([[first_value, 0], [0, first_value]], (q,))
    middle = _matrix([[second_value, 0], [0, second_value]], (r,))
    right = _matrix([[1, 0], [0, 1]], (s,))

    once = kronecker_product(left, middle)
    assert len(once.matrix.entries[0][0].den) == 129
    twice = kronecker_product(once, right)
    expected_value = first_value * second_value
    assert twice.factors == (q, r, s)
    assert _entries(twice) == tuple(
        tuple(expected_value if row == column else Fraction(0) for column in range(8))
        for row in range(8)
    )
    assert kronecker_product(left, kronecker_product(middle, right)) == twice


def test_kronecker_product_composes_boundary_products_with_identity_operands() -> None:
    q = MatrixSubsystem(label="q", dimension=1)
    r = MatrixSubsystem(label="r", dimension=1)
    s = MatrixSubsystem(label="s", dimension=1)
    left_value = Fraction(1, 9 * 10**127 + 1)
    right_value = Fraction(1, 9 * 10**127 + 3)
    product = kronecker_product(
        _matrix([[left_value]], (q,)),
        _matrix([[right_value]], (r,)),
    )
    assert len(product.matrix.entries[0][0].den) == 256

    identity = _matrix([[1]], (s,))
    scaled = kronecker_product(product, identity)
    assert scaled.factors == (q, r, s)
    assert _entries(scaled) == ((left_value * right_value,),)
    prefixed = kronecker_product(identity, product)
    assert prefixed.factors == (s, q, r)
    assert _entries(prefixed) == ((left_value * right_value,),)


def test_native_functions_admit_through_one_typed_request_parse() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left = _matrix([[1, 0], [0, 2]], (q,))
    right = _matrix([[3, 0], [0, 4]], (r,))

    with pytest.raises(ValidationError):
        kronecker_product(left, left)
    with pytest.raises(ValidationError):
        partial_trace(left, ("missing",))
    with pytest.raises(ValidationError):
        psd_order(left, right)


def test_catalog_wrappers_run_the_single_parsed_request_without_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left = _matrix([[1, 0], [0, 2]], (q,))
    right = _matrix([[3, 0], [0, 4]], (r,))
    cases: tuple[
        tuple[type[BaseModel], Callable[[], Any], Callable[[Any], Any]], ...
    ] = (
        (
            SubsystemKroneckerProductRequest,
            lambda: SubsystemKroneckerProductRequest(left=left, right=right),
            compute_kronecker_product,
        ),
        (
            SubsystemPartialTraceRequest,
            lambda: SubsystemPartialTraceRequest(
                matrix=left,
                traced_factor_labels=("q",),
            ),
            compute_partial_trace,
        ),
        (
            PsdOrderRequest,
            lambda: PsdOrderRequest(left=left, right=left),
            decide_psd_order,
        ),
    )
    for model, build, compute in cases:
        parsed: list[BaseModel] = []
        inner = model.__init__

        def counted(
            self: BaseModel,
            *args: object,
            __inner: Callable[..., None] = inner,
            __parsed: list[BaseModel] = parsed,
            **kwargs: object,
        ) -> None:
            __parsed.append(self)
            __inner(self, *args, **kwargs)

        monkeypatch.setattr(model, "__init__", counted)
        compute(build())
        monkeypatch.undo()
        assert len(parsed) == 1
