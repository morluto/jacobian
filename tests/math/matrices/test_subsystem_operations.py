"""Exact contracts for labelled subsystem Hermitian matrix operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices import subsystems
from jacobian.math.matrices.subsystems._models import (
    PsdOrderRequest,
    PsdOrderResult,
    SubsystemKroneckerProductRequest,
    SubsystemPartialTraceRequest,
)
from jacobian.math.matrices.subsystems._operations import (
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
    with pytest.raises(ValidationError, match="shape"):
        _matrix([[1]], (q,))
    with pytest.raises(ValidationError, match="symmetric"):
        _matrix([[1, 2], [3, 4]], (q,))
    with pytest.raises(ValidationError, match="unique"):
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

    with pytest.raises(ValidationError, match="labels"):
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
    with pytest.raises(ValidationError, match="replay"):
        wire.__class__.model_validate(forged)

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
    with pytest.raises(ValidationError, match="occur"):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("r",))
    with pytest.raises(ValidationError, match="unique"):
        SubsystemPartialTraceRequest(matrix=source, traced_factor_labels=("q", "q"))


def test_psd_order_is_source_bound_and_returns_a_replayable_negative_witness() -> None:
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
    with pytest.raises(ValidationError, match="right minus left"):
        PsdOrderResult.model_validate(forged)
    forged = rejected.model_dump(mode="python")
    forged["inertia"] = {"n_positive": 2, "n_negative": 0, "n_zero": 0}
    forged["is_less_or_equal"] = True
    forged["negative_witness"] = None
    with pytest.raises(ValidationError, match="inertia"):
        PsdOrderResult.model_validate(forged)


def test_psd_order_rejects_same_shape_but_different_subsystem_identity() -> None:
    q = MatrixSubsystem(label="q", dimension=2)
    r = MatrixSubsystem(label="r", dimension=2)
    left = _matrix([[0, 0], [0, 0]], (q,))
    right = _matrix([[1, 0], [0, 1]], (r,))
    with pytest.raises(ValidationError, match="exactly equal"):
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
    large = 10**257
    source = _matrix(
        [[Fraction(1, large), 0], [0, Fraction(1, large)]],
        (q,),
    )
    with pytest.raises(ValidationError, match="256"):
        PsdOrderRequest(left=source, right=source)


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
