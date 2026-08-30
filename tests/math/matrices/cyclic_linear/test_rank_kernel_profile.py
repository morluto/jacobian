"""Defining examples for rational cyclic rank and kernel profiles."""

from __future__ import annotations

import time
from fractions import Fraction
from threading import Event
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.matrices.cyclic_linear import (
    CyclicRationalBlockSymbol,
    CyclicRationalBlockSymbolEntry,
    CyclicRationalRankKernelProfileRequest,
    cyclic_rational_rank_kernel_profile,
)
from jacobian.math.matrices.cyclic_linear._tools import TOOLS
from jacobian.math.matrices.cyclic_linear.operations import (
    CyclicRankKernelAdmissionError,
)
from jacobian.math.matrices.values import (
    SimpleNumberFieldMatrix,
    SimpleNumberFieldVectorSpaceBasis,
)
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
    embeddings,
)
from jacobian.math.number_theory.number_fields.operations import (
    NumberFieldEmbeddingAdmissionError,
)
from jacobian.process import bounded_process_cancellation


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _symbol(
    *,
    period: int,
    source_dimension: int = 1,
    target_dimension: int = 1,
    entries: tuple[tuple[int, int, int, Fraction | int], ...],
) -> CyclicRationalBlockSymbol:
    return CyclicRationalBlockSymbol(
        period=period,
        source_block_dimension=source_dimension,
        target_block_dimension=target_dimension,
        entries=tuple(
            CyclicRationalBlockSymbolEntry(
                target_coordinate=target,
                source_coordinate=source,
                shift=shift,
                coefficient=(
                    _q(value)
                    if isinstance(value, int)
                    else CanonicalRational.from_fraction(value)
                ),
            )
            for target, source, shift, value in entries
        ),
    )


def _coefficients(element: object) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in element.coefficients_ascending)  # type: ignore[attr-defined]


def test_scalar_x_minus_one_drops_only_the_trivial_component() -> None:
    result = cyclic_rational_rank_kernel_profile(
        _symbol(period=6, entries=((0, 0, 0, -1), (0, 0, 1, 1)))
    )

    assert tuple(component.order for component in result.components) == (1, 2, 3, 6)
    assert tuple(
        (component.order, component.rank, component.nullity)
        for component in result.components
    ) == ((1, 0, 1), (2, 1, 0), (3, 1, 0), (6, 1, 0))
    assert result.exceptional_component_orders == (1,)
    assert (result.global_rank, result.global_nullity) == (5, 1)
    assert result.global_kernel_basis.ambient_dimension == 6
    vector = tuple(
        value.as_fraction() for value in result.global_kernel_basis.vectors[0]
    )
    assert len(set(vector)) == 1
    assert vector[0] != 0
    assert cyclic_rational_rank_kernel_profile(result.symbol) == result


def _rational_rowspace(
    vectors: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    from sympy import Matrix

    reduced, _pivots = Matrix(vectors).rref()
    return tuple(
        tuple(Fraction(int(value.p), int(value.q)) for value in vector)
        for vector in reduced.tolist()
    )


def _expanded_matrix(symbol: CyclicRationalBlockSymbol) -> Any:
    from sympy import Matrix, Rational

    row_count = symbol.period * symbol.target_block_dimension
    column_count = symbol.period * symbol.source_block_dimension
    rows = [[Rational(0) for _ in range(column_count)] for _ in range(row_count)]
    for entry in symbol.entries:
        numerator, denominator = entry.coefficient.as_integer_ratio()
        coefficient = Rational(numerator, denominator)
        for source_shift in range(symbol.period):
            target_shift = (source_shift + entry.shift) % symbol.period
            row = target_shift * symbol.target_block_dimension + entry.target_coordinate
            column = (
                source_shift * symbol.source_block_dimension + entry.source_coordinate
            )
            rows[row][column] += coefficient
    return Matrix(rows)


def _public_polynomial(value: object) -> Any:
    from sympy import Poly, Rational, Symbol

    x = Symbol("x")
    expression = 0
    for term in value.polynomial.terms:  # type: ignore[attr-defined]
        numerator, denominator = term.coefficient.as_integer_ratio()
        expression += Rational(numerator, denominator) * x ** term.exponents[0]
    return Poly(expression, x, domain="QQ")


def test_composite_period_distinguishes_galois_components() -> None:
    # Phi_3(x) = x^2 + x + 1 vanishes only on the order-three component of C_6.
    result = cyclic_rational_rank_kernel_profile(
        _symbol(
            period=6,
            entries=((0, 0, 0, 1), (0, 0, 1, 1), (0, 0, 2, 1)),
        )
    )

    assert tuple(
        (component.order, component.field.degree, component.rank, component.nullity)
        for component in result.components
    ) == ((1, 1, 1, 0), (2, 1, 1, 0), (3, 2, 0, 1), (6, 2, 1, 0))
    assert result.exceptional_component_orders == (3,)
    assert (result.global_rank, result.global_nullity) == (4, 2)
    vectors = tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in result.global_kernel_basis.vectors
    )
    expected = (
        (
            Fraction(1),
            Fraction(0),
            Fraction(-1),
            Fraction(1),
            Fraction(0),
            Fraction(-1),
        ),
        (
            Fraction(0),
            Fraction(1),
            Fraction(-1),
            Fraction(0),
            Fraction(1),
            Fraction(-1),
        ),
    )
    assert _rational_rowspace(vectors) == _rational_rowspace(expected)


def test_block_rank_drop_returns_source_bound_minor_and_kernel() -> None:
    # diag(x-1, 1) on C_3 has one trivial-component kernel direction.
    source = _symbol(
        period=3,
        source_dimension=2,
        target_dimension=2,
        entries=(
            (0, 0, 0, -1),
            (0, 0, 1, 1),
            (1, 1, 0, 1),
        ),
    )
    result = cyclic_rational_rank_kernel_profile(source)

    assert tuple(
        (component.order, component.rank, component.nullity)
        for component in result.components
    ) == ((1, 1, 1), (3, 2, 0))
    trivial = result.components[0]
    assert trivial.nonzero_minor is not None
    assert trivial.nonzero_minor.row_indices == (1,)
    assert trivial.nonzero_minor.column_indices == (1,)
    assert _coefficients(trivial.nonzero_minor.determinant) == (Fraction(1),)
    assert tuple(_coefficients(value) for value in trivial.kernel_basis.vectors[0]) == (
        (Fraction(1),),
        (Fraction(0),),
    )
    assert (result.global_rank, result.global_nullity) == (5, 1)


@pytest.mark.parametrize("kind", ["zero", "identity"])
def test_zero_and_identity_operators_have_complete_exact_profiles(kind: str) -> None:
    entries: tuple[tuple[int, int, int, Fraction | int], ...]
    entries = () if kind == "zero" else ((0, 0, 0, 1), (1, 1, 0, 1))
    result = cyclic_rational_rank_kernel_profile(
        _symbol(
            period=4,
            source_dimension=2,
            target_dimension=2,
            entries=entries,
        )
    )

    expected_rank = 0 if kind == "zero" else 8
    assert result.global_rank == expected_rank
    assert result.global_nullity == 8 - expected_rank
    assert len(result.global_kernel_basis.vectors) == result.global_nullity
    assert all(
        component.nonzero_minor is None
        for component in result.components
        if kind == "zero"
    )
    assert all(
        component.nonzero_minor is not None
        for component in result.components
        if kind == "identity"
    )


def test_period_59_block_fixture_is_representable_and_reconstructs() -> None:
    # A nontrivial unimodular Laurent symbol keeps the required degree-58 field
    # component exact without expanding a 118 by 118 source matrix.
    result = cyclic_rational_rank_kernel_profile(
        _symbol(
            period=59,
            source_dimension=2,
            target_dimension=2,
            entries=(
                (0, 0, 0, 1),
                (0, 1, 1, 1),
                (1, 1, 0, 1),
            ),
        )
    )

    assert tuple(
        (component.order, component.field.degree) for component in result.components
    ) == (
        (1, 1),
        (59, 58),
    )
    assert (result.global_rank, result.global_nullity) == (118, 0)
    assert result.global_kernel_basis.ambient_dimension == 118
    assert result.global_kernel_basis.vectors == ()
    assert (
        SimpleNumberFieldPresentation.model_validate(
            result.components[1].field.model_dump(mode="json"), strict=True
        )
        == result.components[1].field
    )
    assert (
        SimpleNumberFieldMatrix.model_validate(
            result.components[1].component_matrix.model_dump(mode="json"), strict=True
        )
        == result.components[1].component_matrix
    )
    assert (
        SimpleNumberFieldVectorSpaceBasis.model_validate(
            result.components[1].kernel_basis.model_dump(mode="json"), strict=True
        )
        == result.components[1].kernel_basis
    )

    # Widening the shared carrier to degree 58 does not widen the distinct
    # exact-embedding operation's proved degree-eight execution envelope.
    with pytest.raises(NumberFieldEmbeddingAdmissionError, match="degree 8"):
        embeddings(result.components[1].field)


def test_crt_idempotents_select_exactly_their_components() -> None:
    from sympy import Poly, Symbol, cyclotomic_poly

    x = Symbol("x")
    result = cyclic_rational_rank_kernel_profile(
        _symbol(period=6, entries=((0, 0, 0, 1),))
    )
    total = Poly(x**6 - 1, x, domain="QQ")
    idempotents: dict[int, Any] = {
        component.order: _public_polynomial(component.crt_idempotent)
        for component in result.components
    }

    assert sum(idempotents.values(), Poly(0, x, domain="QQ")).rem(total) == Poly(
        1, x, domain="QQ"
    )
    for order, idempotent in idempotents.items():
        for comparison_order in idempotents:
            modulus = Poly(cyclotomic_poly(comparison_order, x), x, domain="QQ")
            expected = Poly(1 if comparison_order == order else 0, x, domain="QQ")
            assert idempotent.rem(modulus) == expected


@pytest.mark.parametrize(
    "source",
    [
        _symbol(period=1, entries=()),
        _symbol(period=2, entries=((0, 0, 0, 2), (0, 0, 1, -3))),
        _symbol(period=3, entries=((0, 0, 0, 1), (0, 0, 1, 1))),
        _symbol(
            period=4,
            source_dimension=2,
            target_dimension=1,
            entries=((0, 0, 0, 1), (0, 1, 1, Fraction(1, 2))),
        ),
        _symbol(
            period=6,
            source_dimension=2,
            target_dimension=2,
            entries=(
                (0, 0, 0, 1),
                (0, 0, 2, -1),
                (0, 1, 1, 2),
                (1, 0, 3, 1),
                (1, 1, 0, -1),
                (1, 1, 1, 1),
            ),
        ),
    ],
)
def test_global_profile_matches_independently_expanded_rational_map(
    source: CyclicRationalBlockSymbol,
) -> None:
    from sympy import Matrix, Rational

    expanded = _expanded_matrix(source)
    result = cyclic_rational_rank_kernel_profile(source)
    assert result.global_rank == expanded.rank()
    assert result.global_nullity == expanded.cols - expanded.rank()

    rows = []
    for vector in result.global_kernel_basis.vectors:
        sympy_vector = Matrix(
            [Rational(*coordinate.as_integer_ratio()) for coordinate in vector]
        )
        assert expanded * sympy_vector == Matrix.zeros(expanded.rows, 1)
        rows.append(list(sympy_vector))
    if rows:
        assert Matrix.hstack(*(Matrix(row) for row in rows)).rank() == len(rows)


def test_mutation_changes_the_exact_nonvanishing_witness() -> None:
    full_rank = cyclic_rational_rank_kernel_profile(
        _symbol(period=5, entries=((0, 0, 0, -1), (0, 0, 1, 1)))
    )
    mutated = cyclic_rational_rank_kernel_profile(
        _symbol(period=5, entries=((0, 0, 0, 1),))
    )

    assert full_rank.components[0].nonzero_minor is None
    assert mutated.components[0].nonzero_minor is not None
    assert _coefficients(mutated.components[0].nonzero_minor.determinant) == (
        Fraction(1),
    )
    assert full_rank.symbol != mutated.symbol


def test_request_and_result_round_trip_strictly() -> None:
    request = CyclicRationalRankKernelProfileRequest(
        symbol=_symbol(period=4, entries=((0, 0, 0, 1), (0, 0, 2, -1)))
    )
    result = TOOLS[0].run(request)

    assert (
        CyclicRationalRankKernelProfileRequest.model_validate(
            request.model_dump(mode="json"), strict=True
        )
        == request
    )
    assert (
        TOOLS[0].result_type.model_validate(result.model_dump(mode="json"), strict=True)
        == result
    )


def test_symbol_requires_canonical_support_and_bounded_rationals() -> None:
    with pytest.raises(ValueError, match=r"row-major|canonical"):
        _symbol(period=3, entries=((0, 0, 1, 1), (0, 0, 0, 1)))
    with pytest.raises(ValueError, match="zero"):
        _symbol(period=3, entries=((0, 0, 0, 0),))
    with pytest.raises(ValueError, match="64 decimal digits"):
        _symbol(period=3, entries=((0, 0, 0, 10**64),))


def test_fraction_free_height_bound_rejects_before_elimination() -> None:
    source = _symbol(
        period=1,
        source_dimension=128,
        target_dimension=128,
        entries=tuple((index, index, 0, 10**63) for index in range(128)),
    )
    with pytest.raises(CyclicRankKernelAdmissionError, match="fraction-free"):
        cyclic_rational_rank_kernel_profile(source)


def test_owner_checkpoint_observes_cancellation() -> None:
    cancellation = Event()
    cancellation.set()
    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="cancelled"),
    ):
        cyclic_rational_rank_kernel_profile(_symbol(period=3, entries=((0, 0, 0, 1),)))


def test_owner_checkpoint_observes_existing_request_deadline() -> None:
    started = time.monotonic()
    with (
        request_execution(started),
        pytest.raises(OperationExecutionTimeoutError, match="deadline expired"),
    ):
        bind_request_deadline(started - 1)
        cyclic_rational_rank_kernel_profile(_symbol(period=3, entries=((0, 0, 0, 1),)))
