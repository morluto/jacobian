"""Defining examples for rational cyclic rank and kernel profiles."""

from __future__ import annotations

import time
from fractions import Fraction
from threading import Event
from typing import Any

import pytest
from jsonschema import Draft202012Validator

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
    CyclicRationalRankKernelProfile,
    RationalCyclotomicElement,
    RationalCyclotomicField,
    RationalCyclotomicMatrix,
    RationalCyclotomicVectorSpaceBasis,
    cyclic_rational_rank_kernel_profile,
)
from jacobian.math.matrices.cyclic_linear._models import (
    CyclicRationalRankKernelProfileRequest,
)
from jacobian.math.matrices.cyclic_linear._tools import TOOLS
from jacobian.math.matrices.cyclic_linear.operations import (
    _CYCLIC_PROFILE_WALL_SECONDS,
    CyclicRankKernelAdmissionError,
    _multiplication_norm,
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


def test_multiplication_norm_matches_power_basis_structure_constants() -> None:
    import sympy

    variable = sympy.Symbol("x")
    for order in (3, 4, 5, 8):
        polynomial = sympy.Poly(sympy.cyclotomic_poly(order, variable), variable)
        degree = polynomial.degree()
        output_sums = [0] * degree
        for left_power in range(degree):
            for right_power in range(degree):
                remainder = sympy.Poly(
                    variable ** (left_power + right_power), variable
                ).rem(polynomial)
                for output_power in range(degree):
                    output_sums[output_power] += abs(int(remainder.nth(output_power)))

        assert _multiplication_norm(polynomial, degree, variable) == max(output_sums)


def test_cyclotomic_element_has_explicit_shared_number_field_conversion() -> None:
    element = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=3),
        coefficients_ascending=(_q(0), _q(1)),
    )

    converted = element.to_simple_number_field_element()

    assert converted.presentation.coefficients_descending == ("1", "1", "1")
    assert converted.coefficients_ascending == element.coefficients_ascending


def test_order_67_cyclotomic_element_converts_to_shared_number_field() -> None:
    element = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=67),
        coefficients_ascending=(_q(1),) + (_q(0),) * 65,
    )

    converted = element.to_simple_number_field_element()

    assert converted.presentation.degree == 66
    assert converted.coefficients_ascending == element.coefficients_ascending


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


def _public_cyclotomic_element_polynomial(
    value: RationalCyclotomicElement,
) -> Any:
    from sympy import Poly, Rational, Symbol

    x = Symbol("x")
    return Poly(
        sum(
            Rational(*coefficient.as_integer_ratio()) * x**power
            for power, coefficient in enumerate(value.coefficients_ascending)
        ),
        x,
        domain="QQ",
    )


def test_composite_period_distinguishes_galois_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.process as process

    worker_calls = 0
    run_bounded_process = process.run_bounded_process

    def count_worker(*args: Any, **kwargs: Any) -> process.BoundedProcessResult:
        nonlocal worker_calls
        worker_calls += 1
        return run_bounded_process(*args, **kwargs)

    monkeypatch.setattr(process, "run_bounded_process", count_worker)
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
    assert worker_calls == 1
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


def test_component_matrix_minor_and_kernel_satisfy_direct_quotient_identities() -> None:
    from sympy import Matrix, Poly, Rational, Symbol, cyclotomic_poly

    source = _symbol(
        period=5,
        source_dimension=3,
        target_dimension=2,
        entries=(
            (0, 0, 1, 1),
            (0, 1, 0, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 1),
            (1, 2, 0, 1),
        ),
    )
    result = cyclic_rational_rank_kernel_profile(source)
    component = next(item for item in result.components if item.order == 5)
    x = Symbol("x")
    modulus = Poly(cyclotomic_poly(5, x), x, domain="QQ")

    for target in range(source.target_block_dimension):
        for source_coordinate in range(source.source_block_dimension):
            expected = Poly(
                sum(
                    Rational(*entry.coefficient.as_integer_ratio()) * x**entry.shift
                    for entry in source.entries
                    if entry.target_coordinate == target
                    and entry.source_coordinate == source_coordinate
                ),
                x,
                domain="QQ",
            ).rem(modulus)
            actual = _public_cyclotomic_element_polynomial(
                component.component_matrix.entries[target][source_coordinate]
            ).rem(modulus)
            assert actual == expected

    assert component.nullity == 1
    assert len(component.kernel_basis.vectors) == 1
    kernel_vector = component.kernel_basis.vectors[0]
    assert any(
        not _public_cyclotomic_element_polynomial(value).is_zero
        for value in kernel_vector
    )
    for matrix_row in component.component_matrix.entries:
        image = sum(
            (
                _public_cyclotomic_element_polynomial(matrix_value)
                * _public_cyclotomic_element_polynomial(vector_value)
                for matrix_value, vector_value in zip(
                    matrix_row, kernel_vector, strict=True
                )
            ),
            Poly(0, x, domain="QQ"),
        )
        assert image.rem(modulus).is_zero

    assert component.nonzero_minor is not None
    minor = component.nonzero_minor
    selected = Matrix(
        [
            [
                _public_cyclotomic_element_polynomial(
                    component.component_matrix.entries[row][column]
                ).as_expr()
                for column in minor.column_indices
            ]
            for row in minor.row_indices
        ]
    )
    determinant = Poly(selected.det(), x, domain="QQ").rem(modulus)
    assert determinant == _public_cyclotomic_element_polynomial(minor.determinant).rem(
        modulus
    )


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


def test_period_59_phi_59_flips_rank_and_reconstructs_nontrivial_kernel() -> None:
    # Phi_59 = 1 + x + ... + x^58 is 59 on the trivial component and zero on
    # the degree-58 primitive component. This is the source-scale rank flip
    # that a transcendental-symbol rank cannot see.
    result = cyclic_rational_rank_kernel_profile(
        _symbol(
            period=59,
            entries=tuple((0, 0, shift, 1) for shift in range(59)),
        )
    )

    assert tuple(
        (
            component.order,
            component.field.degree,
            component.rank,
            component.nullity,
        )
        for component in result.components
    ) == (
        (1, 1, 1, 0),
        (59, 58, 0, 1),
    )
    assert result.exceptional_component_orders == (59,)
    assert (result.global_rank, result.global_nullity) == (1, 58)
    assert result.global_kernel_basis.ambient_dimension == 59
    assert len(result.global_kernel_basis.vectors) == 58

    global_vectors = tuple(
        tuple(coordinate.as_fraction() for coordinate in vector)
        for vector in result.global_kernel_basis.vectors
    )
    # This source expands to the all-ones matrix, so its kernel equation is
    # exactly zero coordinate sum. FLINT independently checks basis rank.
    assert all(sum(vector) == 0 for vector in global_vectors)
    from jacobian.math.matrices._flint import rational_rref

    assert rational_rref(global_vectors)[1] == 58

    mutated = cyclic_rational_rank_kernel_profile(
        _symbol(
            period=59,
            entries=tuple((0, 0, shift, 2 if shift == 0 else 1) for shift in range(59)),
        )
    )
    primitive = next(
        component for component in mutated.components if component.order == 59
    )
    assert (primitive.rank, primitive.nullity) == (1, 0)
    assert primitive.nonzero_minor is not None
    assert (
        _coefficients(primitive.nonzero_minor.determinant)
        == (Fraction(1),) + (Fraction(0),) * 57
    )
    assert (mutated.global_rank, mutated.global_nullity) == (59, 0)

    assert (
        RationalCyclotomicField.model_validate(
            result.components[1].field.model_dump(mode="json"), strict=True
        )
        == result.components[1].field
    )
    assert (
        RationalCyclotomicMatrix.model_validate(
            result.components[1].component_matrix.model_dump(mode="json"), strict=True
        )
        == result.components[1].component_matrix
    )
    assert (
        RationalCyclotomicVectorSpaceBasis.model_validate(
            result.components[1].kernel_basis.model_dump(mode="json"), strict=True
        )
        == result.components[1].kernel_basis
    )
    assert (
        RationalCyclotomicElement.model_validate(
            result.components[1].kernel_basis.vectors[0][0].model_dump(mode="json"),
            strict=True,
        )
        == result.components[1].kernel_basis.vectors[0][0]
    )


def test_cyclotomic_parent_is_bound_to_exact_component_order() -> None:
    result = cyclic_rational_rank_kernel_profile(
        _symbol(period=3, entries=((0, 0, 0, 1),))
    )
    payload = result.model_dump(mode="json")
    components = payload["components"]
    assert isinstance(components, list)
    component = components[1]
    assert isinstance(component, dict)
    component["order"] = 2

    with pytest.raises(ValueError, match=r"Phi_order|declared field"):
        CyclicRationalRankKernelProfile.model_validate(payload, strict=True)

    extra_polynomial_payload = result.components[1].field.model_dump(mode="json")
    extra_polynomial_payload["coefficients_descending"] = ["1", "0", "1"]
    with pytest.raises(ValueError, match="Extra inputs"):
        RationalCyclotomicField.model_validate(
            extra_polynomial_payload,
            strict=True,
        )


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


def test_native_profile_is_not_rejected_by_the_transport_byte_cap() -> None:
    source = _symbol(
        period=1,
        source_dimension=128,
        target_dimension=1,
        entries=((0, 0, 0, 10**63),),
    )

    result = cyclic_rational_rank_kernel_profile(source)

    assert (result.global_rank, result.global_nullity) == (1, 127)


def test_published_profile_adapter_uses_the_mathematical_admission() -> None:
    request = CyclicRationalRankKernelProfileRequest(
        symbol=_symbol(
            period=1,
            source_dimension=128,
            target_dimension=1,
            entries=((0, 0, 0, 10**63),),
        )
    )

    result = TOOLS[0].run(request)

    assert (result.global_rank, result.global_nullity) == (1, 127)


def test_symbol_requires_canonical_support_and_bounded_rationals() -> None:
    with pytest.raises(ValueError, match=r"row-major|canonical"):
        _symbol(period=3, entries=((0, 0, 1, 1), (0, 0, 0, 1)))
    with pytest.raises(ValueError, match="zero"):
        _symbol(period=3, entries=((0, 0, 0, 0),))
    with pytest.raises(ValueError, match="64 decimal digits"):
        _symbol(period=3, entries=((0, 0, 0, 10**64),))


def test_symbol_schema_projects_the_sign_aware_64_digit_rational_bound() -> None:
    schema = CyclicRationalRankKernelProfileRequest.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    coefficient_schema = schema["$defs"]["CyclicRationalBlockSymbolEntry"][
        "properties"
    ]["coefficient"]
    assert coefficient_schema["properties"]["num"]["maxLength"] == 65
    assert coefficient_schema["properties"]["den"]["maxLength"] == 64

    def payload(num: str, den: str = "1") -> dict[str, object]:
        return {
            "symbol": {
                "period": 1,
                "target_block_dimension": 1,
                "source_block_dimension": 1,
                "entries": [
                    {
                        "target_coordinate": 0,
                        "source_coordinate": 0,
                        "shift": 0,
                        "coefficient": {"num": num, "den": den},
                    }
                ],
            }
        }

    accepted = payload("-" + "9" * 64)
    assert validator.is_valid(accepted)
    CyclicRationalRankKernelProfileRequest.model_validate(accepted, strict=True)

    for rejected in (
        payload("9" * 65),
        payload("-" + "9" * 65),
        payload("1", "9" * 65),
        payload("1", "-1"),
    ):
        assert not validator.is_valid(rejected)
        with pytest.raises(ValueError):
            CyclicRationalRankKernelProfileRequest.model_validate(rejected, strict=True)


def test_fraction_free_height_bound_rejects_before_elimination() -> None:
    source = _symbol(
        period=1,
        source_dimension=128,
        target_dimension=128,
        entries=tuple((index, index, 0, 10**63) for index in range(128)),
    )
    with pytest.raises(CyclicRankKernelAdmissionError, match="fraction-free"):
        cyclic_rational_rank_kernel_profile(source)


def test_distinct_denominator_growth_rejects_before_large_lcm_power() -> None:
    from sympy import nextprime

    denominators = tuple(int(nextprime(10**63 + 1_000 * index)) for index in range(128))
    source = _symbol(
        period=1,
        source_dimension=128,
        target_dimension=128,
        entries=tuple(
            (index, index, 0, Fraction(1, denominator))
            for index, denominator in enumerate(denominators)
        ),
    )

    with pytest.raises(
        CyclicRankKernelAdmissionError,
        match=r"common denominator|rank power",
    ):
        cyclic_rational_rank_kernel_profile(source)


def test_period_59_scalar_height_is_charged_only_to_affected_work() -> None:
    source = _symbol(
        period=59,
        entries=((0, 0, 0, 10**63),),
    )

    result = cyclic_rational_rank_kernel_profile(source)

    assert (result.global_rank, result.global_nullity) == (59, 0)
    assert all(component.rank == 1 for component in result.components)
    assert all(component.nonzero_minor is not None for component in result.components)


def test_dense_hadamard_axis_exceeds_the_field_work_envelope() -> None:
    source = _symbol(
        period=1,
        source_dimension=64,
        target_dimension=64,
        entries=tuple(
            (
                row,
                column,
                0,
                -1 if (row & column).bit_count() % 2 else 1,
            )
            for row in range(64)
            for column in range(64)
        ),
    )

    with pytest.raises(CyclicRankKernelAdmissionError, match="scalar-bit work"):
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


def test_owner_checkpoint_observes_deadline_inside_cyclotomic_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.matrices.cyclic_linear import operations

    expired = False
    original_checkpoint = operations._require_execution_active

    def expire_at_multiplication_norm(stage: str) -> None:
        nonlocal expired
        if "multiplication-norm" in stage:
            expired = True
        original_checkpoint(stage)

    monkeypatch.setattr(
        time,
        "monotonic",
        lambda: _CYCLIC_PROFILE_WALL_SECONDS + 1 if expired else 0.0,
    )
    monkeypatch.setattr(
        operations, "_require_execution_active", expire_at_multiplication_norm
    )

    with (
        request_execution(started_at=0.0),
        pytest.raises(OperationExecutionTimeoutError, match="multiplication-norm"),
    ):
        cyclic_rational_rank_kernel_profile(
            _symbol(period=127, entries=((0, 0, 0, 1),))
        )


def test_owner_binds_deadline_from_original_request_start() -> None:
    started = time.monotonic() - _CYCLIC_PROFILE_WALL_SECONDS - 1
    with (
        request_execution(started),
        pytest.raises(OperationExecutionTimeoutError, match="deadline expired"),
    ):
        cyclic_rational_rank_kernel_profile(_symbol(period=3, entries=((0, 0, 0, 1),)))
