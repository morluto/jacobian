"""Domain tests for exact symbolic matrices over rational-function fields."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.symbolic._models import (
    SquareSymbolicMatrixRequest,
    SymbolicCharacteristicPolynomialRequest,
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantRequest,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicLinearSystemRequest,
    SymbolicMatrix,
    SymbolicMatrixProductRequest,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_characteristic_polynomial,
    compute_symbolic_determinant,
    compute_symbolic_eigenvalues,
    compute_symbolic_matrix_product,
    compute_symbolic_rank,
)
from jacobian.math.matrices.symbolic._tools import TOOLS
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

Term = tuple[int, int, tuple[int, ...]]


def _sparse(*terms: Term) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_integer_ratio(
                    numerator, denominator
                ),
                exponents=exponents,
            )
            for numerator, denominator, exponents in terms
        )
    )


def _rf(
    variables: tuple[str, ...],
    *numerator: Term,
    denominator: Sequence[Term] | None = None,
) -> RationalFunction:
    if denominator is None:
        denominator = ((1, 1, (0,) * len(variables)),)
    return RationalFunction(
        variables=variables,
        numerator=_sparse(*numerator),
        denominator=_sparse(*denominator),
    )


def _constant(value: int) -> RationalFunction:
    return _rf((), *((value, 1, ()),) if value else ())


def _variable(variables: tuple[str, ...], index: int) -> RationalFunction:
    exponents = tuple(
        1 if position == index else 0 for position in range(len(variables))
    )
    return _rf(variables, (1, 1, exponents))


def _request(
    entries: Sequence[Sequence[RationalFunction]],
    variables: tuple[str, ...],
) -> SymbolicMatrixRequest:
    return SymbolicMatrixRequest(
        matrix=SymbolicMatrix(variables=variables, entries=tuple(map(tuple, entries)))
    )


def _product_request(
    left: Sequence[Sequence[RationalFunction]],
    right: Sequence[Sequence[RationalFunction]],
    variables: tuple[str, ...],
) -> SymbolicMatrixProductRequest:
    return SymbolicMatrixProductRequest(
        left=SymbolicMatrix(variables=variables, entries=tuple(map(tuple, left))),
        right=SymbolicMatrix(variables=variables, entries=tuple(map(tuple, right))),
    )


def _square_request(
    entries: Sequence[Sequence[RationalFunction]],
    variables: tuple[str, ...],
) -> SquareSymbolicMatrixRequest:
    return SquareSymbolicMatrixRequest(
        matrix=SymbolicMatrix(variables=variables, entries=tuple(map(tuple, entries)))
    )


def _determinant_request(
    entries: Sequence[Sequence[RationalFunction]],
    variables: tuple[str, ...],
) -> SymbolicDeterminantRequest:
    return SymbolicDeterminantRequest(matrix=_square_request(entries, variables).matrix)


def _characteristic_request(
    entries: Sequence[Sequence[RationalFunction]],
    variables: tuple[str, ...],
) -> SymbolicCharacteristicPolynomialRequest:
    return SymbolicCharacteristicPolynomialRequest(
        matrix=_square_request(entries, variables).matrix
    )


def _generic_two_by_two() -> SymbolicDeterminantRequest:
    variables = ("a", "b", "c", "d")
    a, b, c, d = (_variable(variables, index) for index in range(4))
    return _determinant_request(((a, c), (b, d)), variables)


def test_symbolic_determinant_of_two_by_two() -> None:
    result = compute_symbolic_determinant(_generic_two_by_two())
    assert isinstance(result, SymbolicDeterminantResult)
    assert result.determinant == _rf(
        ("a", "b", "c", "d"),
        (1, 1, (1, 0, 0, 1)),
        (-1, 1, (0, 1, 1, 0)),
    )


def test_symbolic_determinant_of_constant_matrix() -> None:
    request = _determinant_request(
        ((_constant(1), _constant(2)), (_constant(3), _constant(4))), ()
    )
    assert compute_symbolic_determinant(request).determinant == _constant(-2)


def test_determinant_request_rejects_unrepresentable_expansion() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    zero = _rf(variables)
    diagonal = tuple(
        _rf(
            variables,
            (1, 1, tuple(2 if position == index else 0 for position in range(8))),
            (1, 1, tuple(1 if position == index else 0 for position in range(8))),
            (1, 1, (0,) * 8),
        )
        for index in range(8)
    )
    entries = tuple(
        tuple(diagonal[row] if row == column else zero for column in range(8))
        for row in range(8)
    )

    with pytest.raises(ValidationError):
        SymbolicDeterminantRequest(
            matrix=SymbolicMatrix(variables=variables, entries=entries)
        )


def test_determinant_request_admission_does_not_execute_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.matrices.symbolic as symbolic

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("determinant kernel ran during request validation")

    monkeypatch.setattr(symbolic, "symbolic_determinant", fail_if_called)

    assert isinstance(_generic_two_by_two(), SymbolicDeterminantRequest)


def test_symbolic_rank_of_full_and_singular_matrices() -> None:
    full = compute_symbolic_rank(_generic_two_by_two())
    assert isinstance(full, SymbolicRankResult)
    assert full.rank == 2
    assert full.pivot_columns == (0, 1)

    variables = ("a",)
    a = _variable(variables, 0)
    singular = compute_symbolic_rank(_request(((a, a), (a, a)), variables))
    assert singular.rank == 1


def test_symbolic_matrix_product_is_exact_and_composes_with_rank() -> None:
    variables = ("a", "b")
    a, b = (_variable(variables, index) for index in range(2))
    one = _rf(variables, (1, 1, (0, 0)))
    product = compute_symbolic_matrix_product(
        _product_request(((a, b),), ((one,), (one,)), variables)
    )
    assert product == SymbolicMatrix(
        variables=variables,
        entries=((_rf(variables, (1, 1, (1, 0)), (1, 1, (0, 1))),),),
    )
    assert compute_symbolic_rank(SymbolicMatrixRequest(matrix=product)).rank == 1


def test_symbolic_matrix_product_cancels_rational_function_entries() -> None:
    variables = ("e", "f")
    e_over_f = _rf(
        variables,
        (1, 1, (1, 0)),
        denominator=((1, 1, (0, 1)),),
    )
    f_over_e = _rf(
        variables,
        (1, 1, (0, 1)),
        denominator=((1, 1, (1, 0)),),
    )
    product = compute_symbolic_matrix_product(
        _product_request(((e_over_f,),), ((f_over_e,),), variables)
    )
    assert product.entries == ((_rf(variables, (1, 1, (0, 0))),),)


def test_symbolic_matrix_product_rejects_field_and_shape_mismatches() -> None:
    a = _variable(("a",), 0)
    b = _variable(("b",), 0)
    with pytest.raises(ValidationError):
        SymbolicMatrixProductRequest(
            left=SymbolicMatrix(variables=("a",), entries=((a,),)),
            right=SymbolicMatrix(variables=("b",), entries=((b,),)),
        )
    with pytest.raises(ValidationError):
        _product_request(((_constant(1), _constant(1)),), ((_constant(1),),), ())


def test_symbolic_matrix_product_admits_expansion_that_collects_into_budget() -> None:
    """Raw scalar products obey the aggregate work budget, not the result cap."""

    variables = ("x",)
    many_terms = _rf(
        variables,
        *((1, 1, (exponent,)) for exponent in range(16, -1, -1)),
    )
    request = _product_request(((many_terms,),), ((many_terms,),), variables)
    product = compute_symbolic_matrix_product(request)
    collected = tuple(
        (min(exponent + 1, 33 - exponent), 1, (exponent,))
        for exponent in range(32, -1, -1)
    )
    assert product.entries == ((_rf(variables, *collected),),)


def test_symbolic_matrix_product_rejects_collected_support_over_result_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-entry result limit binds collected support, not raw products."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x", "y")
    row = _rf(variables, *((1, 1, (index, 0)) for index in range(15, -1, -1)))
    column = _rf(
        variables,
        *((1, 1, (1, index)) for index in range(15, -1, -1)),
        *((1, 1, (0, index)) for index in range(15, -1, -1)),
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # The cell's 16 * 32 = 512 raw products fit the aggregate expansion
    # budget but collect onto 272 distinct monomials, which exceeds the
    # per-entry 256-term exact result budget.
    with pytest.raises(ValidationError):
        _product_request(((row,),), ((column,),), variables)


def test_symbolic_matrix_product_rejects_cancellation_coefficient_amplification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact division can hide larger coefficients than the expansion shows."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)
    amplitude = 9 * 10**126
    heights = [min(index + 1, 63 - index) for index in range(63)]
    hidden = [
        height * amplitude * (-1) ** index for index, height in enumerate(heights)
    ]
    visible = (
        [hidden[0]]
        + [hidden[index] + hidden[index - 1] for index in range(1, 63)]
        + [hidden[62]]
    )
    amplified = _rf(
        variables,
        *((visible[power], 1, (power,)) for power in range(len(visible) - 1, -1, -1)),
    )
    one_over_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # The unreduced expansion fits every budget, yet the reduced quotient of
    # the exact division would carry a 129-digit coefficient.
    assert max(len(str(abs(digits))) for digits in hidden) == 129
    with pytest.raises(ValidationError):
        _product_request(((amplified,),), ((one_over_successor,),), variables)


def test_symbolic_matrix_product_rejects_aggregate_expansion_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each bounded cell may still push the whole product over the total."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)

    def seven_terms() -> RationalFunction:
        return _rf(
            variables,
            *((1, 1, (exponent,)) for exponent in range(6, -1, -1)),
        )

    left = tuple(tuple(seven_terms() for _ in range(2)) for _ in range(8))
    right = tuple((seven_terms(),) for _ in range(2))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    with pytest.raises(ValidationError):
        _product_request(left, right, variables)


def test_symbolic_matrix_product_rejects_aggregate_canonical_support_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expansion work inside its budget can still overflow the result type."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)

    paired = tuple(
        _rf(
            variables,
            (1, 1, (exponent + 1,)),
            (1, 1, (exponent,)),
        )
        for exponent in (6, 4, 2, 0)
    )
    left = tuple(paired for _ in range(8))
    one = _rf(variables, (1, 1, (0,)))
    right = tuple((one, one, one, one, one, one, one, one) for _ in range(4))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # The work accumulator charges exactly 512 scalar products, but every
    # canonical cell collects eight distinct monomials plus one unit
    # denominator, which the returned SymbolicMatrix would reject.
    with pytest.raises(ValidationError):
        _product_request(left, right, variables)


def test_symbolic_matrix_product_admits_boundary_aggregate_canonical_support() -> None:
    """Exactly 512 canonical result terms still execute and return typed."""

    variables = ("x",)

    def sixty_three_terms() -> RationalFunction:
        return _rf(
            variables,
            *((1, 1, (exponent,)) for exponent in range(62, -1, -1)),
        )

    dense = sixty_three_terms()
    one = _rf(variables, (1, 1, (0,)))
    left = ((dense,),)
    right = (tuple(one for _ in range(8)),)
    product = compute_symbolic_matrix_product(_product_request(left, right, variables))
    assert product == SymbolicMatrix(
        variables=variables,
        entries=(tuple(dense for _ in range(8)),),
    )


def test_symbolic_matrix_product_charges_retained_denominator_in_scalar_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied multi-term denominator keeps every term in the result."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)
    dense_denominator = tuple((1, 1, (exponent,)) for exponent in range(63, -1, -1))
    dense_inverse = _rf(
        variables,
        (1, 1, (0,)),
        denominator=dense_denominator,
    )
    two = _rf(variables, (2, 1, (0,)))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # Every scaled copy retains one numerator term plus all 64 denominator
    # terms, so eight cells carry 520 canonical result terms.
    with pytest.raises(ValidationError):
        _product_request(((dense_inverse,),), ((two,) * 8,), variables)


def test_symbolic_matrix_product_keeps_dense_denominator_under_scalar_copy() -> None:
    """Scalar scaling retains the operand's full denominator verbatim."""

    variables = ("x",)
    dense_denominator = tuple((1, 1, (exponent,)) for exponent in range(63, -1, -1))
    dense_inverse = _rf(
        variables,
        (1, 1, (0,)),
        denominator=dense_denominator,
    )
    two = _rf(variables, (2, 1, (0,)))
    product = compute_symbolic_matrix_product(
        _product_request(((dense_inverse,),), ((two,),), variables)
    )
    assert product.entries == (
        (
            _rf(
                variables,
                (2, 1, (0,)),
                denominator=dense_denominator,
            ),
        ),
    )


def test_symbolic_matrix_product_ignores_unit_denominator_coefficient_digits() -> None:
    """Unit denominators add no height, matching their inert expansion work."""

    variables: tuple[str, ...] = ()
    large = _constant(10**120)
    one = _constant(1)
    left = ((large,) * 8,)
    right = tuple((one,) for _ in range(8))
    product = compute_symbolic_matrix_product(_product_request(left, right, variables))
    assert product.entries == ((_constant(8 * 10**120),),)


def test_symbolic_matrix_product_rejects_coefficient_growth_before_kernel() -> None:
    digits = "9" * 65
    large = _rf((), (int(digits), 1, ()))
    with pytest.raises(ValidationError):
        _product_request(((large,),), ((large,),), ())


def test_symbolic_matrix_product_rejects_exponent_growth_before_kernel() -> None:
    variables = ("x",)
    x_to_64 = _rf(variables, (1, 1, (64,)))
    x = _rf(variables, (1, 1, (1,)))
    with pytest.raises(ValidationError):
        _product_request(((x_to_64,),), ((x,),), variables)


def test_symbolic_matrix_product_admits_scalar_identity_with_large_coefficients() -> (
    None
):
    variables = ("x",)
    large = _rf(variables, (10**63, 1, (1,)), (1, 1, (0,)))
    one = _rf(variables, (1, 1, (0,)))
    product = compute_symbolic_matrix_product(
        _product_request(((large,),), ((one,),), variables)
    )
    assert product.entries == ((large,),)


def test_symbolic_matrix_product_admits_partial_exponent_collisions() -> None:
    """Only products sharing an exponent are charged to one coefficient."""

    variables = ("x",)
    scaled = _rf(
        variables,
        (10**40, 1, (2,)),
        (10**40, 1, (1,)),
        (10**40, 1, (0,)),
    )
    shifted = _rf(variables, (1, 1, (1,)), (1, 1, (0,)))
    request = _product_request(((scaled,),), ((shifted,),), variables)
    product = compute_symbolic_matrix_product(request)
    assert product.entries == (
        (
            _rf(
                variables,
                (10**40, 1, (3,)),
                (2 * 10**40, 1, (2,)),
                (2 * 10**40, 1, (1,)),
                (10**40, 1, (0,)),
            ),
        ),
    )


def test_symbolic_matrix_product_admits_integral_coefficient_collisions() -> None:
    """Integral colliding products add only an addition carry digit."""

    variables = ("x",)
    scaled = _rf(variables, (10**63, 1, (1,)), (10**63, 1, (0,)))
    shifted = _rf(variables, (1, 1, (1,)), (1, 1, (0,)))
    request = _product_request(((scaled,),), ((shifted,),), variables)
    product = compute_symbolic_matrix_product(request)
    assert product.entries == (
        (
            _rf(
                variables,
                (10**63, 1, (2,)),
                (2 * 10**63, 1, (1,)),
                (10**63, 1, (0,)),
            ),
        ),
    )


def test_symbolic_matrix_product_still_charges_rational_collisions() -> None:
    """Rational colliding products still charge unrelated denominator growth."""

    variables = ("x",)
    scaled = _rf(variables, (10**63, 97, (1,)), (10**63, 97, (0,)))
    shifted = _rf(variables, (1, 1, (1,)), (1, 89, (0,)))
    with pytest.raises(ValidationError):
        _product_request(((scaled,),), ((shifted,),), variables)


def test_symbolic_matrix_product_admits_sparse_support_without_cancellation() -> None:
    """Polynomial cells cannot densify, so sparse support stays bounded."""

    variables = ("x", "y")
    monomial = _rf(variables, (1, 1, (16, 16)))
    product = compute_symbolic_matrix_product(
        _product_request(((monomial,),), ((monomial,),), variables)
    )
    assert product.entries == ((_rf(variables, (1, 1, (32, 32))),),)


def test_symbolic_matrix_product_ignores_zero_partner_for_support_path() -> None:
    """A zero partner drops its pair before the support path is chosen."""

    variables = ("x", "y")
    monomial = _rf(variables, (1, 1, (16, 16)))
    inverse_x = _rf(variables, (1, 1, (0, 0)), denominator=((1, 1, (1, 0)),))
    zero = _rf(variables)
    product = compute_symbolic_matrix_product(
        _product_request(((monomial, inverse_x),), ((monomial,), (zero,)), variables)
    )
    assert product.entries == ((_rf(variables, (1, 1, (32, 32))),),)


def test_symbolic_matrix_product_keeps_monomial_denominator_support() -> None:
    """A monomial common denominator cannot densify under cancellation."""

    variables = ("x", "y")
    inverse = _rf(
        variables,
        (1, 1, (0, 0)),
        denominator=((1, 1, (16, 16)),),
    )
    one = _rf(variables, (1, 1, (0, 0)))
    product = compute_symbolic_matrix_product(
        _product_request(((inverse,),), ((one,),), variables)
    )
    assert product.entries == ((inverse,),)


def test_symbolic_matrix_product_rejects_mixed_denominators_without_coefficient_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairs over different non-monomial denominators still lack a bound."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x", "y")
    inverse_x = _rf(
        variables,
        (1, 1, (0, 0)),
        denominator=((1, 1, (16, 0)), (1, 1, (0, 16))),
    )
    inverse_shifted_x = _rf(
        variables,
        (1, 1, (0, 0)),
        denominator=((1, 1, (15, 0)), (1, 1, (0, 16))),
    )
    one = _rf(variables, (1, 1, (0, 0)))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    with pytest.raises(ValidationError):
        _product_request(((inverse_x, inverse_shifted_x),), ((one,), (one,)), variables)


def test_symbolic_matrix_product_admits_shared_denominator_sums() -> None:
    """Equal shared denominators admit the exact collected numerator sum."""

    variables = ("x",)
    inverse_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    one = _rf(variables, (1, 1, (0,)))
    product = compute_symbolic_matrix_product(
        _product_request(
            ((inverse_successor, inverse_successor),), ((one,), (one,)), variables
        )
    )
    assert product.entries == (
        (
            _rf(
                variables,
                (2, 1, (0,)),
                denominator=((1, 1, (1,)), (1, 1, (0,))),
            ),
        ),
    )


def test_symbolic_matrix_product_admits_single_pair_over_shared_denominator() -> None:
    """One non-scalar pair with a multi-term denominator gets exact bounds."""

    variables = ("x",)
    x_over_successor = _rf(
        variables,
        (1, 1, (1,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    x = _rf(variables, (1, 1, (1,)))
    product = compute_symbolic_matrix_product(
        _product_request(((x_over_successor,),), ((x,),), variables)
    )
    assert product.entries == (
        (
            _rf(
                variables,
                (1, 1, (2,)),
                denominator=((1, 1, (1,)), (1, 1, (0,))),
            ),
        ),
    )


def test_symbolic_matrix_product_admits_shared_denominators_on_both_sides() -> None:
    """Shared left and right denominators form the exact monic quotient."""

    variables = ("x",)
    inverse_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    x_over_shifted = _rf(
        variables,
        (1, 1, (1,)),
        denominator=((1, 1, (1,)), (2, 1, (0,))),
    )
    product = compute_symbolic_matrix_product(
        _product_request(((inverse_successor,),), ((x_over_shifted,),), variables)
    )
    assert product.entries == (
        (
            _rf(
                variables,
                (1, 1, (1,)),
                denominator=((1, 1, (2,)), (3, 1, (1,)), (2, 1, (0,))),
            ),
        ),
    )


def test_symbolic_matrix_product_admits_swapped_pair_denominators() -> None:
    """Pairs whose product denominators coincide admit the exact sum.

    [1/(x+1), 1/(x+2)] * [[1/(x+2)], [1/(x+1)]] has per-side denominators
    that differ, yet both pairs carry the same product denominator
    (x+1)(x+2), so the exact collected value 2/((x+1)(x+2)) is admitted.
    """

    variables = ("x",)
    inverse_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    inverse_shifted = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (2, 1, (0,))),
    )
    product = compute_symbolic_matrix_product(
        _product_request(
            ((inverse_successor, inverse_shifted),),
            ((inverse_shifted,), (inverse_successor,)),
            variables,
        )
    )
    assert product.entries == (
        (
            _rf(
                variables,
                (2, 1, (0,)),
                denominator=((1, 1, (2,)), (3, 1, (1,)), (2, 1, (0,))),
            ),
        ),
    )


def test_symbolic_matrix_product_rejects_mismatched_pair_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairs whose product denominators differ keep the conservative rejection."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)
    inverse_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    inverse_shifted = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (2, 1, (0,))),
    )
    one = _rf(variables, (1, 1, (0,)))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # The pair products (x+1)(x+2) and (x+1)^2 differ, so no shared product
    # denominator fixes the canonical value.
    with pytest.raises(ValidationError):
        _product_request(
            ((one, one),),
            ((inverse_successor,), (inverse_shifted,)),
            variables,
        )


def test_symbolic_matrix_product_admits_shared_denominator_zero_sum() -> None:
    """Collected numerators that cancel completely return canonical zero."""

    variables = ("x",)
    successor = ((1, 1, (1,)), (1, 1, (0,)))
    forward = _rf(variables, (1, 1, (1,)), denominator=successor)
    backward = _rf(variables, (-1, 1, (1,)), denominator=successor)
    one = _rf(variables, (1, 1, (0,)))
    product = compute_symbolic_matrix_product(
        _product_request(((forward, backward),), ((one,), (one,)), variables)
    )
    assert product.entries == ((_rf(variables),),)


def test_symbolic_matrix_product_rejects_shared_denominator_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collected numerator sharing a factor keeps the conservative rejection."""

    import jacobian.math.matrices.symbolic as symbolic

    variables = ("x",)
    successor = ((1, 1, (1,)), (1, 1, (0,)))
    forward = _rf(variables, (1, 1, (1,)), denominator=successor)
    inverse = _rf(variables, (1, 1, (0,)), denominator=successor)
    one = _rf(variables, (1, 1, (0,)))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symbolic multiplication kernel ran during admission")

    monkeypatch.setattr(symbolic, "symbolic_matrix_multiply", fail_if_called)
    # The collected numerator (x + 1) shares the retained denominator's own
    # factor, so the reduced quotient again lacks a pre-execution bound.
    with pytest.raises(ValidationError):
        _product_request(((forward, inverse),), ((one,), (one,)), variables)


def test_symbolic_matrix_product_rejects_shared_denominators_without_result_budget() -> (
    None
):
    """A common-denominator product beyond the result budget stays rejected."""

    variables = ("x",)
    dense_inverse = _rf(
        variables,
        (1, 1, (0,)),
        denominator=tuple((1, 1, (exponent,)) for exponent in range(16, -1, -1)),
    )
    # The squared 17-term denominator needs up to 289 canonical terms, which
    # exceeds the 256-term exact result budget.
    with pytest.raises(ValidationError):
        _product_request(((dense_inverse,),), ((dense_inverse,),), variables)


def test_symbolic_matrix_product_rejects_aggregate_expansion_before_exact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense shared-denominator overflow is rejected without SymPy work.

    Every cell of this 8x8 product reuses one identical product denominator,
    so each cell is eligible for the exact shared-denominator fallback; the
    projection pass must still charge all 64 cells' raw expansion totals and
    reject before any cell invokes SymPy conversion.
    """

    import jacobian.math.matrices.symbolic._models as symbolic_models

    variables = ("x",)
    inverse_successor = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)), (1, 1, (0,))),
    )
    left = tuple(tuple(inverse_successor for _ in range(8)) for _ in range(8))
    right = tuple(tuple(inverse_successor for _ in range(8)) for _ in range(8))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact shared-denominator fallback ran during projection")

    monkeypatch.setattr(
        symbolic_models, "_shared_common_denominator_bounds", fail_if_called
    )
    with pytest.raises(ValidationError):
        _product_request(left, right, variables)


def test_symbolic_matrix_product_admits_boundary_shared_projection() -> None:
    """A projected expansion of exactly 512 still executes the exact fallback."""

    variables = ("x", "y")
    x_numerator = tuple((1, 1, (exponent, 0)) for exponent in range(15, -1, -1))
    y_denominator = tuple((1, 1, (0, exponent)) for exponent in range(15, -1, -1))
    left = ((_rf(variables, *x_numerator, denominator=y_denominator),),)
    right = ((_rf(variables, *x_numerator, denominator=y_denominator),),)
    product = compute_symbolic_matrix_product(_product_request(left, right, variables))
    counts = tuple(min(exponent + 1, 31 - exponent) for exponent in range(30, -1, -1))
    numerator = tuple(
        (count, 1, (exponent, 0))
        for count, exponent in zip(counts, range(30, -1, -1), strict=True)
    )
    denominator = tuple(
        (count, 1, (0, exponent))
        for count, exponent in zip(counts, range(30, -1, -1), strict=True)
    )
    assert product.entries == ((_rf(variables, *numerator, denominator=denominator),),)


def test_symbolic_matrix_product_rejects_projection_above_aggregate_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One extra denominator term pushes the projection past 512 pre-SymPy."""

    import jacobian.math.matrices.symbolic._models as symbolic_models

    variables = ("x", "y")
    x_numerator = tuple((1, 1, (exponent, 0)) for exponent in range(15, -1, -1))
    y_denominator = tuple((1, 1, (0, exponent)) for exponent in range(15, -1, -1))
    long_y_denominator = tuple((1, 1, (0, exponent)) for exponent in range(16, -1, -1))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact shared-denominator fallback ran during projection")

    monkeypatch.setattr(
        symbolic_models, "_shared_common_denominator_bounds", fail_if_called
    )
    # The projected expansion charges 16*16 numerator products plus 16*17
    # pair denominator terms: 528 raw terms against the 512-term budget.
    with pytest.raises(ValidationError):
        _product_request(
            ((_rf(variables, *x_numerator, denominator=y_denominator),),),
            ((_rf(variables, *x_numerator, denominator=long_y_denominator),),),
            variables,
        )


def test_symbolic_matrix_product_admits_identity_times_rational_entry() -> None:
    """Identity multiplication leaves the canonical operand unchanged."""

    variables = ("x", "y")
    rational = _rf(
        variables,
        (1, 1, (16, 16)),
        (1, 1, (0, 0)),
        denominator=((1, 1, (1, 0)), (1, 1, (0, 0))),
    )
    one = _rf(variables, (1, 1, (0, 0)))
    product = compute_symbolic_matrix_product(
        _product_request(((rational,),), ((one,),), variables)
    )
    assert product.entries == ((rational,),)


def test_symbolic_matrix_product_admits_dense_constant_matrices() -> None:
    """Scalar-product work counts the real products, not inert denominators."""

    variables: tuple[str, ...] = ()
    two = _constant(2)
    one = _constant(1)
    left = tuple(tuple(two for _ in range(8)) for _ in range(8))
    right = tuple(tuple(one for _ in range(8)) for _ in range(8))
    product = compute_symbolic_matrix_product(_product_request(left, right, variables))
    assert product.entries == tuple(
        tuple(_constant(16) for _ in range(8)) for _ in range(8)
    )


def test_symbolic_matrix_product_admits_scalar_constant_times_rational_entry() -> None:
    """A nonzero rational scalar scales the operand without cancellation."""

    variables = ("x", "y")
    rational = _rf(
        variables,
        (1, 1, (16, 16)),
        (1, 1, (0, 0)),
        denominator=((1, 1, (1, 0)), (1, 1, (0, 0))),
    )
    two = _rf(variables, (2, 1, (0, 0)))
    scaled = _rf(
        variables,
        (2, 1, (16, 16)),
        (2, 1, (0, 0)),
        denominator=((1, 1, (1, 0)), (1, 1, (0, 0))),
    )
    assert compute_symbolic_matrix_product(
        _product_request(((rational,),), ((two,),), variables)
    ).entries == ((scaled,),)
    assert compute_symbolic_matrix_product(
        _product_request(((two,),), ((rational,),), variables)
    ).entries == ((scaled,),)


def test_symbolic_matrix_product_admits_rational_scalar_at_coefficient_boundary() -> (
    None
):
    """Scalar height adds to the operand height under the shared budget."""

    variables = ("x",)
    base = _rf(variables, (10**63, 1, (1,)), (10**63, 1, (0,)))
    scalar = _rf(variables, (10**63, 97, (0,)))
    product = compute_symbolic_matrix_product(
        _product_request(((base,),), ((scalar,),), variables)
    )
    assert product.entries == (
        (_rf(variables, (10**126, 97, (1,)), (10**126, 97, (0,))),),
    )


def test_symbolic_matrix_product_bounds_scalar_constant_coefficients() -> None:
    """Scalar scaling is charged against the coefficient budget."""

    variables = ("x",)
    base = _rf(variables, (10**63, 1, (0,)))
    oversized_scalar = _rf(variables, (10**65, 1, (0,)))
    with pytest.raises(ValidationError):
        _product_request(((base,),), ((oversized_scalar,),), variables)


def test_rational_function_entries_use_the_advertised_field() -> None:
    variables = ("x",)
    inverse_x = _rf(
        variables,
        (1, 1, (0,)),
        denominator=((1, 1, (1,)),),
    )
    result = compute_symbolic_determinant(
        _determinant_request(((inverse_x,),), variables)
    )
    assert result.determinant == inverse_x


def test_symbolic_characteristic_polynomial_of_constant_matrix() -> None:
    request = _characteristic_request(
        ((_constant(1), _constant(2)), (_constant(3), _constant(4))), ()
    )
    result = compute_symbolic_characteristic_polynomial(request)
    assert isinstance(result, SymbolicCharacteristicPolynomialResult)
    assert result.degree == 2
    assert result.coefficients_descending == (
        _constant(1),
        _constant(-5),
        _constant(-2),
    )


def test_symbolic_characteristic_polynomial_of_zero_matrix() -> None:
    zero = _constant(0)
    result = compute_symbolic_characteristic_polynomial(
        _characteristic_request(((zero, zero), (zero, zero)), ())
    )
    assert result.coefficients_descending == (
        _constant(1),
        _constant(0),
        _constant(0),
    )


def test_symbolic_eigenvalues_of_constant_matrix() -> None:
    request = _characteristic_request(
        ((_constant(1), _constant(2)), (_constant(3), _constant(4))), ()
    )
    result = compute_symbolic_eigenvalues(request)
    assert isinstance(result, SymbolicEigenvaluesResult)
    assert len(result.eigenvalues or ()) == 2
    assert result.multiplicities == (1, 1)


@pytest.mark.parametrize(
    "operation_id",
    (
        "matrix.symbolic.determinant.compute",
        "matrix.symbolic.characteristic_polynomial.compute",
        "matrix.symbolic.eigenvalues.compute",
    ),
)
def test_square_only_descriptors_reject_rectangular_input(operation_id: str) -> None:
    operation = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    with pytest.raises(ValidationError):
        operation.request_type.model_validate(
            {
                "matrix": {
                    "variables": [],
                    "entries": [[_constant(1).model_dump(), _constant(2).model_dump()]],
                }
            }
        )


def test_rectangular_matrix_is_accepted_only_by_rank_contract() -> None:
    result = compute_symbolic_rank(
        _request(((_constant(1), _constant(2), _constant(3)),), ())
    )
    assert result.rank == 1


def test_symbolic_matrix_dimensions_are_bounded_at_eight() -> None:
    entries = tuple(tuple(_constant(1) for _ in range(8)) for _ in range(8))
    _square_request(entries, ())
    with pytest.raises(ValidationError):
        _request((tuple(_constant(1) for _ in range(9)),), ())


def test_symbolic_descriptors_publish_operation_specific_boundaries() -> None:
    request_types = {tool.operation_id: tool.request_type for tool in TOOLS}
    assert request_types == {
        "matrix.symbolic.determinant.compute": SymbolicDeterminantRequest,
        "matrix.symbolic.rank.compute": SymbolicMatrixRequest,
        "matrix.symbolic.characteristic_polynomial.compute": SymbolicCharacteristicPolynomialRequest,
        "matrix.symbolic.eigenvalues.compute": SymbolicCharacteristicPolynomialRequest,
        "matrix.symbolic.linear_system.solve": SymbolicLinearSystemRequest,
        "matrix.symbolic.multiply.compute": SymbolicMatrixProductRequest,
    }


def test_symbolic_native_api_exports_matrix_value_and_product() -> None:
    import jacobian.math.matrices.symbolic as symbolic

    assert tuple(symbolic.__all__) == (
        "SymbolicMatrix",
        "symbolic_characteristic_polynomial",
        "symbolic_determinant",
        "symbolic_eigenvalues",
        "symbolic_linear_system_solve",
        "symbolic_matrix_multiply",
        "symbolic_rank",
    )


def test_matrix_rejects_nonrectangular_mismatched_and_invalid_axes() -> None:
    a = _variable(("a",), 0)
    with pytest.raises(ValidationError):
        SymbolicMatrix(variables=("a",), entries=((a, a), (a,)))
    with pytest.raises(ValidationError):
        SymbolicMatrix(variables=("b",), entries=((a,),))
    with pytest.raises(ValidationError):
        SymbolicMatrix.model_validate(
            {"variables": [""], "entries": [[a.model_dump()]]}
        )


def test_public_matrix_entries_reject_expression_strings_without_execution(
    tmp_path,
) -> None:
    marker = tmp_path / "sympy-evaluated"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(ValidationError):
        SymbolicMatrixRequest.model_validate(
            {"matrix": {"variables": [], "entries": [[payload]]}}
        )
    assert not marker.exists()


def test_noncanonical_rational_functions_are_rejected() -> None:
    variables = ("x",)
    with pytest.raises(ValidationError):
        _rf(
            variables,
            (1, 1, (0,)),
            denominator=((2, 1, (1,)),),
        )
    with pytest.raises(ValidationError):
        _rf(
            variables,
            (1, 1, (1,)),
            denominator=((1, 1, (1,)),),
        )


def test_symbolic_eigenvalues_returns_polynomial_for_unrepresentable_roots() -> None:
    variables = ("a",)
    zero = _rf(variables)
    one = _rf(variables, (1, 1, (0,)))
    a = _variable(variables, 0)
    request = _characteristic_request(
        (
            (zero, zero, zero, zero, a),
            (one, zero, zero, zero, one),
            (zero, one, zero, zero, zero),
            (zero, zero, one, zero, zero),
            (zero, zero, zero, one, zero),
        ),
        variables,
    )
    result = compute_symbolic_eigenvalues(request)
    assert result.representation == "ROOTS_BY_POLYNOMIAL"
    assert result.degree == 5
    assert result.characteristic_polynomial is not None
    assert result.eigenvalues is None


def test_symbolic_eigenvalues_explicit_for_representable_roots() -> None:
    request = _characteristic_request(
        ((_constant(1), _constant(2)), (_constant(3), _constant(4))), ()
    )
    result = compute_symbolic_eigenvalues(request)
    assert result.representation == "EXPLICIT_ROOTS"
    assert result.eigenvalues is not None
    assert result.characteristic_polynomial is None
