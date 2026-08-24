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
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_characteristic_polynomial,
    compute_symbolic_determinant,
    compute_symbolic_eigenvalues,
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

    with pytest.raises(ValidationError, match="result term budget"):
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
    with pytest.raises(ValidationError, match="square"):
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
    with pytest.raises(ValidationError, match="8"):
        _request((tuple(_constant(1) for _ in range(9)),), ())


def test_symbolic_descriptors_publish_operation_specific_boundaries() -> None:
    request_types = {tool.operation_id: tool.request_type for tool in TOOLS}
    assert request_types == {
        "matrix.symbolic.determinant.compute": SymbolicDeterminantRequest,
        "matrix.symbolic.rank.compute": SymbolicMatrixRequest,
        "matrix.symbolic.characteristic_polynomial.compute": SymbolicCharacteristicPolynomialRequest,
        "matrix.symbolic.eigenvalues.compute": SymbolicCharacteristicPolynomialRequest,
        "matrix.symbolic.linear_system.solve": SymbolicLinearSystemRequest,
    }


def test_matrix_rejects_nonrectangular_mismatched_and_invalid_axes() -> None:
    a = _variable(("a",), 0)
    with pytest.raises(ValidationError, match="same length"):
        SymbolicMatrix(variables=("a",), entries=((a, a), (a,)))
    with pytest.raises(ValidationError, match="declared ordered field"):
        SymbolicMatrix(variables=("b",), entries=((a,),))
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
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
    with pytest.raises(ValidationError, match="monic"):
        _rf(
            variables,
            (1, 1, (0,)),
            denominator=((2, 1, (1,)),),
        )
    with pytest.raises(ValidationError, match="coprime"):
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
