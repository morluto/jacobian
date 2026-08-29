import importlib
from fractions import Fraction

import pytest
import sympy
from hypothesis import given
from hypothesis import strategies as st

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math import matrices
from jacobian.math.matrices._operation_models import MatrixDeterminantRequest
from jacobian.math.matrices._tools import compute_determinant
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    rational_matrix_from_fractions,
)


def test_orphan_combinatorial_matrix_models_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.math.combinatorial_matrices._models")


def test_exact_matrix_operations() -> None:
    source = sympy.Matrix([[1, 2], [3, 4]])
    assert matrices.adjugate(source) == sympy.Matrix([[4, -2], [-3, 1]])
    assert matrices.characteristic_polynomial(source, "lambda").all_coeffs() == [
        1,
        -5,
        -2,
    ]
    assert matrices.determinant(source) == -2
    assert matrices.rank(source) == (2, (0, 1))
    assert matrices.inverse(source) == sympy.Matrix(
        [[-2, 1], [sympy.Rational(3, 2), sympy.Rational(-1, 2)]]
    )
    assert matrices.trace(source) == 5
    assert matrices.multiply(source, sympy.eye(2)) == source
    assert matrices.smith_normal_form(sympy.eye(2)) == sympy.eye(2)
    solution, parameters = matrices.solve_linear_system(
        sympy.eye(2), sympy.Matrix([3, 4])
    )
    assert solution == sympy.Matrix([3, 4])
    assert parameters.rows == 0
    reduced, pivots = matrices.rref(sympy.Matrix([[1, 2], [2, 4]]))
    assert reduced == sympy.Matrix([[1, 2], [0, 0]])
    assert pivots == (0,)


def test_characteristic_polynomial_preserves_exact_algebraic_inputs() -> None:
    source = sympy.Matrix([[sympy.sqrt(2), 0], [0, 1]])

    assert matrices.characteristic_polynomial(source, "lambda").all_coeffs() == [
        1,
        -1 - sympy.sqrt(2),
        sympy.sqrt(2),
    ]


def test_native_determinant_preserves_exact_algebraic_inputs() -> None:
    source = sympy.Matrix([[sympy.sqrt(2), 0], [0, 1]])

    assert matrices.determinant(source) == sympy.sqrt(2)


def test_native_determinant_shares_admission_and_flint_kernel_above_order_64() -> None:
    order = 65
    entries = tuple(
        tuple(
            Fraction(1, row + 1) if row == column else Fraction(0)
            for column in range(order)
        )
        for row in range(order)
    )
    source = sympy.diag(*[sympy.Rational(1, row + 1) for row in range(order)])

    native = matrices.determinant(source)
    wire = compute_determinant(
        MatrixDeterminantRequest(matrix=rational_matrix_from_fractions(entries))
    )

    assert native == sympy.Rational(wire.determinant.num, wire.determinant.den)
    assert native == sympy.Rational(1, sympy.factorial(order))


def test_native_determinant_rejects_input_scalars_above_the_shared_digit_bound() -> (
    None
):
    source = sympy.diag(10**256, *[1] * 64)

    with pytest.raises(OperationDomainValidationError, match="256 decimal digits"):
        matrices.determinant(source)


def test_multiply_preserves_exact_algebraic_inputs() -> None:
    source = sympy.Matrix([[sympy.sqrt(2), 0], [0, 1]])

    assert matrices.multiply(source, sympy.eye(2)) == source


def test_multiply_admits_algebraic_inputs_at_shared_axis() -> None:
    source = sympy.diag(*([sympy.sqrt(2)] * 32))

    assert matrices.multiply(source, sympy.eye(32)) == source


def test_multiply_rejects_algebraic_inputs_above_shared_axis() -> None:
    source = sympy.diag(*([sympy.sqrt(2)] * 33))

    with pytest.raises(ValueError, match="between 1 and 32"):
        matrices.multiply(source, sympy.eye(33))


def test_rref_of_identity_is_identity_with_all_pivots() -> None:
    reduced, pivots = matrices.rref(sympy.eye(3))
    assert reduced == sympy.eye(3)
    assert pivots == (0, 1, 2)


def test_rref_of_zero_matrix_has_no_pivots() -> None:
    reduced, pivots = matrices.rref(sympy.zeros(2, 3))
    assert reduced == sympy.zeros(2, 3)
    assert pivots == ()


def test_rref_preserves_rational_entries() -> None:
    matrix = sympy.Matrix([[sympy.Rational(1, 2), 1], [1, 2]])
    reduced, pivots = matrices.rref(matrix)
    assert reduced == sympy.Matrix([[1, 2], [0, 0]])
    assert pivots == (0,)


def test_inverse_round_trip_recovers_original() -> None:
    source = sympy.Matrix([[2, 1], [5, 3]])
    recovered = matrices.inverse(matrices.inverse(source))
    assert recovered == source


def test_inverse_of_3x3_integer_matrix() -> None:
    source = sympy.Matrix([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    expected = sympy.Matrix([[1, 0, -2], [0, 1, -3], [0, 0, 1]])
    assert matrices.inverse(source) == expected


def test_trace_of_identity_equals_dimension() -> None:
    assert matrices.trace(sympy.eye(4)) == 4


def test_trace_of_rational_matrix() -> None:
    source = sympy.Matrix(
        [
            [sympy.Rational(1, 3), sympy.Rational(2, 3)],
            [sympy.Rational(1, 2), sympy.Rational(1, 4)],
        ]
    )
    assert matrices.trace(source) == sympy.Rational(1, 3) + sympy.Rational(1, 4)


def test_trace_requires_square_matrix() -> None:
    with pytest.raises(ValueError):
        matrices.trace(sympy.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_rref_rejects_non_matrix_inputs() -> None:
    with pytest.raises(TypeError):
        matrices.rref([[1, 2], [3, 4]])


def test_rref_rejects_oversized_matrices() -> None:
    with pytest.raises(ValueError):
        matrices.rref(sympy.zeros(MAX_EXACT_LINEAR_MATRIX_AXIS + 1, 1))


def test_rref_admits_an_axis_above_the_square_computation_dimension() -> None:
    reduced, pivots = matrices.rref(sympy.ones(33, 1))
    assert pivots == (0,)
    assert reduced[0, 0] == 1
    assert all(reduced[row, 0] == 0 for row in range(1, 33))


def test_matrix_input_errors_are_stable() -> None:
    with pytest.raises(TypeError):
        matrices.trace([[1]])
    with pytest.raises(ValueError):
        matrices.inverse(sympy.Matrix([[1, 2]]))
    with pytest.raises(ValueError):
        matrices.inverse(sympy.zeros(2))
    with pytest.raises(ValueError):
        matrices.trace(sympy.Matrix([[1.5]]))
    nested_float = sympy.Add(sympy.Float("0.1"), sympy.Rational(1, 3), evaluate=False)
    with pytest.raises(ValueError):
        matrices.trace(sympy.Matrix([[nested_float]]))


@given(
    st.lists(
        st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=2),
        min_size=2,
        max_size=2,
    )
)
def test_rref_pivots_are_strictly_increasing(matrix_data: list[list[int]]) -> None:
    _, pivots = matrices.rref(sympy.Matrix(matrix_data))
    assert pivots == tuple(sorted(pivots))
    assert len(pivots) == len(set(pivots))


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the matrices public API."""
    expected = (
        "SmithNormalForm",
        "adjugate",
        "characteristic_polynomial",
        "determinant",
        "inverse",
        "kronecker_product",
        "multiply",
        "partial_trace",
        "permanent",
        "rank",
        "rref",
        "smith_normal_form",
        "solve_linear_system",
        "trace",
    )
    assert tuple(matrices.__all__) == expected
    assert len(matrices.__all__) == len(set(matrices.__all__))
    assert all(not name.startswith("_") for name in matrices.__all__)
    assert all(hasattr(matrices, name) for name in matrices.__all__)
