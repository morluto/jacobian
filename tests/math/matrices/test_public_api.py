import pytest
import sympy
from hypothesis import given
from hypothesis import strategies as st

from jacobian.math import matrices


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
    with pytest.raises(ValueError, match="square"):
        matrices.trace(sympy.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_rref_rejects_non_matrix_inputs() -> None:
    with pytest.raises(TypeError, match="SymPy MatrixBase"):
        matrices.rref([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_rref_rejects_oversized_matrices() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        matrices.rref(sympy.zeros(33, 1))


def test_matrix_input_errors_are_stable() -> None:
    with pytest.raises(TypeError, match="SymPy MatrixBase"):
        matrices.trace([[1]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        matrices.inverse(sympy.Matrix([[1, 2]]))
    with pytest.raises(ValueError, match="singular"):
        matrices.inverse(sympy.zeros(2))
    with pytest.raises(ValueError, match="exact"):
        matrices.trace(sympy.Matrix([[1.5]]))
    nested_float = sympy.Add(sympy.Float("0.1"), sympy.Rational(1, 3), evaluate=False)
    with pytest.raises(ValueError, match="exact"):
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
