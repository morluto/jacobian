"""Tests for polynomial map operations."""

from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    EvalRequest,
    JacobianRequest,
    RationalPolynomialExpr,
    VariablePoint,
)
from jacobian.math.polynomials.maps._operations import (
    compose_polynomials,
    compute_jacobian,
    evaluate_polynomial,
)


class TestEvaluate:
    def test_simple(self):
        req = EvalRequest(
            polynomial=RationalPolynomialExpr(expression="x**2 + 2*y"),
            point=VariablePoint(
                variables=("x", "y"),
                values=(
                    {"num": "3", "den": "1"},
                    {"num": "1", "den": "1"},
                ),
            ),
        )
        result = evaluate_polynomial(req)
        assert result.value == "11"  # 9 + 2 = 11

    def test_zero(self):
        req = EvalRequest(
            polynomial=RationalPolynomialExpr(expression="x**2"),
            point=VariablePoint(
                variables=("x",),
                values=({"num": "0", "den": "1"},),
            ),
        )
        result = evaluate_polynomial(req)
        assert result.value == "0"

    def test_rejects_rational_function(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="polynomial"):
            EvalRequest(
                polynomial=RationalPolynomialExpr(expression="1/x"),
                point=VariablePoint(
                    variables=("x",),
                    values=({"num": "1", "den": "1"},),
                ),
            )

    def test_rejects_irrational_coefficient(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="rational coefficients"):
            RationalPolynomialExpr(expression="pi*x")

    def test_rejects_incomplete_substitution(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="every free variable"):
            EvalRequest(
                polynomial=RationalPolynomialExpr(expression="x + y"),
                point=VariablePoint(
                    variables=("x",),
                    values=({"num": "1", "den": "1"},),
                ),
            )


class TestJacobian:
    def test_simple(self):
        req = JacobianRequest(
            input_variables=("x", "y"),
            output_polynomials=(
                RationalPolynomialExpr(expression="x**2"),
                RationalPolynomialExpr(expression="y**2"),
            ),
        )
        result = compute_jacobian(req)
        assert result.n_inputs == 2
        assert result.n_outputs == 2
        assert result.entries[0] == "2*x"  # d(x^2)/dx
        assert result.entries[3] == "2*y"  # d(y^2)/dy


class TestComposition:
    def test_simple(self):
        req = CompositionRequest(
            outer=RationalPolynomialExpr(expression="x**2"),
            inner=RationalPolynomialExpr(expression="x + 1"),
            inner_variable="x",
            outer_variable="x",
        )
        result = compose_polynomials(req)
        # (x+1)^2 = x^2 + 2*x + 1
        from sympy import expand, sympify

        result_expr = expand(sympify(result.expression))
        assert result_expr == expand(sympify("(x + 1)**2"))
