"""Tests for polynomial vector calculus operations."""

from jacobian.math.polynomial_vector_calc._models import (
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    VectorFieldRequest,
)
from jacobian.math.polynomial_vector_calc._operations import (
    compute_curl,
    compute_directional_derivative,
    compute_divergence,
    compute_gradient,
    compute_laplacian,
)
from jacobian.math.polynomial_vector_calc._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial_field.scalar.gradient.compute",
        "polynomial_field.scalar.laplacian.compute",
        "polynomial_field.scalar.directional_derivative.compute",
        "polynomial_field.vector.divergence.compute",
        "polynomial_field.vector.curl.compute",
    }


def test_gradient_of_x2_y2() -> None:
    request = ScalarFieldRequest(variables=("x", "y"), polynomial="x**2 + y**2")
    result = compute_gradient(request)
    assert result.components == ("2*x", "2*y")


def test_laplacian_of_x3_y3() -> None:
    request = ScalarFieldRequest(variables=("x", "y"), polynomial="x**3 + y**3")
    result = compute_laplacian(request)
    assert result.result == "6*x + 6*y"


def test_directional_derivative() -> None:
    request = DirectionalDerivativeRequest(
        variables=("x", "y"),
        polynomial="x**2 + y**2",
        direction=("1", "1"),
    )
    result = compute_directional_derivative(request)
    assert "2*x" in result.result and "2*y" in result.result


def test_divergence() -> None:
    request = VectorFieldRequest(variables=("x", "y"), components=("x**2", "y**2"))
    result = compute_divergence(request)
    assert result.result == "2*x + 2*y"


def test_curl_3d() -> None:
    request = VectorFieldRequest(variables=("x", "y", "z"), components=("y", "0", "0"))
    result = compute_curl(request)
    assert result.components == ("0", "0", "-1")


def test_curl_requires_3d() -> None:
    import pytest

    request = VectorFieldRequest(variables=("x", "y"), components=("x", "y"))
    with pytest.raises(ValueError, match="3D"):
        compute_curl(request)


class TestVectorFieldValidation:
    """Tests for the strengthened VectorFieldRequest contract."""

    def test_component_count_mismatch_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="one component per variable"):
            VectorFieldRequest(variables=("x", "y"), components=("x**2",))

    def test_too_many_components_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="one component per variable"):
            VectorFieldRequest(variables=("x", "y"), components=("x", "y", "x*y"))

    def test_correct_component_count_accepted(self) -> None:
        vf = VectorFieldRequest(variables=("x", "y", "z"), components=("x", "y", "z"))
        assert len(vf.components) == 3

    def test_non_polynomial_rejected(self) -> None:
        """The parser should reject non-polynomial expressions like 1/x."""
        import pytest

        with pytest.raises((ValueError, TypeError)):
            compute_gradient(ScalarFieldRequest(variables=("x",), polynomial="1/x"))

    def test_foreign_symbol_rejected(self) -> None:
        """The parser should reject symbols not in the declared variables."""
        import pytest

        with pytest.raises(ValueError, match="undeclared symbols"):
            compute_gradient(ScalarFieldRequest(variables=("x",), polynomial="y + x"))

    def test_distinct_variables_required(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="distinct"):
            ScalarFieldRequest(variables=("x", "x"), polynomial="x")


def test_gradient_single_variable() -> None:
    """Gradient of x**2 in one variable should give (2*x,)."""
    request = ScalarFieldRequest(variables=("x",), polynomial="x**2")
    result = compute_gradient(request)
    assert result.components == ("2*x",)


def test_divergence_single_variable() -> None:
    """Divergence of a 1D vector field (x**2) should give 2*x."""
    request = VectorFieldRequest(variables=("x",), components=("x**2",))
    result = compute_divergence(request)
    assert result.result == "2*x"


def test_laplacian_single_variable() -> None:
    """Laplacian of x**3 in one variable should give 6*x."""
    request = ScalarFieldRequest(variables=("x",), polynomial="x**3")
    result = compute_laplacian(request)
    assert result.result == "6*x"


def test_directional_derivative_single_variable() -> None:
    """Directional derivative of x**2 along (2,) should give 4*x."""
    request = DirectionalDerivativeRequest(
        variables=("x",), polynomial="x**2", direction=("2",)
    )
    result = compute_directional_derivative(request)
    assert result.result == "4*x"
