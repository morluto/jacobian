"""Tests for polynomial support geometry operations (#1797)."""

import pytest

from jacobian.math.polynomial_support_geometry._models import (
    InitialFormRequest,
    NewtonPolytopeRequest,
    SupportRequest,
    WeightProfileRequest,
)
from jacobian.math.polynomial_support_geometry.operations import (
    compute_initial_form,
    compute_newton_polytope,
    compute_support,
    compute_weight_profile,
)


def _xy_terms():
    return (
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [2, 0]},
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 2]},
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1, 1]},
    )


VARS = ("x", "y")


class TestSupport:
    def test_nonzero_support(self) -> None:
        result = compute_support(SupportRequest(terms=_xy_terms(), variables=VARS))
        assert not result.is_zero
        assert result.term_count == 3
        assert result.coordinate_min == (0, 0)
        assert result.coordinate_max == (2, 2)
        assert result.total_degree_min == 2
        assert result.total_degree_max == 2

    def test_zero_support(self) -> None:
        terms = (
            {"coefficient": {"num": "0", "den": "1"}, "exponents": [0, 0]},
        )
        result = compute_support(SupportRequest(terms=terms, variables=VARS))
        assert result.is_zero
        assert result.term_count == 0


class TestNewtonPolytope:
    def test_newton_of_xy(self) -> None:
        result = compute_newton_polytope(
            NewtonPolytopeRequest(terms=_xy_terms(), variables=VARS)
        )
        assert not result.is_zero
        assert result.ambient_dimension == 2
        # x^2 + xy + y^2 has support {(2,0), (0,2), (1,1)}
        # All three are vertices (triangle)
        assert len(result.vertices) == 2
        assert result.affine_dimension == 1

    def test_zero_newton(self) -> None:
        terms = (
            {"coefficient": {"num": "0", "den": "1"}, "exponents": [0, 0]},
        )
        result = compute_newton_polytope(
            NewtonPolytopeRequest(terms=terms, variables=VARS)
        )
        assert result.is_zero


class TestWeightProfile:
    def test_weight_profile_uniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(terms=_xy_terms(), variables=VARS, weight=[1, 1])
        )
        # All exponents have weight 2, so min weight is 2
        assert result.minimum_weight == 2
        assert len(result.minimizing_exponents) == 3

        # There should be one weight layer (all at weight 2)
        assert len(result.weight_layers) == 1

    def test_weight_profile_nonuniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(terms=_xy_terms(), variables=VARS, weight=[1, 0])
        )
        # Weights: (2,0)->2, (0,2)->0, (1,1)->1
        # min weight is 0 at (0,2)
        assert result.minimum_weight == 0
        assert result.minimizing_exponents == ((0, 2),)

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(Exception, match="dimension|weight"):
            WeightProfileRequest(
                terms=_xy_terms(), variables=VARS, weight=[1, 1, 1]
            )


class TestInitialForm:
    def test_initial_form_uniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(terms=_xy_terms(), variables=VARS, weight=[1, 1])
        )
        # All terms at min weight 2, so initial form is the whole polynomial
        assert len(result.face_exponents) == 3

    def test_initial_form_nonuniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(terms=_xy_terms(), variables=VARS, weight=[1, 0])
        )
        # Minimum weight 0 at (0,2), initial form is y^2
        assert result.face_exponents == ((0, 2),)
        assert int(result.face_coefficients[0].num) == 1

    def test_initial_form_with_coeffs(self) -> None:
        terms = (
            {"coefficient": {"num": "3", "den": "1"}, "exponents": [2, 0]},
            {"coefficient": {"num": "5", "den": "1"}, "exponents": [0, 2]},
        )
        result = compute_initial_form(
            InitialFormRequest(terms=terms, variables=VARS, weight=[1, 0])
        )
        # Minimum weight 0 at (0,2), initial form is 5*y^2
        assert result.face_exponents == ((0, 2),)
        assert int(result.face_coefficients[0].num) == 5
