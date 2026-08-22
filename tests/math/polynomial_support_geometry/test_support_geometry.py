"""Tests for polynomial support geometry operations (#1797)."""

import pytest
from pydantic import ValidationError

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
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(
    terms: tuple[dict, ...], variables: tuple[str, ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {"variables": list(variables), "polynomial": {"terms": list(terms)}}
    )


def _term(coeff: str, exponents: list[int]) -> dict:
    return {"coefficient": {"num": coeff, "den": "1"}, "exponents": exponents}


_XY_TERMS = (
    _term("1", [2, 0]),
    _term("1", [1, 1]),
    _term("1", [0, 2]),
)

VARS = ("x", "y")


class TestSupport:
    def test_nonzero_support(self) -> None:
        result = compute_support(
            SupportRequest(polynomial=_polynomial(_XY_TERMS, VARS))
        )
        assert not result.is_zero
        assert result.term_count == 3
        assert result.coordinate_min == (0, 0)
        assert result.coordinate_max == (2, 2)
        assert result.total_degree_min == 2
        assert result.total_degree_max == 2

    def test_zero_support(self) -> None:
        result = compute_support(SupportRequest(polynomial=_polynomial((), VARS)))
        assert result.is_zero
        assert result.term_count == 0

    def test_accepts_canonical_polynomial_value(self) -> None:
        """A serialized producer result validates unchanged as request input."""
        request = SupportRequest(polynomial=_polynomial(_XY_TERMS, VARS))
        revalidated = SupportRequest.model_validate(request.model_dump())
        assert revalidated.polynomial == request.polynomial


class TestNewtonPolytope:
    def test_newton_of_xy(self) -> None:
        result = compute_newton_polytope(
            NewtonPolytopeRequest(polynomial=_polynomial(_XY_TERMS, VARS))
        )
        assert not result.is_zero
        assert result.ambient_dimension == 2
        # Support {(2,0), (1,1), (0,2)} is collinear on x+y=2: exactly the
        # two endpoints are vertices.
        assert set(result.vertices) == {(2, 0), (0, 2)}
        assert result.nonextreme == ((1, 1),)
        assert result.affine_dimension == 1

    def test_triangle_support_all_vertices(self) -> None:
        terms = (_term("1", [2, 2]), _term("1", [2, 0]), _term("1", [0, 2]))
        result = compute_newton_polytope(
            NewtonPolytopeRequest(polynomial=_polynomial(terms, VARS))
        )
        assert set(result.vertices) == {(2, 0), (0, 2), (2, 2)}
        assert result.affine_dimension == 2

    def test_retains_ordered_variables(self) -> None:
        """Identical exponents over different rings stay distinguishable."""
        poly_xz = _polynomial(_XY_TERMS, ("x", "z"))
        result = compute_newton_polytope(NewtonPolytopeRequest(polynomial=poly_xz))
        assert result.variables == ("x", "z")

    def test_interior_point_is_nonextreme(self) -> None:
        terms = (
            _term("1", [4, 0]),
            _term("1", [1, 1]),
            _term("1", [0, 4]),
            _term("1", [0, 0]),
        )
        result = compute_newton_polytope(
            NewtonPolytopeRequest(polynomial=_polynomial(terms, VARS))
        )
        assert set(result.vertices) == {(0, 0), (4, 0), (0, 4)}
        assert result.nonextreme == ((1, 1),)

    def test_skew_vertex_requires_general_direction(self) -> None:
        """(0,2) is a genuine vertex of [(0,0),(0,1),(0,2),(1,5)] even though
        no axis-aligned or small-coordinate direction exposes it; the exact
        extremality kernel must still certify it."""
        terms = (
            _term("1", [1, 5]),
            _term("1", [0, 2]),
            _term("1", [0, 1]),
            _term("1", [0, 0]),
        )
        result = compute_newton_polytope(
            NewtonPolytopeRequest(polynomial=_polynomial(terms, VARS))
        )
        assert set(result.vertices) == {(0, 0), (0, 2), (1, 5)}
        assert result.nonextreme == ((0, 1),)
        assert result.affine_dimension == 2

    def test_term_bound_rejects_infeasible_scan(self) -> None:
        """The Newton operation's term budget is narrower than the canonical one."""
        pairs = sorted(((i % 11, i // 11) for i in range(97)), reverse=True)
        many = tuple(_term("1", list(pair)) for pair in pairs)
        assert len(many) == 97
        with pytest.raises(ValueError, match="96"):
            NewtonPolytopeRequest(polynomial=_polynomial(many, VARS))


class TestWeightProfile:
    def test_weight_profile_uniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 1])
        )
        # All exponents have weight 2, so min weight is 2
        assert result.minimum_weight == 2
        assert len(result.minimizing_exponents) == 3

        # There should be one weight layer (all at weight 2)
        assert len(result.weight_layers) == 1

    def test_weight_profile_nonuniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 0])
        )
        # Weights: (2,0)->2, (0,2)->0, (1,1)->1
        # min weight is 0 at (0,2)
        assert result.minimum_weight == 0
        assert result.minimizing_exponents == ((0, 2),)

    def test_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="weight vector length"):
            WeightProfileRequest(
                polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 1, 1]
            )

    def test_zero_polynomial_rejected(self) -> None:
        """The empty support has no minimum; the zero polynomial is inadmissible."""
        with pytest.raises(ValueError, match="zero polynomial"):
            WeightProfileRequest(polynomial=_polynomial((), VARS), weight=[1, 1])
        with pytest.raises(ValueError, match="zero polynomial"):
            InitialFormRequest(polynomial=_polynomial((), VARS), weight=[1, 1])


class TestInitialForm:
    def test_initial_form_uniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 1])
        )
        # All terms at min weight 2, so initial form is the whole polynomial
        assert len(result.initial_form.polynomial.terms) == 3

    def test_initial_form_nonuniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 0])
        )
        # Minimum weight 0 at (0,2), initial form is y^2 over the same ring
        assert result.initial_form.variables == VARS
        term = result.initial_form.polynomial.terms[0]
        assert tuple(term.exponents) == (0, 2)
        assert term.coefficient.num == "1"

    def test_initial_form_with_coeffs(self) -> None:
        terms = (_term("3", [2, 0]), _term("5", [0, 2]))
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(terms, VARS), weight=[1, 0])
        )
        # Minimum weight 0 at (0,2), initial form is 5*y^2
        term = result.initial_form.polynomial.terms[0]
        assert tuple(term.exponents) == (0, 2)
        assert term.coefficient.num == "5"

    def test_initial_form_binds_to_canonical_value(self) -> None:
        """A serialized producer result revalidates; a mismatched weight
        fails against the retained source."""
        from jacobian.math.polynomial_support_geometry.values import (
            PolynomialFaceData,
        )

        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 0])
        )
        revalidated = PolynomialFaceData.model_validate(result.model_dump())
        assert revalidated.initial_form.variables == VARS

        payload = result.model_dump()
        payload["weight"] = [1, 1]
        with pytest.raises(ValidationError, match="minimum-weight face"):
            PolynomialFaceData.model_validate(payload)

    def test_initial_form_composes_as_polynomial_input(self) -> None:
        """The returned canonical value feeds another request unchanged."""
        first = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=[1, 0])
        )
        follow_up = compute_support(SupportRequest(polynomial=first.initial_form))
        assert follow_up.term_count == 1


class TestTransportableBounds:
    def test_huge_weight_component_rejected(self) -> None:
        """Derived weights must stay inside the interoperable JSON range."""
        big = 9007199254740991
        with pytest.raises(ValueError, match="transportable"):
            WeightProfileRequest(
                polynomial=_polynomial(_XY_TERMS, VARS), weight=[big, 1]
            )

    def test_initial_form_output_growth_bounded(self) -> None:
        """A zero weight makes every term minimal; oversized sources are
        rejected so the doubled serialization cannot breach the envelope."""
        many = tuple(
            _term("1", list(pair))
            for pair in sorted(((i % 32, i // 32) for i in range(1025)), reverse=True)
        )
        with pytest.raises(ValueError, match="limited to 1024"):
            InitialFormRequest(
                polynomial=_polynomial(many, VARS),
                weight=[0, 0],
            )
