"""Tests for polynomial support geometry operations (#1797)."""

from collections.abc import Iterator
from contextlib import contextmanager

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


@contextmanager
def raises_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == f"polynomial_support_geometry.{code}"


@contextmanager
def raises_pydantic_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == code


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
        with raises_code("initial_form_mismatch"):
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
        with raises_code("weight_component_out_of_range"):
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
        with raises_code("initial_form_term_count_exceeded"):
            InitialFormRequest(
                polynomial=_polynomial(many, VARS),
                weight=[0, 0],
            )


class TestNewtonInvariants:
    def test_forged_polytope_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        with raises_code("ambient_dimension_mismatch"):
            NewtonPolytope(
                is_zero=False,
                variables=("x", "y"),
                ambient_dimension=3,
                affine_dimension=1,
                vertices=((2, 0),),
                nonextreme=(),
                all_support_exponents=((2, 0),),
            )
        with raises_code("newton_support_partition_mismatch"):
            NewtonPolytope(
                is_zero=False,
                variables=("x", "y"),
                ambient_dimension=2,
                affine_dimension=1,
                vertices=((2, 0),),
                nonextreme=((0, 2),),
                all_support_exponents=((2, 0), (1, 1)),
            )

    def test_empty_support_has_no_degree_extrema(self) -> None:

        result = compute_support(SupportRequest(polynomial=_polynomial((), VARS)))
        assert result.is_zero
        assert result.total_degree_min is None
        assert result.total_degree_max is None


class TestNativeSurface:
    def test_domain_value_kernels(self) -> None:
        """Native exports accept canonical polynomial values."""
        from jacobian.math.polynomial_support_geometry import (
            exponent_support,
            newton_polytope,
        )

        polynomial = _polynomial(_XY_TERMS, VARS)
        support = exponent_support(polynomial)
        assert support.term_count == 3
        polytope = newton_polytope(polynomial)
        assert polytope.vertices is not None

    def test_native_weighted_functions_keep_domain_validation(self) -> None:
        """Native weighted calls reject the zero polynomial and mismatched
        weight dimensions instead of leaking host exceptions."""
        from jacobian.math.polynomial_support_geometry import (
            initial_form,
            weight_profile,
        )

        zero = _polynomial((), VARS)
        with pytest.raises(ValueError, match="nonzero"):
            weight_profile(zero, (1, 1))
        with pytest.raises(ValueError, match="nonzero"):
            initial_form(zero, (1, 1))
        nonzero = _polynomial(_XY_TERMS, VARS)
        with pytest.raises(ValueError, match="weight vector length"):
            weight_profile(nonzero, (1,))
        with pytest.raises(ValueError, match="weight vector length"):
            initial_form(nonzero, (1,))


class TestSupportCrossFieldValidation:
    def test_forged_support_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import (
            PolynomialSupport,
        )

        with raises_code("term_count_mismatch"):
            PolynomialSupport(
                is_zero=False,
                term_count=1,
                exponents=((2, 0), (0, 2)),
                coefficients=(
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ),
                variables=VARS,
                total_degree_min=2,
                total_degree_max=2,
            )


class TestNewtonReplay:
    def test_ragged_exponent_widths_rejected(self) -> None:
        """Exponents narrower than the ambient dimension are rejected with a
        ValidationError instead of leaking an IndexError from the replay."""
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        payload = {
            "is_zero": False,
            "variables": ["x", "y"],
            "ambient_dimension": 2,
            "affine_dimension": 0,
            "vertices": [[0, 0]],
            "nonextreme": [[0]],
            "all_support_exponents": [[0, 0], [0]],
        }
        with raises_code("newton_exponent_dimension_mismatch"):
            NewtonPolytope.model_validate(payload)

    def test_oversized_retained_support_rejected(self) -> None:
        """A caller-authored payload cannot bypass the admitted hull size."""
        from jacobian.math.polynomial_support_geometry.values import (
            MAX_NEWTON_TERMS,
            NewtonPolytope,
        )

        exponents = [[i] for i in range(MAX_NEWTON_TERMS + 1)]
        with pytest.raises(ValidationError):
            NewtonPolytope.model_validate(
                {
                    "is_zero": False,
                    "variables": ["x"],
                    "ambient_dimension": 1,
                    "affine_dimension": 0,
                    "vertices": exponents,
                    "nonextreme": [],
                    "all_support_exponents": exponents,
                }
            )

    def test_forged_classification_rejected(self) -> None:
        """A payload claiming an interior point is a vertex fails the exact
        hull replay."""
        from jacobian.math.polynomial_support_geometry.values import (
            NewtonPolytope,
        )

        terms = (
            _term("1", [4, 0]),
            _term("1", [1, 1]),
            _term("1", [0, 4]),
            _term("1", [0, 0]),
        )
        polynomial = _polynomial(terms, VARS)
        result = compute_newton_polytope(NewtonPolytopeRequest(polynomial=polynomial))
        payload = result.model_dump()
        good = payload["vertices"]
        payload["vertices"] = [*list(good), [1, 1]]
        payload["nonextreme"] = []
        with raises_code("newton_vertex_classification_mismatch"):
            NewtonPolytope.model_validate(payload)

    def test_wrong_affine_dimension_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        result = compute_newton_polytope(
            NewtonPolytopeRequest(polynomial=_polynomial(_XY_TERMS, VARS))
        )
        payload = result.model_dump()
        payload["affine_dimension"] = 2
        with raises_code("newton_affine_dimension_mismatch"):
            NewtonPolytope.model_validate(payload)


class TestSupportValueInvariants:
    def test_zero_coefficient_rejected_in_claimed_support(self) -> None:
        """A nonzero support cannot retain a zero-coefficient exponent."""
        from jacobian.math.polynomial_support_geometry.values import (
            PolynomialSupport,
        )

        with raises_code("zero_support_coefficient"):
            PolynomialSupport(
                is_zero=False,
                term_count=1,
                exponents=((1, 0),),
                coefficients=({"num": "0", "den": "1"},),
                variables=VARS,
                total_degree_min=1,
                total_degree_max=1,
            )

    def test_nonzero_newton_result_must_retain_support(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        with raises_code("nonzero_newton_missing_support"):
            NewtonPolytope.model_validate(
                {
                    "is_zero": False,
                    "variables": ["x"],
                    "ambient_dimension": 1,
                    "affine_dimension": 0,
                }
            )

    def test_duplicate_or_invalid_variables_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import (
            PolynomialSupport,
        )

        with pytest.raises(ValidationError):
            PolynomialSupport(
                is_zero=False,
                term_count=1,
                exponents=((1, 0),),
                coefficients=({"num": "1", "den": "1"},),
                variables=("x", "x"),
                total_degree_min=1,
                total_degree_max=1,
            )

    def test_native_export_list_excludes_wire_handlers(self) -> None:
        import jacobian.math.polynomial_support_geometry as package

        assert not any(name.startswith("compute_") for name in package.__all__)
        assert "exponent_support" in package.__all__

    def test_duplicate_support_exponents_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import PolynomialSupport

        with raises_code("exponents_not_distinct"):
            PolynomialSupport(
                is_zero=False,
                term_count=2,
                exponents=((1,), (1,)),
                coefficients=({"num": "1", "den": "1"}, {"num": "2", "den": "1"}),
                variables=("x",),
                total_degree_min=1,
                total_degree_max=1,
            )

    def test_zero_support_cannot_carry_coordinate_extrema(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import PolynomialSupport

        with raises_code("zero_support_coordinate_extrema"):
            PolynomialSupport(
                is_zero=True,
                term_count=0,
                variables=("x",),
                coordinate_min=(3,),
                coordinate_max=(5,),
            )

    def test_zero_newton_polytope_requires_zero_affine_dimension(self) -> None:
        """A deserialized zero result must not contradict its own empty
        support: the empty polytope has affine dimension zero."""
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        with raises_code("zero_newton_affine_dimension"):
            NewtonPolytope(
                is_zero=True,
                variables=("x",),
                ambient_dimension=1,
                affine_dimension=1,
            )
        assert NewtonPolytope(
            is_zero=True,
            variables=("x",),
            ambient_dimension=1,
            affine_dimension=0,
        ).is_zero

    def test_exponents_outside_canonical_domain_rejected(self) -> None:
        """Support points outside the canonical polynomial exponent domain
        cannot revalidate: the source type rejects negative exponents and
        anything above the shared representation limit."""
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomial_support_geometry.values import PolynomialSupport

        unit = CanonicalRational(num="1", den="1")
        for exponent in (-1, 40000):
            with raises_code("exponents_out_of_domain"):
                PolynomialSupport(
                    is_zero=False,
                    term_count=1,
                    exponents=((exponent,),),
                    coefficients=(unit,),
                    variables=("x",),
                    coordinate_min=(exponent,),
                    coordinate_max=(exponent,),
                    total_degree_min=exponent,
                    total_degree_max=exponent,
                )

    def test_newton_exponents_outside_canonical_domain_rejected(self) -> None:
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        for point in (-1, 40000):
            with raises_code("exponents_out_of_domain"):
                NewtonPolytope.model_validate(
                    {
                        "is_zero": False,
                        "variables": ["x"],
                        "ambient_dimension": 1,
                        "affine_dimension": 0,
                        "vertices": [[point]],
                        "nonextreme": [],
                        "all_support_exponents": [[point]],
                    }
                )

    def test_empty_variable_axis_rejected(self) -> None:
        """Every canonical polynomial ring names at least one variable."""
        from jacobian.math.polynomial_support_geometry.values import PolynomialSupport

        with raises_pydantic_code("too_short"):
            PolynomialSupport(is_zero=True, term_count=0, variables=())
        from jacobian._exact import CanonicalRational

        with raises_pydantic_code("too_short"):
            PolynomialSupport(
                is_zero=False,
                term_count=1,
                exponents=((),),
                coefficients=(CanonicalRational(num="5", den="1"),),
                variables=(),
                total_degree_min=0,
                total_degree_max=0,
            )

    def test_duplicate_newton_points_rejected(self) -> None:
        """Retained vertices, nonextreme points, and support are sets of
        distinct exponents; duplicates would make the tuple fields
        noncanonical before any set-based check runs."""
        from jacobian.math.polynomial_support_geometry.values import NewtonPolytope

        with raises_code("newton_points_not_distinct"):
            NewtonPolytope.model_validate(
                {
                    "is_zero": False,
                    "variables": ["x"],
                    "ambient_dimension": 1,
                    "affine_dimension": 1,
                    "vertices": [[0], [0], [2]],
                    "nonextreme": [[1]],
                    "all_support_exponents": [[0], [1], [2]],
                }
            )
        with raises_code("newton_points_not_distinct"):
            NewtonPolytope.model_validate(
                {
                    "is_zero": False,
                    "variables": ["x"],
                    "ambient_dimension": 1,
                    "affine_dimension": 0,
                    "vertices": [[0]],
                    "nonextreme": [],
                    "all_support_exponents": [[0], [0]],
                }
            )
