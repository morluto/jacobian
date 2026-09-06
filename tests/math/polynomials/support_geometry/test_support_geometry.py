"""Tests for polynomial support geometry operations."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.support_geometry._models import (
    InitialFormRequest,
    NewtonPolytopeRequest,
    SupportRequest,
    WeightProfileRequest,
)
from jacobian.math.polynomials.support_geometry.operations import (
    exponent_support,
    initial_form,
    newton_polytope,
    verify_polynomial_support,
    weight_profile,
)
from jacobian.math.polynomials.support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)
from jacobian.math.polynomials.values import RationalPolynomial


def compute_support(request: SupportRequest) -> PolynomialSupport:
    return exponent_support(request.polynomial)


def compute_newton_polytope(request: NewtonPolytopeRequest) -> NewtonPolytope:
    return newton_polytope(request.polynomial)


def compute_weight_profile(request: WeightProfileRequest) -> PolynomialWeightProfile:
    return weight_profile(request.polynomial, request.weight)


def compute_initial_form(request: InitialFormRequest) -> PolynomialFaceData:
    return initial_form(request.polynomial, request.weight)


class _CanonicalRationalPayload(TypedDict):
    num: int
    den: int


class _RationalTermPayload(TypedDict):
    coefficient: _CanonicalRationalPayload
    exponents: list[int]


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


@contextmanager
def raises_domain_code(code: str) -> Iterator[None]:
    with pytest.raises(OperationDomainValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == code


def _polynomial(
    terms: tuple[_RationalTermPayload, ...], variables: tuple[str, ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {"variables": list(variables), "polynomial": {"terms": list(terms)}}
    )


def _term(coeff: int, exponents: list[int]) -> _RationalTermPayload:
    return {"coefficient": {"num": coeff, "den": 1}, "exponents": exponents}


_XY_TERMS = (
    _term(1, [2, 0]),
    _term(1, [1, 1]),
    _term(1, [0, 2]),
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
        source = _polynomial((), VARS)
        result = compute_support(SupportRequest(polynomial=source))
        assert result.is_zero
        assert result.term_count == 0
        assert result.polynomial is source
        assert result.polynomial.domain == "QQ"
        assert result.polynomial.variables == VARS

    def test_accepts_canonical_polynomial_value(self) -> None:
        """A serialized producer result validates unchanged as request input."""
        request = SupportRequest(polynomial=_polynomial(_XY_TERMS, VARS))
        revalidated = SupportRequest.model_validate(request.model_dump())
        assert revalidated.polynomial == request.polynomial

    def test_result_retains_source_for_json_composition(self) -> None:
        """Support output carries one canonical source into another operation."""
        source = _polynomial(_XY_TERMS, VARS)
        result = compute_support(SupportRequest(polynomial=source))

        assert result.polynomial is source
        assert result.polynomial.domain == "QQ"
        assert result.polynomial.variables == VARS
        payload_json = result.model_dump_json()
        payload = json.loads(payload_json)
        assert "variables" not in payload
        assert "coefficients" not in payload
        restored = type(result).model_validate_json(payload_json)
        assert restored == result
        assert verify_polynomial_support(restored)

        follow_up = SupportRequest.model_validate_json(
            json.dumps({"polynomial": payload["polynomial"]})
        )
        assert follow_up.polynomial == source
        assert compute_support(follow_up).exponents == result.exponents


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
        terms = (_term(1, [2, 2]), _term(1, [2, 0]), _term(1, [0, 2]))
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
            _term(1, [4, 0]),
            _term(1, [1, 1]),
            _term(1, [0, 4]),
            _term(1, [0, 0]),
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
            _term(1, [1, 5]),
            _term(1, [0, 2]),
            _term(1, [0, 1]),
            _term(1, [0, 0]),
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
        many = tuple(_term(1, list(pair)) for pair in pairs)
        assert len(many) == 97
        request = NewtonPolytopeRequest(polynomial=_polynomial(many, VARS))
        with raises_domain_code(
            "polynomial_support_geometry.newton_term_count_exceeded"
        ):
            compute_newton_polytope(request)


class TestWeightProfile:
    def test_weight_profile_uniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 1))
        )
        # All exponents have weight 2, so min weight is 2
        assert result.minimum_weight == 2
        assert len(result.minimizing_exponents) == 3

        # There should be one weight layer (all at weight 2)
        assert len(result.weight_layers) == 1

    def test_weight_profile_nonuniform(self) -> None:
        result = compute_weight_profile(
            WeightProfileRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 0))
        )
        # Weights: (2,0)->2, (0,2)->0, (1,1)->1
        # min weight is 0 at (0,2)
        assert result.minimum_weight == 0
        assert result.minimizing_exponents == ((0, 2),)

    def test_dimension_mismatch(self) -> None:
        request = WeightProfileRequest(
            polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 1, 1)
        )
        with raises_domain_code(
            "polynomial_support_geometry.weight_dimension_mismatch"
        ):
            compute_weight_profile(request)

    def test_zero_polynomial_rejected(self) -> None:
        """The empty support has no minimum; the zero polynomial is inadmissible."""
        weight_request = WeightProfileRequest(
            polynomial=_polynomial((), VARS), weight=(1, 1)
        )
        initial_request = InitialFormRequest(
            polynomial=_polynomial((), VARS), weight=(1, 1)
        )
        with raises_domain_code("polynomial_support_geometry.zero_weight_profile"):
            compute_weight_profile(weight_request)
        with raises_domain_code("polynomial_support_geometry.zero_weight_profile"):
            compute_initial_form(initial_request)


class TestInitialForm:
    def test_initial_form_uniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 1))
        )
        # All terms at min weight 2, so initial form is the whole polynomial
        assert len(result.initial_form.polynomial.terms) == 3

    def test_initial_form_nonuniform(self) -> None:
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 0))
        )
        # Minimum weight 0 at (0,2), initial form is y^2 over the same ring
        assert result.initial_form.variables == VARS
        term = result.initial_form.polynomial.terms[0]
        assert tuple(term.exponents) == (0, 2)
        assert term.coefficient.num == 1

    def test_initial_form_with_coeffs(self) -> None:
        terms = (_term(3, [2, 0]), _term(5, [0, 2]))
        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(terms, VARS), weight=(1, 0))
        )
        # Minimum weight 0 at (0,2), initial form is 5*y^2
        term = result.initial_form.polynomial.terms[0]
        assert tuple(term.exponents) == (0, 2)
        assert term.coefficient.num == 5

    def test_initial_form_binds_to_canonical_value(self) -> None:
        """The result round-trips as a canonical value."""
        from jacobian.math.polynomials.support_geometry.values import (
            PolynomialFaceData,
        )

        result = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 0))
        )
        revalidated = PolynomialFaceData.model_validate(result.model_dump())
        assert revalidated.initial_form.variables == VARS

    def test_initial_form_composes_as_polynomial_input(self) -> None:
        """The returned canonical value feeds another request unchanged."""
        first = compute_initial_form(
            InitialFormRequest(polynomial=_polynomial(_XY_TERMS, VARS), weight=(1, 0))
        )
        follow_up = compute_support(SupportRequest(polynomial=first.initial_form))
        assert follow_up.term_count == 1


class TestTransportableBounds:
    def test_huge_weight_component_rejected(self) -> None:
        """Derived weights must stay inside the interoperable JSON range."""
        big = 9007199254740991
        request = WeightProfileRequest(
            polynomial=_polynomial(_XY_TERMS, VARS), weight=(big, 1)
        )
        with raises_domain_code(
            "polynomial_support_geometry.weight_component_out_of_range"
        ):
            compute_weight_profile(request)

    def test_initial_form_output_growth_bounded(self) -> None:
        """A zero weight makes every term minimal; oversized sources are
        rejected so the doubled serialization cannot breach the envelope."""
        many = tuple(
            _term(1, list(pair))
            for pair in sorted(((i % 32, i // 32) for i in range(1025)), reverse=True)
        )
        request = InitialFormRequest(
            polynomial=_polynomial(many, VARS),
            weight=(0, 0),
        )
        with raises_domain_code(
            "polynomial_support_geometry.weighted_term_count_exceeded"
        ):
            compute_initial_form(request)


class TestNewtonInvariants:
    def test_forged_polytope_rejected(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

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
        from jacobian.math.polynomials.support_geometry import (
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
        from jacobian.math.polynomials.support_geometry import (
            initial_form,
            weight_profile,
        )

        zero = _polynomial((), VARS)
        with pytest.raises(OperationDomainValidationError, match="nonzero"):
            weight_profile(zero, (1, 1))
        with pytest.raises(OperationDomainValidationError, match="nonzero"):
            initial_form(zero, (1, 1))
        nonzero = _polynomial(_XY_TERMS, VARS)
        with pytest.raises(
            OperationDomainValidationError, match="weight vector length"
        ):
            weight_profile(nonzero, (1,))
        with pytest.raises(
            OperationDomainValidationError, match="weight vector length"
        ):
            initial_form(nonzero, (1,))


class TestSupportCrossFieldValidation:
    def test_forged_support_rejected(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import (
            PolynomialSupport,
        )

        with raises_code("term_count_mismatch"):
            PolynomialSupport(
                polynomial=_polynomial(_XY_TERMS, VARS),
                is_zero=False,
                term_count=1,
                exponents=((2, 0), (0, 2)),
                total_degree_min=2,
                total_degree_max=2,
            )

    def test_source_and_claim_forgery_are_rejected(self) -> None:
        """The explicit verifier checks source and derived fields together."""
        source = _polynomial(_XY_TERMS, VARS)
        claim = compute_support(SupportRequest(polynomial=source))

        source_forgery = claim.model_dump()
        source_forgery["polynomial"]["polynomial"]["terms"][2]["exponents"] = [0, 3]
        forged_source = type(claim).model_validate(source_forgery)
        assert not verify_polynomial_support(forged_source)

        claim_forgery = claim.model_dump()
        claim_forgery["coordinate_max"] = [3, 2]
        forged_claim = type(claim).model_validate(claim_forgery)
        assert not verify_polynomial_support(forged_claim)

    def test_verifier_rejects_oversized_constructed_source(self) -> None:
        """Verification has a fixed source scan bound even for trusted bypasses."""
        from jacobian.math.polynomials.support_geometry.values import PolynomialSupport
        from jacobian.math.polynomials.values import SparseRationalPolynomial

        source = _polynomial(_XY_TERMS, VARS)
        oversized = RationalPolynomial.model_construct(
            domain="QQ",
            variables=VARS,
            polynomial=SparseRationalPolynomial.model_construct(
                terms=source.polynomial.terms * 1366
            ),
        )
        claim = PolynomialSupport.model_construct(
            polynomial=oversized,
            is_zero=True,
            term_count=0,
            exponents=(),
            coordinate_min=(),
            coordinate_max=(),
            total_degree_min=None,
            total_degree_max=None,
        )
        assert not verify_polynomial_support(claim)

    def test_verifier_rejects_noncanonical_constructed_sources(self) -> None:
        """Source claims cannot bypass the canonical polynomial envelope."""
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomials.values import (
            RationalPolynomialTerm,
            SparseRationalPolynomial,
        )

        source = _polynomial(_XY_TERMS, VARS)

        def malformed_claim(
            polynomial: RationalPolynomial,
            exponents: tuple[tuple[int, ...], ...] = ((2, 0), (1, 1), (0, 2)),
        ) -> PolynomialSupport:
            return PolynomialSupport.model_construct(
                polynomial=polynomial,
                is_zero=False,
                term_count=3,
                exponents=exponents,
                coordinate_min=(0, 0),
                coordinate_max=(2, 2),
                total_degree_min=2,
                total_degree_max=2,
            )

        def source_with(
            *,
            domain: object = "QQ",
            variables: object = VARS,
            terms: object = source.polynomial.terms,
        ) -> RationalPolynomial:
            return RationalPolynomial.model_construct(
                domain=domain,
                variables=variables,
                polynomial=SparseRationalPolynomial.model_construct(terms=terms),
            )

        reversed_terms = tuple(reversed(source.polynomial.terms))
        noncanonical_coefficient = (
            RationalPolynomialTerm.model_construct(
                coefficient=CanonicalRational.model_construct(num="2", den="2"),
                exponents=(2, 0),
            ),
            *source.polynomial.terms[1:],
        )
        cases = (
            malformed_claim(source_with(domain="ZZ")),
            malformed_claim(source_with(variables=["x", "y"])),
            malformed_claim(source_with(variables=("x", "x"))),
            malformed_claim(source_with(variables=("x", "bad-name"))),
            malformed_claim(source_with(terms=list(source.polynomial.terms))),
            malformed_claim(
                source_with(terms=reversed_terms),
                exponents=tuple(term.exponents for term in reversed_terms),
            ),
            malformed_claim(source_with(terms=noncanonical_coefficient)),
        )
        assert all(not verify_polynomial_support(case) for case in cases)

    def test_verifier_rejects_hostile_tuple_subclasses(self) -> None:
        """Verifier carrier checks do not iterate untrusted tuple subclasses."""

        class EvilTuple(tuple[object, ...]):
            def __iter__(self) -> Iterator[object]:
                raise RuntimeError("hostile tuple iterator")

        source = _polynomial(_XY_TERMS, VARS)
        claim = compute_support(SupportRequest(polynomial=source))
        for field in ("exponents", "coordinate_min", "coordinate_max"):
            forged = claim.model_copy(update={field: EvilTuple(getattr(claim, field))})
            assert not verify_polynomial_support(forged)

        hostile_variables = source.model_copy(
            update={"variables": EvilTuple(source.variables)}
        )
        assert not verify_polynomial_support(
            claim.model_copy(update={"polynomial": hostile_variables})
        )

        hostile_terms = source.polynomial.model_copy(
            update={"terms": EvilTuple(source.polynomial.terms)}
        )
        assert not verify_polynomial_support(
            claim.model_copy(
                update={
                    "polynomial": source.model_copy(
                        update={"polynomial": hostile_terms}
                    )
                }
            )
        )

        hostile_term = source.polynomial.terms[0].model_copy(
            update={"exponents": EvilTuple(source.polynomial.terms[0].exponents)}
        )
        hostile_sparse = source.polynomial.model_copy(
            update={"terms": (hostile_term, *source.polynomial.terms[1:])}
        )
        assert not verify_polynomial_support(
            claim.model_copy(
                update={
                    "polynomial": source.model_copy(
                        update={"polynomial": hostile_sparse}
                    )
                }
            )
        )

    def test_verifier_rejects_hostile_rational_subclasses(self) -> None:
        """Verifier scalar checks do not invoke subclass-overridden methods."""
        from jacobian._exact import CanonicalRational

        class EvilRational(CanonicalRational):
            def as_fraction(self) -> Fraction:
                raise RuntimeError("hostile rational method")

        source = _polynomial(_XY_TERMS, VARS)
        hostile_term = source.polynomial.terms[0].model_copy(
            update={"coefficient": EvilRational(num=1, den=1)}
        )
        hostile_sparse = source.polynomial.model_copy(
            update={"terms": (hostile_term, *source.polynomial.terms[1:])}
        )
        claim = compute_support(SupportRequest(polynomial=source))
        hostile_source = source.model_copy(update={"polynomial": hostile_sparse})
        assert not verify_polynomial_support(
            claim.model_copy(update={"polynomial": hostile_source})
        )


class TestNewtonReplay:
    def test_ragged_exponent_widths_rejected(self) -> None:
        """Exponents narrower than the ambient dimension are rejected with a
        ValidationError instead of leaking an IndexError from the replay."""
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

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
        from jacobian.math.polynomials.support_geometry.values import (
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


class TestSupportValueInvariants:
    def test_forged_support_claim_is_checked_against_source(self) -> None:
        """A decoded support keeps its canonical polynomial source."""
        from jacobian.math.polynomials.support_geometry.values import PolynomialSupport

        payload = _polynomial(_XY_TERMS, VARS).model_dump(mode="json")
        claim = PolynomialSupport.model_validate_json(
            json.dumps(
                {
                    "polynomial": payload,
                    "is_zero": True,
                    "term_count": 0,
                    "exponents": [],
                }
            )
        )
        assert not verify_polynomial_support(claim)

    def test_nonzero_newton_result_must_retain_support(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

        with raises_code("nonzero_newton_missing_support"):
            NewtonPolytope.model_validate(
                {
                    "is_zero": False,
                    "variables": ["x"],
                    "ambient_dimension": 1,
                    "affine_dimension": 0,
                }
            )

    def test_native_export_list_excludes_wire_handlers(self) -> None:
        import jacobian.math.polynomials.support_geometry as package

        assert not any(name.startswith("compute_") for name in package.__all__)
        assert "exponent_support" in package.__all__

    def test_duplicate_support_exponents_rejected(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import PolynomialSupport

        with raises_code("exponents_not_distinct"):
            PolynomialSupport(
                polynomial=_polynomial((_term(1, [1]), _term(2, [0])), ("x",)),
                is_zero=False,
                term_count=2,
                exponents=((1,), (1,)),
                total_degree_min=1,
                total_degree_max=1,
            )

    def test_zero_support_cannot_carry_coordinate_extrema(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import PolynomialSupport

        with raises_code("zero_support_coordinate_extrema"):
            PolynomialSupport(
                polynomial=_polynomial((), ("x",)),
                is_zero=True,
                term_count=0,
                coordinate_min=(3,),
                coordinate_max=(5,),
            )

    def test_zero_newton_polytope_requires_zero_affine_dimension(self) -> None:
        """A deserialized zero result must not contradict its own empty
        support: the empty polytope has affine dimension zero."""
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

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
        from jacobian.math.polynomials.support_geometry.values import PolynomialSupport

        for exponent in (-1, 40000):
            with raises_code("exponents_out_of_domain"):
                PolynomialSupport(
                    polynomial=_polynomial((_term(1, [0]),), ("x",)),
                    is_zero=False,
                    term_count=1,
                    exponents=((exponent,),),
                    coordinate_min=(exponent,),
                    coordinate_max=(exponent,),
                    total_degree_min=exponent,
                    total_degree_max=exponent,
                )

    def test_newton_exponents_outside_canonical_domain_rejected(self) -> None:
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

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
        with raises_pydantic_code("too_short"):
            _polynomial((), ())

    def test_duplicate_newton_points_rejected(self) -> None:
        """Retained vertices, nonextreme points, and support are sets of
        distinct exponents; duplicates would make the tuple fields
        noncanonical before any set-based check runs."""
        from jacobian.math.polynomials.support_geometry.values import NewtonPolytope

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
