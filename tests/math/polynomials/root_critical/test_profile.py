"""Independent exact checks for root--critical distance profiles."""

from __future__ import annotations

import time

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.root_critical._models import (
    RootCriticalDistanceProfileRequest,
)
from jacobian.math.polynomials.root_critical._tools import TOOLS
from jacobian.math.polynomials.root_critical.operations import (
    root_critical_distance_profile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(*coefficients: tuple[int, int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=value, den=1),
                    exponents=(exponent,),
                )
                for exponent, value in sorted(coefficients, reverse=True)
                if value
            )
        ),
    )


def test_cubic_profile_has_complete_rows_and_unit_distances() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -1)))

    assert len(result.roots) == 3
    assert len(result.critical_points) == 1
    assert len(result.pairs) == 3
    assert all(row.distance_squared.polynomial == (1, -1) for row in result.pairs)
    assert all(row.kind == "POSITIVE" for row in result.pairs)
    assert all(row.isolating_interval.lower.as_fraction() == 1 for row in result.pairs)


def test_repeated_root_and_zero_distance_are_retained() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (1, -3), (0, 2)))

    assert [root.multiplicity for root in result.roots] == [2, 1]
    assert [critical.multiplicity for critical in result.critical_points] == [1, 1]
    zero_rows = [row for row in result.pairs if row.kind == "ZERO_DISTANCE"]
    assert len(zero_rows) == 1
    assert zero_rows[0].distance_squared.polynomial == (1, 0)


def test_nonreal_roots_use_exact_nonnegative_squared_distances() -> None:
    # (z^2+1)(z-2), with derivative roots 1 and 1/3.  The expected values
    # are obtained independently from the explicit roots ±i and 2.
    result = root_critical_distance_profile(
        _polynomial((3, 1), (2, -2), (1, 1), (0, -2))
    )

    assert len(result.pairs) == 6
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert values == {(1, -1), (1, -2), (9, -10), (9, -25)}
    assert all(row.isolating_interval.lower.as_fraction() >= 0 for row in result.pairs)
    assert any(
        root.rectangle.imaginary_lower.as_fraction() < 0 for root in result.roots
    )
    assert any(
        root.rectangle.imaginary_upper.as_fraction() > 0 for root in result.roots
    )


def test_admission_rejects_incomplete_pair_budget_before_expansion() -> None:
    request = RootCriticalDistanceProfileRequest(
        polynomial=_polynomial((3, 1), (0, -1)),
        max_pair_rows=2,
    )
    with pytest.raises(OperationResourceAdmissionError):
        TOOLS[0].run(request)


def test_constant_polynomial_is_outside_profile_domain() -> None:
    with pytest.raises(OperationDomainValidationError):
        root_critical_distance_profile(_polynomial((0, 3)))


def test_operation_is_published_with_stable_id() -> None:
    assert TOOLS[0].operation_id == "polynomial.root_critical_distance_profile.compute"


def test_request_wire_model_is_not_a_native_export() -> None:
    import jacobian.math.polynomials.root_critical as package

    assert "RootCriticalDistanceProfileRequest" not in package.__all__
    profile = root_critical_distance_profile(_polynomial((4, 1), (0, -2)))
    assert profile.pairs


def test_primitive_derivative_factor_height_is_admitted() -> None:
    scale = 10**128 - 1
    source = RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=scale, den=1),
                    exponents=(2,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=scale),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(0,),
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(source)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.root_critical.factor_coefficient_bound"
    )


def test_cube_root_cubic_profile_is_admitted() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -2)))
    assert len(result.roots) == 3
    assert len(result.critical_points) == 1
    assert len(result.pairs) == 3
    assert all(row.kind == "POSITIVE" for row in result.pairs)


def test_conjugate_distance_degree_is_admitted() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(_polynomial((4, 1), (1, 1), (0, 1)))
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.root_critical.distance_degree_bound"
    )


def test_factor_coefficients_are_canonical_decimal_integers() -> None:
    import json

    from jacobian.math.number_theory.algebraic_numbers.complex import (
        ComplexAlgebraicValue,
    )
    from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue

    profile = root_critical_distance_profile(_polynomial((3, 1), (0, -1)))
    dumped = json.loads(profile.model_dump_json())
    for root in (*dumped["roots"], *dumped["critical_points"]):
        assert "polynomial" in root["value"]
        assert "factor" not in root
    assert isinstance(profile.critical_points[0].value, RealAlgebraicValue)
    assert any(isinstance(root.value, ComplexAlgebraicValue) for root in profile.roots)


def test_cube_root_cubic_returns_exact_squared_distances() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -2)))
    assert len(result.pairs) == 3
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert values == {(1, 0, 0, -4)}


def test_quartic_with_conjugates_is_refused_before_minpoly() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="distance"):
        root_critical_distance_profile(_polynomial((4, 1), (1, 1), (0, 1)))


def test_pair_budget_is_rejected_before_exact_root_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_critical_distance_profile(_polynomial((4, 1), (0, -2)), max_pair_rows=0)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial.root_critical.pair_output_bound"
    )


def test_shifted_quadratic_surd_profile_selects_square_root_distances() -> None:
    # (z-5/2)(z^2-2) = z^3 - (5/2)z^2 - 2z + 5
    source = RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(3,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-5, den=2),
                    exponents=(2,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-2, den=1),
                    exponents=(1,),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=5, den=1),
                    exponents=(0,),
                ),
            )
        ),
    )
    result = root_critical_distance_profile(source)
    assert len(result.pairs) == 6
    assert all(row.kind == "POSITIVE" for row in result.pairs)
    assert {(1, -12, 4), (4, -1)}.issubset(
        {tuple(row.distance_squared.polynomial) for row in result.pairs}
    )


def test_shifted_quadratic_square_root_pair_selects_the_distance() -> None:
    # (z - 5/2)(z^2 - 2) stays in the square-root grammar; the pair √2 and 2
    # has squared distance 6 - 4√2 whose coarse box is wider than one isolating
    # interval, but unique intersection still selects the smaller root.
    result = root_critical_distance_profile(
        _polynomial((3, 2), (2, -5), (1, -4), (0, 10))
    )
    assert len(result.pairs) >= 1
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert (1, -12, 4) in values


def test_zero_pair_budget_rejects_before_root_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="row budget"):
        root_critical_distance_profile(
            _polynomial((4, 1), (0, -5)), max_pair_rows=0
        )


def test_native_pair_budget_matches_catalog_range() -> None:
    polynomial = _polynomial((3, 1), (0, -1))
    with pytest.raises(OperationDomainValidationError, match=r"0\.\.64"):
        root_critical_distance_profile(polynomial, max_pair_rows=65)
    with pytest.raises(OperationDomainValidationError, match="non-boolean"):
        root_critical_distance_profile(polynomial, max_pair_rows="64")  # type: ignore[arg-type]


def test_root_rectangles_ignore_crootof_cache_refinement() -> None:
    import sympy
    from sympy.polys.rootoftools import CRootOf

    CRootOf.clear_cache()
    try:
        variable = sympy.Symbol("z")
        cached = sympy.CRootOf(variable**2 - 2, 0)
        cached.refine()
        cached.eval_rational(n=80)
        result = root_critical_distance_profile(_polynomial((2, 1), (0, -2)))
        digits = [
            max(len(str(abs(component.num))), len(str(component.den)))
            for root in (*result.roots, *result.critical_points)
            for component in (
                root.rectangle.real_lower,
                root.rectangle.real_upper,
                root.rectangle.imaginary_lower,
                root.rectangle.imaginary_upper,
            )
        ]
        assert digits
        assert max(digits) <= 256
        again = root_critical_distance_profile(_polynomial((2, 1), (0, -2)))
        assert again.model_dump() == result.model_dump()
    finally:
        CRootOf.clear_cache()


def test_catalog_example_states_the_univariate_qq_precondition() -> None:
    description = TOOLS[0].examples[0].description
    assert "bounded nonconstant univariate" in description
    assert "QQ" in description


def test_expired_owner_deadline_stops_before_the_sympy_kernel() -> None:
    with (
        request_execution(time.monotonic() - 61.0),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        root_critical_distance_profile(_polynomial((3, 1), (0, -1)))
