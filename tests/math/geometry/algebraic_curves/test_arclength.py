"""Behavioral evidence for the bounded regular plane-curve arclength slice."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.geometry.algebraic_curves._arclength import enclose_arclength
from jacobian.math.geometry.algebraic_curves._arclength_models import (
    PlaneCurveArclengthRequest,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial(*terms: tuple[int, tuple[int, int]]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient), exponents=exponents
                )
                for coefficient, exponents in sorted(
                    terms, key=lambda term: term[1], reverse=True
                )
            )
        ),
    )


def _box(
    x_lower: int | Fraction,
    x_upper: int | Fraction,
    y_lower: int | Fraction,
    y_upper: int | Fraction,
) -> RationalBox:
    return RationalBox(
        variables=("x", "y"),
        intervals=(
            ClosedRationalInterval(lower=_rational(x_lower), upper=_rational(x_upper)),
            ClosedRationalInterval(lower=_rational(y_lower), upper=_rational(y_upper)),
        ),
    )


def _request(
    polynomial: RationalPolynomial, box: RationalBox, width: Fraction = Fraction(1, 10)
) -> PlaneCurveArclengthRequest:
    return PlaneCurveArclengthRequest(
        polynomial=polynomial,
        box=box,
        target_width=_rational(width),
        resource_budget={
            "precision_bits": 192,
            "max_segments": 128,
            "wall_seconds": 120,
        },
    )


def test_empty_curve_is_exact_zero() -> None:
    result = enclose_arclength(
        _request(_polynomial((1, (2, 0)), (1, (0, 2)), (1, (0, 0))), _box(-2, 2, -2, 2))
    )
    assert result.outcome.status == "EMPTY"
    assert result.outcome.lower == 0
    assert result.outcome.upper == 0


def test_repeated_circle_is_not_counted_with_multiplicity() -> None:
    result = enclose_arclength(
        _request(
            _polynomial(
                (1, (4, 0)),
                (2, (2, 2)),
                (1, (0, 4)),
                (-2, (2, 0)),
                (-2, (0, 2)),
                (1, (0, 0)),
            ),
            _box(-2, 2, -2, 2),
        )
    )
    assert result.outcome.status == "SINGULAR_CASE_UNSUPPORTED"
    assert result.outcome.reason == "SINGULAR_LEVEL_SET"


def test_transverse_clipped_circle_has_a_certified_nonzero_length() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(-2, 2, 0, 2),
        )
    )
    assert result.outcome.status == "ENCLOSED"
    assert result.outcome.lower.as_fraction() > 3
    assert result.outcome.upper.as_fraction() < 4
    assert (
        result.outcome.upper.as_fraction() - result.outcome.lower.as_fraction()
        <= Fraction(1, 10)
    )


def test_unit_circle_enclosure_contains_two_pi() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(-2, 2, -2, 2),
            Fraction(1, 100),
        )
    )
    assert result.outcome.status == "ENCLOSED"
    assert result.outcome.lower.as_fraction() < Fraction(6284, 1000)
    assert result.outcome.upper.as_fraction() > Fraction(6283, 1000)
    assert (
        result.outcome.upper.as_fraction() - result.outcome.lower.as_fraction()
        <= Fraction(1, 100)
    )


def test_nonconverged_integral_is_unknown(monkeypatch) -> None:
    import jacobian.math.geometry.algebraic_curves._arclength as kernel

    monkeypatch.setattr(kernel, "_integrate_cell", lambda *args, **kwargs: None)
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(-2, 2, -2, 2),
        )
    )
    assert result.outcome.status == "UNKNOWN"
    assert result.outcome.reason == "REFINEMENT_INCOMPLETE"


def test_left_of_irrational_ellipse_is_not_declared_empty() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (2, (0, 2)), (-2, (0, 0))),
            RationalBox(
                variables=("x", "y"),
                intervals=(
                    ClosedRationalInterval(
                        lower=_rational(-1), upper=_rational(Fraction(-1, 2))
                    ),
                    ClosedRationalInterval(lower=_rational(-2), upper=_rational(2)),
                ),
            ),
        )
    )
    assert result.outcome.status != "EMPTY"


def test_contained_irrational_ellipse_is_enclosed() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (2, (0, 2)), (-2, (0, 0))),
            _box(-2, 2, -2, 2),
            Fraction(1, 10),
        )
    )
    assert result.outcome.status == "ENCLOSED"
    assert result.outcome.lower.as_fraction() > 0


def test_finite_tangency_is_singular_unsupported() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(-1, 1, -2, 2),
        )
    )
    assert result.outcome.status == "SINGULAR_CASE_UNSUPPORTED"
    assert result.outcome.reason == "BOUNDARY_NONTRANSVERSE"


def test_projective_left_tangency_is_singular_unsupported() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(-1, 0, -2, 2),
        )
    )
    assert result.outcome.status == "SINGULAR_CASE_UNSUPPORTED"
    assert result.outcome.reason == "BOUNDARY_NONTRANSVERSE"


def test_degenerate_intersecting_box_has_exact_zero_length() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(0, 0, -2, 2),
        )
    )
    assert result.outcome.status == "EMPTY"


def test_tangency_outside_the_box_face_is_empty() -> None:
    result = enclose_arclength(
        _request(
            _polynomial((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))),
            _box(Fraction(-4, 5), Fraction(-3, 5), 1, 2),
        )
    )
    assert result.outcome.status == "EMPTY"
