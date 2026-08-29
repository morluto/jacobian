"""Tests for ideal saturation operations."""

import shutil
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.ideals._models import IdealSaturationRequest
from jacobian.math.polynomials.ideals.operations import ideal_saturation
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _run_saturation(request: IdealSaturationRequest):
    return ideal_saturation(
        request.ideal, request.denominator, resource_budget=request.resource_budget
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _ideal(
    variables: tuple[str, ...],
    *generators: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(_polynomial(variables, generator) for generator in generators),
    )


requires_singular = pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)


def _ideals_equal(
    left: RationalPolynomialIdeal, right: RationalPolynomialIdeal
) -> bool:
    """Check two ideals are equal via mutual Groebner reduction (independent oracle)."""

    import sympy

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
        symbols_for_variables,
    )

    assert left.variables == right.variables
    variables = left.variables
    symbols = symbols_for_variables(variables)
    left_exprs = [rational_polynomial_to_sympy(g).as_expr() for g in left.generators]
    right_exprs = [rational_polynomial_to_sympy(g).as_expr() for g in right.generators]
    # Groebner bases with the same order give a canonical ideal membership test
    left_gb = sympy.groebner(left_exprs, *symbols, order="grevlex", domain=sympy.QQ)
    right_gb = sympy.groebner(right_exprs, *symbols, order="grevlex", domain=sympy.QQ)
    return all(right_gb.reduce(expr)[1] == 0 for expr in left_exprs) and all(
        left_gb.reduce(expr)[1] == 0 for expr in right_exprs
    )


class TestIdealSaturation:
    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_saturation_xy_by_x(self) -> None:
        """<xy> : <x>^inf = <y> in Q[x,y]."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _polynomial(("x", "y"), {(1, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = _run_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None
        assert result.backend_version is not None
        expected = _ideal(("x", "y"), {(0, 1): 1})
        assert _ideals_equal(result.saturation, expected), (
            "saturation <xy>:<x>^inf should be <y>"
        )

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_already_saturated(self) -> None:
        """An already saturated ideal remains unchanged."""
        ideal = _ideal(("x", "y"), {(1, 0): 1})
        denominator = _polynomial(("x", "y"), {(0, 1): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = _run_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None
        # <x> : <y>^inf = <x> (check ideal equality, not just non-null)
        assert _ideals_equal(result.saturation, ideal)

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_saturation_by_unit(self) -> None:
        """Saturation by a unit (nonzero constant) returns the original ideal."""
        ideal = _ideal(("x", "y"), {(2, 0): 1})
        denominator = _polynomial(("x", "y"), {(0, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = _run_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None
        assert _ideals_equal(result.saturation, ideal)

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_principal_saturation_by_product_gives_unit_ideal(self) -> None:
        """<xy> : <xy>^inf is the unit ideal for the single polynomial xy."""

        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _polynomial(("x", "y"), {(1, 1): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = _run_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None
        assert _ideals_equal(result.saturation, _ideal(("x", "y"), {(0, 0): 1})), (
            "saturation <xy>:<xy>^inf should be the unit ideal"
        )

    def test_denominator_must_be_single_polynomial(self) -> None:
        """The operation advertises principal saturation I : <d>^infinity."""

        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _ideal(("x", "y"), {(1, 0): 1}, {(0, 1): 1})
        with pytest.raises(ValidationError):
            IdealSaturationRequest.model_validate(
                {"ideal": ideal, "denominator": denominator}
            )

    def test_zero_denominator_rejected(self) -> None:
        """Saturation by the zero polynomial is not admitted."""

        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _polynomial(("x", "y"), {})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        with pytest.raises(OperationDomainValidationError):
            _run_saturation(request)

    def test_mismatched_rings_rejected(self) -> None:
        """Saturation operands must use the same ordered ring."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _polynomial(("x", "y", "z"), {(1, 0, 0): 1})
        with pytest.raises(ValidationError):
            IdealSaturationRequest(ideal=ideal, denominator=denominator)

    def test_denominator_exceeding_total_degree_rejected(self) -> None:
        """The denominator polynomial obeys the same degree-20 bound."""

        ideal = _ideal(("x",), {(2,): 1})
        denominator = _polynomial(("x",), {(21,): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        with pytest.raises(OperationDomainValidationError):
            _run_saturation(request)

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_saturation_result_has_backend_version(self) -> None:
        """Computed saturation should include a backend version."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _polynomial(("x", "y"), {(1, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = _run_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.backend_version is not None
        assert result.backend_version != ""
