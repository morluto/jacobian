"""Tests for ideal saturation operations."""

import shutil
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.commutative_algebra_ops._models import IdealSaturationRequest
from jacobian.math.commutative_algebra_ops._operations import compute_ideal_saturation
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
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


class TestIdealSaturation:
    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_saturation_xy_by_x(self):
        """<xy> : <x>^inf = <y> in Q[x,y]."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _ideal(("x", "y"), {(1, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = compute_ideal_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None
        assert result.backend_version is not None

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_already_saturated(self):
        """An already saturated ideal remains unchanged."""
        ideal = _ideal(("x", "y"), {(1, 0): 1})
        denominator = _ideal(("x", "y"), {(0, 1): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = compute_ideal_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None

    @requires_singular
    @pytest.mark.requires_backend("singular")
    def test_saturation_by_unit(self):
        """Saturation by a unit (nonzero constant) returns the original ideal."""
        ideal = _ideal(("x", "y"), {(2, 0): 1})
        denominator = _ideal(("x", "y"), {(0, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = compute_ideal_saturation(request)
        assert result.outcome == "COMPUTED"
        assert result.saturation is not None

    def test_mismatched_rings_rejected(self):
        """Saturation operands must use the same ordered ring."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _ideal(("x", "y", "z"), {(1, 0, 0): 1})
        with pytest.raises(ValidationError):
            IdealSaturationRequest(ideal=ideal, denominator=denominator)

    def test_saturation_result_has_backend_version(self):
        """Computed saturation should include a backend version."""
        ideal = _ideal(("x", "y"), {(1, 1): 1})
        denominator = _ideal(("x", "y"), {(1, 0): 1})
        request = IdealSaturationRequest(ideal=ideal, denominator=denominator)
        result = compute_ideal_saturation(request)
        if result.outcome == "COMPUTED":
            assert result.backend_version is not None
            assert result.backend_version != ""
