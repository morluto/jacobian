"""Operational-failure regressions for shared recession-cone geometry."""

from __future__ import annotations

from fractions import Fraction

import pytest
from sympy import Rational

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.polytopes import Halfspace, _rational_geometry
from jacobian.math.geometry.polytopes import operations as polytope_operations
from jacobian.math.geometry.polytopes._rational_geometry import (
    RecessionConeComputationError,
)
from jacobian.math.geometry.polytopes.lattice import operations as lattice_operations


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def test_recession_rank_failure_is_operational_for_both_polytope_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedRankMatrix:
        def __init__(self, _rows: object) -> None:
            pass

        def rank(self) -> int:
            raise RuntimeError("backend rank failure")

    monkeypatch.setattr(_rational_geometry, "Matrix", FailedRankMatrix)
    halfspaces = (
        Halfspace(coefficients=(_rational(1), _rational(0)), offset=_rational(1)),
        Halfspace(coefficients=(_rational(0), _rational(1)), offset=_rational(1)),
        Halfspace(coefficients=(_rational(-1), _rational(-1)), offset=_rational(0)),
    )
    raw_halfspaces = [
        ([Rational(1), Rational(0)], Rational(1)),
        ([Rational(0), Rational(1)], Rational(1)),
        ([Rational(-1), Rational(-1)], Rational(0)),
    ]

    with pytest.raises(RecessionConeComputationError, match="rank computation failed"):
        polytope_operations._is_bounded_h(halfspaces)
    with pytest.raises(RecessionConeComputationError, match="rank computation failed"):
        lattice_operations._is_bounded_h(raw_halfspaces, 2)
