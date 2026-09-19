"""Operational-failure regressions for shared recession-cone geometry."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.polytopes import (
    Halfspace,
    _polyhedral_conversion,
)
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
    def failed_rank(*_args: object, **_kwargs: object) -> int:
        raise _polyhedral_conversion.PolyhedralConversionError(
            "exact polyhedral rank computation failed"
        )

    monkeypatch.setattr(_polyhedral_conversion, "_rank", failed_rank)
    halfspaces = (
        Halfspace(coefficients=(_rational(1), _rational(0)), offset=_rational(1)),
        Halfspace(coefficients=(_rational(0), _rational(1)), offset=_rational(1)),
        Halfspace(coefficients=(_rational(-1), _rational(-1)), offset=_rational(0)),
    )
    with pytest.raises(RecessionConeComputationError, match="cone computation failed"):
        polytope_operations._is_bounded_h(halfspaces)
    with pytest.raises(RuntimeError, match="vertex enumeration computation failed"):
        lattice_operations._facets_and_box(None, halfspaces, 2)
