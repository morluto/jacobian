"""A failed exact solve cannot establish that a polytope has no vertices."""

import pytest
from sympy import Rational

from jacobian.math.geometry.polytopes import _polyhedral_conversion
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    PolyhedralConversionError,
)
from jacobian.math.geometry.polytopes._rational_geometry import vertices_from_halfspaces


def test_vertex_enumeration_preserves_failed_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [([Rational(1)], Rational(1)), ([Rational(-1)], Rational(0))]
    assert set(vertices_from_halfspaces(rows, 1)) == {(Rational(0),), (Rational(1),)}

    def fail(*args: object, **kwargs: object) -> None:
        raise PolyhedralConversionError("backend inverse failure")

    monkeypatch.setattr(_polyhedral_conversion, "_inverse_columns", fail)
    with pytest.raises(RuntimeError, match=r"vertex.*computation failed"):
        vertices_from_halfspaces(rows, 1)
