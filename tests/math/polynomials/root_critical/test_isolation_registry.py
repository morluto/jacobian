"""Registry of isolation audits for every root-critical rectangle producer.

Every operation in this package that publishes an ``isolating_rectangle`` is
registered here with the squarefree support it must separate against and the
rectangles it returns.  The same invariant helper then audits each producer, so
a future operation inherits the audit by adding one registry entry instead of
re-deriving the invariant.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import pytest
import sympy
from tests.math.polynomials.root_critical._isolation_invariants import (
    require_axis_rectangles_are_isolating,
    require_rectangle_isolates_one_root,
)

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.root_critical import (
    ExactSplittingField,
    RootCriticalDistanceProfile,
    exact_splitting_field,
    root_critical_distance_profile,
)
from jacobian.math.polynomials.root_critical._splitting import (
    _polynomial_from_ascending,
)

_Kind = Literal["axis", "single"]
_Case = tuple[str, _Kind, sympy.Poly, tuple[object, ...]]

_PROFILE_EXPRESSIONS = (
    "z**3 - 1",
    "(z**2 + 1)*(z - 2)",
    "z**3 - 2",
    "z**4 - 2",
    "(z**2 - 2)*(z**2 - 3)",
    "(z**2 - 2)*(z**2 + 1)",
    "z**4 + 1",
)
_SPLITTING_COEFFICIENTS = (
    (-2, 0, 1),
    (-1, -1, 1),
    (-2, 0, 0, 1),
    (0, 0, 1),
)


def _profile_cases() -> list[_Case]:
    z = sympy.Symbol("z")
    cases: list[_Case] = []
    for expression in _PROFILE_EXPRESSIONS:
        source = sympy.Poly(sympy.sympify(expression, locals={"z": z}), z, domain="QQ")
        profile: RootCriticalDistanceProfile = root_critical_distance_profile(
            rational_polynomial_from_sympy(source, ("z",))
        )
        cases.append(
            (
                f"profile.roots[{expression}]",
                "axis",
                source.sqf_part(),
                tuple(root.rectangle for root in profile.roots),
            )
        )
        cases.append(
            (
                f"profile.critical[{expression}]",
                "axis",
                source.diff().sqf_part(),
                tuple(point.rectangle for point in profile.critical_points),
            )
        )
    return cases


def _splitting_cases() -> list[_Case]:
    cases: list[_Case] = []
    for ascending in _SPLITTING_COEFFICIENTS:
        field: ExactSplittingField = exact_splitting_field(
            _polynomial_from_ascending([Fraction(value) for value in ascending], "x")
        )
        cases.append(
            (
                f"splitting.roots{ascending}",
                "axis",
                rational_polynomial_to_sympy(field.squarefree_support).sqf_part(),
                tuple(root.rectangle for root in field.roots),
            )
        )
        cases.append(
            (
                f"splitting.embedding{ascending}",
                "single",
                rational_polynomial_to_sympy(field.defining_polynomial).sqf_part(),
                (field.embedding_rectangle,),
            )
        )
    return cases


def _all_cases() -> list[_Case]:
    return _profile_cases() + _splitting_cases()


_CASES = _all_cases()
_CASE_IDS = [case[0] for case in _CASES]


@pytest.mark.parametrize(
    "kind, support, rectangles",
    [case[1:] for case in _CASES],
    ids=_CASE_IDS,
)
def test_registered_isolation_audit(
    kind: _Kind, support: sympy.Poly, rectangles: tuple[object, ...]
) -> None:
    """Every registered producer separates the complete support root family."""
    if kind == "single":
        for rectangle in rectangles:
            require_rectangle_isolates_one_root(support, rectangle)
        return
    require_axis_rectangles_are_isolating(support, rectangles)
