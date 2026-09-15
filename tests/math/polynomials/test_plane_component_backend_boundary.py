"""QEPCAD backend boundary for plane-component profiles.

This module deliberately carries no ``qepcad`` skip: degenerate requests are
expected to resolve without the external backend, while a nondegenerate
request without QEPCAD must raise a typed refusal rather than a verdict.
"""

from __future__ import annotations

import shutil
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.backends import BackendUnavailableError
from jacobian.math.polynomials.real_algebra import compute_plane_component_profile
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    PlaneComponentProfileComputed,
    PlaneSemialgebraicSet,
    PlaneSign,
    PlaneSignCondition,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    terms: tuple[tuple[int, tuple[int, int]], ...],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def test_empty_and_whole_plane_resolve_without_qepcad() -> None:
    empty = PlaneSemialgebraicSet(axis=("x", "y"), polynomials=(), sign_conditions=())
    result = compute_plane_component_profile(empty)
    assert result.outcome.status == "COMPUTED"
    assert isinstance(result.outcome, PlaneComponentProfileComputed)
    assert result.outcome.components == ()

    whole = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(),
        sign_conditions=(PlaneSignCondition(signs=()),),
    )
    whole_result = compute_plane_component_profile(whole)
    assert whole_result.outcome.status == "COMPUTED"
    assert isinstance(whole_result.outcome, PlaneComponentProfileComputed)
    assert len(whole_result.outcome.components) == 1


@pytest.mark.skipif(
    shutil.which("qepcad") is not None,
    reason="this refusal path requires the QEPCAD backend to be absent",
)
def test_nondegenerate_request_without_qepcad_is_a_typed_refusal() -> None:
    disk = _polynomial(((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))))
    semialgebraic_set = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(disk,),
        sign_conditions=(PlaneSignCondition(signs=(PlaneSign.NEGATIVE,)),),
    )
    with pytest.raises(BackendUnavailableError, match="qepcad"):
        compute_plane_component_profile(semialgebraic_set)
