from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    construct,
    fiber,
    positive_grading,
)


def _configuration(entries: tuple[tuple[int, ...], ...]) -> AffineConfiguration:
    return AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=entries,
    )


def test_positive_grading_uses_complete_exact_feasibility() -> None:
    result = positive_grading(_configuration(((10, -9), (-1, 1))))
    assert result.positive
    assert tuple(value.as_fraction() for value in result.grading) == (
        Fraction(2),
        Fraction(19),
    )


def test_affine_fiber_rejects_target_work_before_recursion() -> None:
    semigroup = construct(
        _configuration(((1, 0), (0, 1))),
        (CanonicalRational(num=1, den=1), CanonicalRational(num=1, den=1)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        fiber(semigroup, (10**100, 0))
