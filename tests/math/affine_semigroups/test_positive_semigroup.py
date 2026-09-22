from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    AffineFiber,
    AffineMembershipResult,
    PositiveAffineSemigroup,
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


def test_positive_grading_rejects_model_constructed_configuration() -> None:
    forged = AffineConfiguration.model_construct(
        row_labels=("x",), generator_labels=("a",), entries=()
    )
    with pytest.raises(OperationDomainValidationError):
        positive_grading(forged)


def test_serialized_affine_results_reject_false_factorizations() -> None:
    config = AffineConfiguration(
        row_labels=("r",), generator_labels=("g",), entries=((1,),)
    )
    semigroup = PositiveAffineSemigroup(
        configuration=config,
        grading=(CanonicalRational(num=1, den=1),),
    )
    with pytest.raises(ValidationError):
        AffineFiber.model_validate(
            {
                "semigroup": semigroup.model_dump(mode="json"),
                "target": [1],
                "factorizations": [[9]],
            }
        )
    with pytest.raises(ValidationError):
        AffineMembershipResult.model_validate(
            {
                "semigroup": semigroup.model_dump(mode="json"),
                "target": [1],
                "cone_member": True,
                "lattice_member": True,
                "semigroup_member": True,
                "factorization": [9],
            }
        )


def test_affine_fiber_rejects_target_work_before_recursion() -> None:
    semigroup = construct(
        _configuration(((1, 0), (0, 1))),
        (CanonicalRational(num=1, den=1), CanonicalRational(num=1, den=1)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        fiber(semigroup, (10**100, 0))
