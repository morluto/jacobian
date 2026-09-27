"""Exact rational-exponent local-series carrier contracts."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.local_series import PuiseuxTerm, TruncatedPuiseuxWindow


def q(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def test_cusp_branch_prefix_retains_fractional_exponents_and_cutoff() -> None:
    branch = TruncatedPuiseuxWindow(
        variable="x",
        center=q(0),
        valuation_lower=q(3, 2),
        precision=q(4),
        ramification_index=2,
        terms=(PuiseuxTerm(exponent=q(3, 2), coefficient=q(1)),),
    )

    restored = TruncatedPuiseuxWindow.model_validate_json(branch.model_dump_json())
    assert restored == branch
    assert restored.terms[0].exponent.as_fraction() == Fraction(3, 2)
    assert restored.precision.as_fraction() == 4


def test_mixed_half_and_third_exponents_require_ramification_six() -> None:
    mixed = TruncatedPuiseuxWindow(
        valuation_lower=q(1, 3),
        precision=q(2),
        ramification_index=6,
        terms=(
            PuiseuxTerm(exponent=q(1, 3), coefficient=q(1)),
            PuiseuxTerm(exponent=q(1, 2), coefficient=q(1)),
        ),
    )
    assert mixed.ramification_index == 6


@pytest.mark.parametrize(
    "changes",
    [
        {"ramification_index": 2},
        {"terms": (PuiseuxTerm(exponent=q(4), coefficient=q(1)),)},
        {
            "terms": (
                PuiseuxTerm(exponent=q(1), coefficient=q(1)),
                PuiseuxTerm(exponent=q(1), coefficient=q(2)),
            )
        },
    ],
)
def test_puiseux_window_rejects_inconsistent_lattice_and_support(changes: dict) -> None:
    values = {
        "valuation_lower": q(0),
        "precision": q(3),
        "ramification_index": 1,
        "terms": (PuiseuxTerm(exponent=q(1), coefficient=q(1)),),
    }
    values.update(changes)
    with pytest.raises(ValidationError):
        TruncatedPuiseuxWindow(**values)


def test_zero_puiseux_prefix_is_distinct_from_an_infinite_zero_claim() -> None:
    zero_prefix = TruncatedPuiseuxWindow(
        valuation_lower=q(-1, 2),
        precision=q(3, 2),
        ramification_index=2,
        terms=(),
    )
    assert zero_prefix.terms == ()
    assert zero_prefix.precision.as_fraction() == Fraction(3, 2)
