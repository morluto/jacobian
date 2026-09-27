"""Exact and contract tests for tropical scalar addition."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.tropical import (
    TropicalScalar,
    TropicalSemiring,
    tropical_scalar_add,
)
from jacobian.math.polynomials.tropical._models import (
    ScalarAddRequest,
    ScalarAddResult,
)
from jacobian.math.polynomials.tropical._tools import compute_scalar_add


def _semiring(convention: str = "MIN_PLUS", base: str = "QQ") -> TropicalSemiring:
    return TropicalSemiring(convention=convention, base=base)  # type: ignore[arg-type]


def _finite(semiring: TropicalSemiring, value: Fraction) -> TropicalScalar:
    return TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_fraction(value),
    )


def _infinite(semiring: TropicalSemiring) -> TropicalScalar:
    kind = (
        "POSITIVE_INFINITY"
        if semiring.convention == "MIN_PLUS"
        else "NEGATIVE_INFINITY"
    )
    return TropicalScalar(semiring=semiring, kind=kind)  # type: ignore[arg-type]


def test_min_plus_known_answer() -> None:
    semiring = _semiring("MIN_PLUS")
    result = compute_scalar_add(
        ScalarAddRequest(
            semiring=semiring,
            left=_finite(semiring, Fraction(3)),
            right=_finite(semiring, Fraction(5)),
        )
    )

    assert result.result == _finite(semiring, Fraction(3))
    assert result.branch == "LEFT"
    assert result.infinity_case == "NONE"


def test_max_plus_known_answer() -> None:
    semiring = _semiring("MAX_PLUS")
    result = compute_scalar_add(
        ScalarAddRequest(
            semiring=semiring,
            left=_finite(semiring, Fraction(3)),
            right=_finite(semiring, Fraction(5)),
        )
    )

    assert result.result == _finite(semiring, Fraction(5))
    assert result.branch == "RIGHT"
    assert result.infinity_case == "NONE"


def test_additive_identity_absorbs_on_both_sides() -> None:
    for convention in ("MIN_PLUS", "MAX_PLUS"):
        semiring = _semiring(convention)
        finite = _finite(semiring, Fraction(-7, 3))
        zero = _infinite(semiring)
        for left, right, branch, infinity_case in (
            (finite, zero, "LEFT", "RIGHT_INFINITE"),
            (zero, finite, "RIGHT", "LEFT_INFINITE"),
        ):
            result = compute_scalar_add(
                ScalarAddRequest(semiring=semiring, left=left, right=right)
            )
            assert result.result == finite
            assert result.branch == branch
            assert result.infinity_case == infinity_case

    semiring = _semiring("MIN_PLUS")
    both = compute_scalar_add(
        ScalarAddRequest(
            semiring=semiring, left=_infinite(semiring), right=_infinite(semiring)
        )
    )
    assert both.result == _infinite(semiring)
    assert both.branch == "TIE"
    assert both.infinity_case == "BOTH_INFINITE"


def test_tie_reports_equal_operands() -> None:
    semiring = _semiring("MAX_PLUS")
    result = compute_scalar_add(
        ScalarAddRequest(
            semiring=semiring,
            left=_finite(semiring, Fraction(1, 2)),
            right=_finite(semiring, Fraction(1, 2)),
        )
    )

    assert result.branch == "TIE"
    assert result.result == result.left == result.right


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
@pytest.mark.parametrize(
    "triple",
    [
        (Fraction(1), Fraction(2), Fraction(3)),
        (Fraction(-4), Fraction(0), Fraction(7, 2)),
        (Fraction(5), Fraction(5), Fraction(-1)),
    ],
)
def test_idempotence_commutativity_associativity(
    convention: str, triple: tuple[Fraction, Fraction, Fraction]
) -> None:
    semiring = _semiring(convention)
    values = [_finite(semiring, each) for each in triple]

    for value in values:
        added, _, _ = tropical_scalar_add(semiring, value, value)
        assert added == value

    first, second, third = values
    forward, _, _ = tropical_scalar_add(semiring, first, second)
    backward, _, _ = tropical_scalar_add(semiring, second, first)
    assert forward == backward

    left_inner, _, _ = tropical_scalar_add(semiring, first, second)
    left_total, _, _ = tropical_scalar_add(semiring, left_inner, third)
    right_inner, _, _ = tropical_scalar_add(semiring, second, third)
    right_total, _, _ = tropical_scalar_add(semiring, first, right_inner)
    assert left_total == right_total


def test_defining_invariant_matches_min_max() -> None:
    cases = [
        ("MIN_PLUS", Fraction(3, 4), Fraction(-2, 5)),
        ("MAX_PLUS", Fraction(3, 4), Fraction(-2, 5)),
        ("MIN_PLUS", Fraction(0), Fraction(9, 7)),
        ("MAX_PLUS", Fraction(-8), Fraction(-8, 3)),
    ]
    for convention, left_value, right_value in cases:
        semiring = _semiring(convention)
        total, branch, _ = tropical_scalar_add(
            semiring,
            _finite(semiring, left_value),
            _finite(semiring, right_value),
        )
        expected = (
            min(left_value, right_value)
            if convention == "MIN_PLUS"
            else max(left_value, right_value)
        )
        assert total.value is not None
        assert total.value.as_fraction() == expected
        assert branch == (
            "TIE"
            if left_value == right_value
            else "LEFT"
            if expected == left_value
            else "RIGHT"
        )


def test_native_and_catalog_results_agree() -> None:
    semiring = _semiring("MIN_PLUS", "ZZ")
    left = _finite(semiring, Fraction(4))
    right = _finite(semiring, Fraction(9))
    native, branch, infinity_case = tropical_scalar_add(semiring, left, right)
    catalog = compute_scalar_add(
        ScalarAddRequest(semiring=semiring, left=left, right=right)
    )

    assert catalog.result == native
    assert catalog.branch == branch
    assert catalog.infinity_case == infinity_case
    assert ScalarAddResult.model_validate_json(catalog.model_dump_json()) == catalog


def test_unlicensed_infinity_is_rejected_structurally() -> None:
    semiring = _semiring("MAX_PLUS")
    with pytest.raises(ValidationError):
        TropicalScalar(semiring=semiring, kind="POSITIVE_INFINITY")

    forged = TropicalScalar.model_construct(
        semiring=semiring, kind="POSITIVE_INFINITY", value=None
    )
    with pytest.raises(OperationDomainValidationError):
        tropical_scalar_add(semiring, forged, _finite(semiring, Fraction(1)))


def test_dual_semiring_operand_is_rejected() -> None:
    semiring = _semiring("MIN_PLUS")
    other = _semiring("MAX_PLUS")
    with pytest.raises(ValidationError):
        ScalarAddRequest(
            semiring=semiring,
            left=_finite(semiring, Fraction(1)),
            right=_finite(other, Fraction(1)),
        )
    with pytest.raises(OperationDomainValidationError):
        tropical_scalar_add(
            semiring, _finite(semiring, Fraction(1)), _finite(other, Fraction(1))
        )


def test_nonintegral_zz_scalar_is_rejected() -> None:
    semiring = _semiring("MIN_PLUS", "ZZ")
    with pytest.raises(ValidationError):
        _finite(semiring, Fraction(1, 2))

    forged = TropicalScalar.model_construct(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational(num=1, den=2),
    )
    with pytest.raises(OperationDomainValidationError):
        tropical_scalar_add(semiring, forged, _finite(semiring, Fraction(3)))


def test_forged_infinity_case_is_rejected() -> None:
    semiring = _semiring("MIN_PLUS")
    request = ScalarAddRequest(
        semiring=semiring,
        left=_finite(semiring, Fraction(1)),
        right=_finite(semiring, Fraction(2)),
    )
    result = compute_scalar_add(request)

    with pytest.raises(ValidationError):
        ScalarAddResult(
            semiring=semiring,
            left=request.left,
            right=request.right,
            result=result.result,
            branch=result.branch,
            infinity_case="LEFT_INFINITE",
        )
