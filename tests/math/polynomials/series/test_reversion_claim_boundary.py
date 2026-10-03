"""Authored reversion claims need complete canonical carriers and exact relations."""

from collections.abc import Sequence
from fractions import Fraction
from math import comb
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational as Q
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.series import operations, reversion, verify_reversion
from jacobian.math.polynomials.series._models import (
    SeriesReversionResult,
    TruncatedSeries,
)


def series(values: Sequence[int | Fraction], order: int = 8) -> TruncatedSeries:
    values = [*values, *([Fraction()] * (order - len(values)))]
    return TruncatedSeries(
        variable="q",
        truncation_order=order,
        coefficients=tuple(Q.from_fraction(Fraction(v)) for v in values),
    )


def claim_for_quadratic(a: Fraction, b: Fraction, order: int) -> SeriesReversionResult:
    # Lagrange inversion: [x^k]G=(-b)^(k-1)*Catalan(k-1)/a^(2k-1).
    coefficients = (
        [Fraction(), 1 / a]
        if not b
        else [
            Fraction(),
            *(
                Fraction(comb(2 * k - 2, k - 1), k) * (-b) ** (k - 1) / a ** (2 * k - 1)
                for k in range(1, order)
            ),
        ]
    )
    zeros = (Q.from_integer_ratio(0, 1),) * order
    return SeriesReversionResult(
        source=series([0, a] if not b else [0, a, b], order),
        result=series(coefficients, order),
        left_residual=zeros,
        right_residual=zeros,
    )


@pytest.mark.parametrize(
    "a,b",
    [
        (Fraction(1), Fraction(1)),
        (Fraction(-3, 2), Fraction(1, 5)),
        (Fraction(2, 3), Fraction(-5, 7)),
    ],
)
def test_independently_authored_rational_witnesses_verify(
    a: Fraction, b: Fraction
) -> None:
    claim = claim_for_quadratic(a, b, 16)
    assert verify_reversion(claim)
    assert verify_reversion(type(claim).model_validate_json(claim.model_dump_json()))
    assert reversion(claim.source).result == claim.result


@pytest.mark.parametrize("order", [2, 8, 25280])
def test_full_linear_witnesses_remain_valid(order: int) -> None:
    claim = claim_for_quadratic(Fraction(-3, 2), Fraction(), order)
    assert verify_reversion(claim)


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("field", ["left_residual", "right_residual"])
@pytest.mark.parametrize("length", [0, 1, 7, 9])
def test_incomplete_or_long_ledgers_are_not_vacuously_true(
    linear: bool, field: str, length: int
) -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(0 if linear else 1), 8)
    forged = claim.model_copy(update={field: (Q.from_integer_ratio(0, 1),) * length})
    assert not verify_reversion(forged)
    with pytest.raises(ValidationError):
        type(claim).model_validate_json(forged.model_dump_json())


@pytest.mark.parametrize("field", ["source", "result"])
@pytest.mark.parametrize("length", [0, 1, 2, 7, 9])
def test_native_carrier_length_must_match_precision(field: str, length: int) -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(), 8)
    original = getattr(claim, field)
    values = (
        original.coefficients[:length]
        if length <= 8
        else (*original.coefficients, Q.from_integer_ratio(0, 1))
    )
    forged = claim.model_copy(
        update={field: original.model_copy(update={"coefficients": values})}
    )
    assert not verify_reversion(forged)


@pytest.mark.parametrize("field", ["source", "result"])
@pytest.mark.parametrize(
    "updates", [{"variable": "bad axis"}, {"truncation_order": True}, {"variable": "x"}]
)
def test_native_context_cannot_be_bypassed(field: str, updates: dict[str, Any]) -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(), 8)
    forged = claim.model_copy(
        update={field: getattr(claim, field).model_copy(update=updates)}
    )
    assert not verify_reversion(forged)


@pytest.mark.parametrize("field", ["source", "result"])
def test_unreduced_native_zero_is_not_a_canonical_witness(field: str) -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(), 8)
    value = getattr(claim, field)
    forged = value.model_copy(
        update={
            "coefficients": (Q.model_construct(num=0, den=2), *value.coefficients[1:])
        }
    )
    assert not verify_reversion(claim.model_copy(update={field: forged}))


def test_structural_decoding_does_not_establish_the_composition_identities() -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(1), 8)
    bad = claim.result.model_copy(
        update={
            "coefficients": (
                *claim.result.coefficients[:2],
                Q.from_integer_ratio(0, 1),
                *claim.result.coefficients[3:],
            )
        }
    )
    forged = claim.model_copy(update={"result": bad})
    decoded = type(claim).model_validate_json(forged.model_dump_json())
    assert not verify_reversion(decoded)


@pytest.mark.parametrize("numerator,denominator", [(1, 3), (1000000, 1)])
def test_necessary_lagrange_bounds_refute_claim_before_replay(
    monkeypatch: pytest.MonkeyPatch, numerator: int, denominator: int
) -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(1), 3)
    result = claim.result.model_copy(
        update={
            "coefficients": (
                *claim.result.coefficients[:2],
                Q.from_integer_ratio(numerator, denominator),
            )
        }
    )

    def unexpected(*args: object) -> Any:
        pytest.fail("impossible inverse entered replay")

    monkeypatch.setattr(operations, "_compose_coefficients", unexpected)
    assert not verify_reversion(claim.model_copy(update={"result": result}))


def test_oversized_claim_is_resource_refusal_not_mathematical_negative() -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(1), 8)
    result = claim.result.model_copy(
        update={
            "coefficients": (
                *claim.result.coefficients[:2],
                Q.from_integer_ratio(10**4096, 1),
                *claim.result.coefficients[3:],
            )
        }
    )
    with pytest.raises(OperationResourceAdmissionError):
        verify_reversion(claim.model_copy(update={"result": result}))


def test_source_order_refusal_still_propagates() -> None:
    claim = claim_for_quadratic(Fraction(1), Fraction(1), 513)
    with pytest.raises(OperationResourceAdmissionError):
        verify_reversion(claim)


def test_producer_rejects_a_short_native_source_too() -> None:
    source = series([0, 1]).model_copy(
        update={"coefficients": series([0, 1]).coefficients[:2]}
    )
    with pytest.raises(OperationDomainValidationError):
        reversion(source)


def test_valid_large_nonlinear_inverse_survives_claim_envelope() -> None:
    assert verify_reversion(claim_for_quadratic(Fraction(1), Fraction(1), 512))
