"""Exact inverse/quotient scale, complete private bounds, and authored claims."""

from collections.abc import Sequence
from fractions import Fraction
from random import Random
from typing import Any

import pytest

from jacobian._exact import CanonicalRational as Q
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.series import (
    _flint,
    _unit_bounds,
    divide,
    inverse,
    reversion,
    verify_divide,
    verify_inverse,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def _series(values: Sequence[int | Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="q",
        truncation_order=len(values),
        coefficients=tuple(Q.from_fraction(Fraction(value)) for value in values),
    )


def _unit(order: int) -> TruncatedSeries:
    return _series([1, -1, *([0] * (order - 2))])


@pytest.mark.parametrize("order", [513, 1024, 2048])
def test_complete_large_inverse_and_quotient(order: int) -> None:
    from flint import ctx

    cap = ctx.cap
    source = _unit(order)
    inverted = inverse(source)
    assert all(
        value == Q.from_integer_ratio(1, 1) for value in inverted.result.coefficients
    )
    assert verify_inverse(
        type(inverted).model_validate_json(inverted.model_dump_json())
    )
    numerator = _series([1, 0, -1, *([0] * (order - 3))])
    quotient = divide(numerator, source)
    assert [v.as_fraction() for v in quotient.quotient.coefficients] == [
        1,
        1,
        *([0] * (order - 2)),
    ]
    assert verify_divide(type(quotient).model_validate_json(quotient.model_dump_json()))
    # Actual output can be supplied as the next numerator with unchanged axes/order.
    recovered = divide(inverted.result, source)
    assert [v.as_fraction() for v in recovered.quotient.coefficients] == list(
        range(1, order + 1)
    )
    assert ctx.cap == cap


def test_dense_integer_quotient_matches_independent_recurrence() -> None:
    rng = Random(2935)
    order = 1024
    denominator = [1, *(rng.choice((-1, 1)) for _ in range(order - 1))]
    numerator = [rng.choice((-1, 0, 1)) for _ in range(order)]
    expected: list[int] = []
    for degree in range(order):
        expected.append(
            numerator[degree]
            - sum(denominator[j] * expected[degree - j] for j in range(1, degree + 1))
        )
    result = divide(_series(numerator), _series(denominator))
    assert [v.as_fraction() for v in result.quotient.coefficients] == expected
    assert verify_divide(result)


@pytest.mark.parametrize("seed", range(6))
def test_rational_signed_division_matches_independent_recurrence(seed: int) -> None:
    rng = Random(seed)
    n = 25
    denominator = [
        Fraction(-2, 3),
        *(Fraction(rng.randint(-3, 3), rng.randint(1, 5)) for _ in range(n - 1)),
    ]
    numerator = [Fraction(rng.randint(-3, 3), rng.randint(1, 7)) for _ in range(n)]
    expected: list[Fraction] = []
    for k in range(n):
        expected.append(
            (
                numerator[k]
                - sum(
                    (denominator[j] * expected[k - j] for j in range(1, k + 1)),
                    Fraction(),
                )
            )
            / denominator[0]
        )
    result = divide(_series(numerator), _series(denominator))
    assert [v.as_fraction() for v in result.quotient.coefficients] == expected
    assert verify_divide(result)


def test_constant_denominator_keeps_existing_large_order_without_global_lcm() -> None:
    n = 25280
    numerator = _series([Fraction((i % 3) - 1, 2 + (i % 71)) for i in range(n)])
    denominator = _series([Fraction(3, 7), *([0] * (n - 1))])
    result = divide(numerator, denominator)
    assert all(
        q.as_fraction() == a.as_fraction() * Fraction(7, 3)
        for a, q in zip(
            numerator.coefficients, result.quotient.coefficients, strict=True
        )
    )
    assert verify_divide(result)
    assert verify_inverse(inverse(denominator))


def test_large_order_refuses_before_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object) -> Any:
        pytest.fail("out-of-envelope source entered FLINT")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    with pytest.raises(OperationResourceAdmissionError, match="2048"):
        inverse(_unit(2049))
    with pytest.raises(OperationResourceAdmissionError, match="2048"):
        divide(_unit(2049), _unit(2049))


def test_high_source_height_retains_useful_accepted_edge() -> None:
    source = _series([1 << 850, 1, *([0] * 12)])
    result = inverse(source)
    assert result.result.coefficients[-1].as_fraction() == Fraction(-1, 1 << 11900)
    assert verify_inverse(result)
    with pytest.raises(OperationResourceAdmissionError):
        inverse(_series([1 << 850, 1, *([0] * 13)]))


@pytest.mark.parametrize(
    "constant,amount",
    [
        ("MAX_UNIT_WORK", 5),
        ("MAX_UNIT_SCRATCH_BITS", 64),
        ("MAX_UNIT_STORAGE_BITS", 100),
        ("MAX_UNIT_LIMB_WORK", 5),
    ],
)
def test_envelope_inclusive_boundaries(
    monkeypatch: pytest.MonkeyPatch, constant: str, amount: int
) -> None:
    envelope = _unit_bounds.Envelope(5, 64, 100)
    monkeypatch.setattr(_unit_bounds, constant, amount)
    envelope.checked()
    monkeypatch.setattr(_unit_bounds, constant, amount - 1)
    with pytest.raises(OperationResourceAdmissionError):
        envelope.checked()


@pytest.mark.parametrize(
    "updates",
    [
        {"variable": "bad axis"},
        {"truncation_order": True},
        {"coefficients": ()},
        {"coefficients": [Q.from_integer_ratio(1, 1)]},
    ],
)
def test_native_shape_rejection_is_typed(updates: dict[str, Any]) -> None:
    bad = _unit(513).model_copy(update=updates)
    with pytest.raises(OperationDomainValidationError):
        inverse(bad)


@pytest.mark.parametrize("num,den", [(2, 2), (0, 2), (1, 0), (True, 1)])
def test_noncanonical_native_scalar_rejected(num: Any, den: int) -> None:
    source = _unit(513)
    coefficients = (Q.model_construct(num=num, den=den), *source.coefficients[1:])
    with pytest.raises(OperationDomainValidationError):
        inverse(source.model_copy(update={"coefficients": coefficients}))


def test_forged_result_and_residual_are_mathematical_negatives() -> None:
    claim = inverse(_unit(513))
    false_result = claim.result.model_copy(
        update={
            "coefficients": (Q.from_integer_ratio(2, 1), *claim.result.coefficients[1:])
        }
    )
    assert not verify_inverse(claim.model_copy(update={"result": false_result}))
    assert not verify_inverse(
        claim.model_copy(
            update={
                "residual_coefficients": (
                    Q.from_integer_ratio(1, 1),
                    *claim.residual_coefficients[1:],
                )
            }
        )
    )
    wrong_axes = claim.result.model_copy(update={"variable": "x"})
    assert not verify_inverse(claim.model_copy(update={"result": wrong_axes}))


def test_untrusted_result_height_is_admitted_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = inverse(_unit(513))
    huge = Q.from_integer_ratio(10**4096, 1)
    false_result = claim.result.model_copy(
        update={"coefficients": (huge, *claim.result.coefficients[1:])}
    )

    def unexpected(*args: object) -> Any:
        pytest.fail("unadmitted claim entered FLINT")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    with pytest.raises(OperationResourceAdmissionError):
        verify_inverse(claim.model_copy(update={"result": false_result}))


@pytest.mark.parametrize(
    "error_type", [OperationExecutionCancelledError, OperationExecutionTimeoutError]
)
def test_newton_cancellation_keeps_operational_classification(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    claim = inverse(_unit(513))

    def cancelled(*args: object) -> None:
        raise error_type("interrupted Newton")

    monkeypatch.setattr(_flint, "request_checkpoint", cancelled)
    with pytest.raises(error_type):
        inverse(_unit(513))
    with pytest.raises(error_type):
        divide(_unit(513), _unit(513))
    with pytest.raises(error_type):
        verify_inverse(claim)
    with pytest.raises(error_type):
        reversion(_series([0, 1, 1, *([0] * 13)]))


def test_large_constant_quotient_height_still_replays() -> None:
    n = 25280
    p = 10**255
    numerator = _series([Fraction(p - 1, p - 3)] * n)
    denominator = _series([Fraction(p - 5, p - 7), *([0] * (n - 1))])
    claim = divide(numerator, denominator)
    assert claim.quotient.coefficients[-1].as_fraction() == Fraction(
        p - 1, p - 3
    ) / Fraction(p - 5, p - 7)
    assert verify_divide(claim)


def test_replay_limb_work_refuses_before_multiplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = inverse(_unit(513))
    false_result = claim.result.model_copy(
        update={
            "coefficients": (
                Q.from_integer_ratio(10**3999, 1),
                *claim.result.coefficients[1:],
            )
        }
    )
    claim = claim.model_copy(update={"result": false_result})
    monkeypatch.setattr(_unit_bounds, "MAX_UNIT_LIMB_WORK", 1 << 29)
    with pytest.raises(OperationResourceAdmissionError, match="limb_work"):
        verify_inverse(claim)


def test_supplied_denominator_diversity_refuses_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy import primerange

    claim = inverse(_unit(513))
    primes = list(primerange(2, 5000))[:513]
    result = claim.result.model_copy(
        update={
            "coefficients": tuple(
                Q.from_integer_ratio(1, int(p) ** (1024 // int(p).bit_length()))
                for p in primes
            )
        }
    )

    def unexpected(*args: object) -> Any:
        pytest.fail("unadmitted common denominator entered FLINT")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    with pytest.raises(OperationResourceAdmissionError, match="clearing"):
        verify_inverse(claim.model_copy(update={"result": result}))
