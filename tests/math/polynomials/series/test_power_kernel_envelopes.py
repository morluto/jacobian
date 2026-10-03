"""Exact binary powers, complete distributions, and resource ownership."""

from collections.abc import Sequence
from fractions import Fraction
from random import Random
from typing import Any

import pytest

from jacobian._exact import CanonicalRational as Q
from jacobian._execution import (
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.series import (
    _flint,
    _power_bounds,
    operations,
    power,
    truncate,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def series(values: Sequence[int | Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="x",
        truncation_order=len(values),
        coefficients=tuple(Q.from_fraction(Fraction(v)) for v in values),
    )


def trinomial_coefficients(exponent: int) -> list[int]:
    # (1+x+x²)F'=exponent*(1+2x)F gives an independent exact recurrence.
    result = [1]
    for k in range(2 * exponent):
        value, remainder = divmod(
            (exponent - k) * result[k]
            + (2 * exponent - k + 1) * (result[k - 1] if k else 0),
            k + 1,
        )
        assert remainder == 0
        result.append(value)
    return result


def test_complete_trinomial_distribution_at_order_2001() -> None:
    from flint import ctx

    cap = ctx.cap
    result = power(series([1, 1, 1, *([0] * 1998)]), 1000)
    values = [v.as_fraction() for v in result.result.coefficients]
    assert values == trinomial_coefficients(1000)
    assert values == values[::-1]
    assert sum(values) == 3**1000
    assert result.multiplication_count == 15
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result
    assert (
        truncate(decoded.result, 512).result.coefficients
        == result.result.coefficients[:512]
    )
    assert ctx.cap == cap


@pytest.mark.parametrize("seed", range(6))
def test_signed_rational_powers_match_repeated_convolution(seed: int) -> None:
    rng = Random(seed)
    n = 17
    values = [Fraction(rng.randint(-3, 3), rng.randint(1, 7)) for _ in range(n)]
    exponent = seed + 1
    expected = [Fraction(1), *([Fraction()] * (n - 1))]
    for _ in range(exponent):
        expected = [
            sum((expected[j] * values[k - j] for j in range(k + 1)), Fraction())
            for k in range(n)
        ]
    result = power(series(values), exponent)
    assert [v.as_fraction() for v in result.result.coefficients] == expected
    assert (
        result.multiplication_count == exponent.bit_count() + exponent.bit_length() - 1
    )


@pytest.mark.parametrize(
    "constant,exponent",
    [(0, 0), (0, 1), (0, 1000), (-1, 999), (-1, 1000), (2, 0), (2, 12)],
)
def test_constant_and_zero_cases_keep_binary_counts(
    constant: int, exponent: int
) -> None:
    result = power(series([constant, *([0] * 2047)]), exponent)
    assert [v.as_fraction() for v in result.result.coefficients] == [
        constant**exponent,
        *([0] * 2047),
    ]
    assert result.multiplication_count == (
        exponent.bit_count() + exponent.bit_length() - 1 if exponent else 0
    )


def test_zero_exponent_does_not_clear_unused_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy import primerange

    source = series(
        [
            Fraction(1, int(p) ** (850 // int(p).bit_length()))
            for p in list(primerange(2, 20000))[:2048]
        ]
    )

    def unexpected(*args: object) -> Any:
        pytest.fail("zero exponent entered backend conversion")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    result = power(source, 0)
    assert result.result.coefficients[0].as_fraction() == 1
    assert all(v.num == 0 for v in result.result.coefficients[1:])
    assert result.multiplication_count == 0


def test_order_boundary_and_refusal_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = power(series([1, -1, *([0] * 2046)]), 2)
    assert [v.as_fraction() for v in accepted.result.coefficients[:3]] == [1, -2, 1]

    def unexpected(*args: object) -> Any:
        pytest.fail("oversized power entered backend")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    with pytest.raises(OperationResourceAdmissionError, match="2048"):
        power(series([1, -1, *([0] * 2047)]), 2)


def test_dense_high_height_limb_work_refuses_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object) -> Any:
        pytest.fail("unadmitted work entered backend")

    monkeypatch.setattr(_flint, "_series_from_fractions", unexpected)
    with pytest.raises(OperationResourceAdmissionError, match="limb_work"):
        power(series([1] * 2048), 1000)


def test_constant_large_coefficients_keep_useful_height_boundary() -> None:
    source = series([1 << 750, *([0] * 2047)])
    result = power(source, 18)
    assert result.result.coefficients[0].as_fraction() == 1 << 13500
    assert all(v.num == 0 for v in result.result.coefficients[1:])
    with pytest.raises(OperationResourceAdmissionError):
        power(source, 19)


@pytest.mark.parametrize("exponent", [-1, 1001, True, "2"])
def test_invalid_exponents_remain_domain_failures(exponent: Any) -> None:
    with pytest.raises(OperationDomainValidationError):
        power(series([1, 1]), exponent)


@pytest.mark.parametrize(
    "updates",
    [{"variable": "bad axis"}, {"coefficients": ()}, {"truncation_order": True}],
)
def test_native_structure_rejected_before_power(updates: dict[str, Any]) -> None:
    with pytest.raises(OperationDomainValidationError):
        power(series([1, 1]).model_copy(update=updates), 2)


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_POWER_WORK", 5),
        ("MAX_POWER_LIMB_WORK", 5),
        ("MAX_POWER_SCRATCH_BITS", 64),
        ("MAX_POWER_STORAGE_BITS", 100),
    ],
)
def test_power_envelope_inclusive_boundaries(
    monkeypatch: pytest.MonkeyPatch, name: str, value: int
) -> None:
    envelope = _power_bounds.PowerEnvelope(5, 64, 100)
    monkeypatch.setattr(_power_bounds, name, value)
    envelope.checked()
    monkeypatch.setattr(_power_bounds, name, value - 1)
    with pytest.raises(OperationResourceAdmissionError):
        envelope.checked()


@pytest.mark.parametrize(
    "error_type", [OperationExecutionCancelledError, OperationExecutionTimeoutError]
)
@pytest.mark.parametrize("late", [False, True])
def test_power_interruptions_keep_operational_classification(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception], late: bool
) -> None:
    def interrupted(*args: object) -> None:
        raise error_type("interrupted power")

    monkeypatch.setattr(
        operations if late else _flint, "request_checkpoint", interrupted
    )
    with pytest.raises(error_type):
        power(series([1, 1, 1, *([0] * 510)]), 7)


def test_power_backend_failure_is_operational(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*args: object) -> Any:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(_flint, "_series_from_fractions", failed)
    with pytest.raises(OperationBackendError):
        power(series([1, 1]), 2)


def test_dense_large_admitted_power_matches_binomial_coefficients() -> None:
    from math import comb

    result = power(series([1] * 2048), 500)
    assert all(
        v.as_fraction() == comb(k + 499, 499)
        for k, v in enumerate(result.result.coefficients)
    )
