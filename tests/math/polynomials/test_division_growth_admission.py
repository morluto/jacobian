"""Exact division admits complete scalar growth before the maintained kernel."""

from fractions import Fraction
from functools import cache
from math import prod
from random import Random
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import rational_polynomial_division
from jacobian.math.polynomials.multivariate import multivariate_division
from jacobian.math.polynomials.multivariate.operations import (
    verify_multivariate_division,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(coefficients: dict[int, Fraction]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(value),
                    exponents=(degree,),
                )
                for degree, value in sorted(coefficients.items(), reverse=True)
                if value
            )
        ),
    )


def _coefficients(polynomial: RationalPolynomial) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


@cache
def _denominators(count: int) -> tuple[int, ...]:
    primes = [
        p for p in range(7, 1000) if all(p % d for d in range(2, int(p**0.5) + 1))
    ][:count]
    values = []
    for prime in primes:
        denominator = prime
        while denominator * prime < 10**256:
            denominator *= prime
        values.append(denominator)
    return tuple(values)


def _family(degree: int) -> tuple[RationalPolynomial, RationalPolynomial]:
    return (
        _poly({i: Fraction(1, d) for i, d in enumerate(_denominators(degree + 1))}),
        _poly({1: Fraction(1), 0: Fraction(-1, 10**255)}),
    )


def _digits(value: int) -> int:
    estimate = max(1, (abs(value).bit_length() * 30103) // 100000 + 1)
    return estimate - 1 if abs(value) < 10 ** (estimate - 1) else estimate


def _assert_identity(
    left: RationalPolynomial,
    right: RationalPolynomial,
    quotient: RationalPolynomial,
    remainder: RationalPolynomial,
) -> None:
    reconstruction = _coefficients(remainder)
    for a, c in _coefficients(quotient).items():
        for b, d in _coefficients(right).items():
            reconstruction[a + b] = reconstruction.get(a + b, Fraction()) + c * d
    assert {e: c for e, c in reconstruction.items() if c} == _coefficients(left)
    assert (
        not remainder.polynomial.terms
        or remainder.polynomial.terms[0].exponents[0]
        < right.polynomial.terms[0].exponents[0]
    )


@pytest.mark.parametrize("public", (False, True))
def test_representable_prime_power_neighbor_round_trips(public: bool) -> None:
    left, right = _family(63)
    result = (
        multivariate_division(left, right)
        if public
        else rational_polynomial_division(left, right)
    )
    point = Fraction(1, 10**255)
    expected = sum(
        (Fraction(1, d) * point**i for i, d in enumerate(_denominators(64))), Fraction()
    )
    assert expected.denominator == 10 ** (255 * 63) * prod(_denominators(64))
    assert _digits(expected.denominator) == 32384
    assert _coefficients(result.remainder) == {0: expected}
    _assert_identity(left, right, result.quotient, result.remainder)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    if public:
        assert verify_multivariate_division(result)  # type: ignore[arg-type]


@pytest.mark.parametrize("public", (False, True))
def test_unrepresentable_prime_power_remainder_is_typed(public: bool) -> None:
    left, right = _family(64)
    point = Fraction(1, 10**255)
    expected = sum(
        (Fraction(1, d) * point**i for i, d in enumerate(_denominators(65))), Fraction()
    )
    assert expected.denominator == 10 ** (255 * 64) * prod(_denominators(65))
    assert _digits(expected.denominator) == 32894
    with pytest.raises(OperationResourceAdmissionError) as error:
        (multivariate_division if public else rational_polynomial_division)(left, right)
    assert error.value.errors()[0]["type"] == "polynomial.division.coefficient_height"


@pytest.mark.parametrize("public", (False, True))
def test_same_large_source_still_divides_itself(public: bool) -> None:
    left, _ = _family(64)
    result = (multivariate_division if public else rational_polynomial_division)(
        left, left
    )
    assert _coefficients(result.quotient) == {0: Fraction(1)}
    assert not result.remainder.polynomial.terms
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("order", ("lex", "grlex", "grevlex"))
def test_general_divisors_reconstruct_with_independent_fraction_oracle(
    order: Any,
) -> None:
    random = Random(4365)
    for _ in range(20):
        n, m = random.randrange(1, 9), random.randrange(1, 6)
        left = _poly(
            {
                i: Fraction(random.choice((-5, -3, 1, 2, 7)), random.randrange(1, 8))
                for i in range(n + 1)
            }
        )
        right = _poly(
            {
                i: Fraction(random.choice((-5, -3, 1, 2, 7)), random.randrange(1, 8))
                for i in range(m + 1)
            }
        )
        result = multivariate_division(left, right, order)
        native = rational_polynomial_division(left, right)
        assert (result.quotient, result.remainder) == (
            native.quotient,
            native.remainder,
        )
        _assert_identity(left, right, result.quotient, result.remainder)


@pytest.mark.parametrize(
    "divisor", (Fraction(1), Fraction(-1), Fraction(2, 3), Fraction(10**255, 7))
)
def test_large_common_denominators_do_not_inflate_monomial_outputs(
    divisor: Fraction,
) -> None:
    left, _ = _family(127)
    right = _poly({64: divisor})
    result = rational_polynomial_division(left, right)
    assert _coefficients(result.quotient) == {
        i - 64: c / divisor for i, c in _coefficients(left).items() if i >= 64
    }
    assert _coefficients(result.remainder) == {
        i: c for i, c in _coefficients(left).items() if i < 64
    }
    _assert_identity(left, right, result.quotient, result.remainder)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_zero_and_lower_degree_branches_keep_exact_sources() -> None:
    zero = _poly({})
    left, right = (
        _poly({2: Fraction(2, 3), 0: Fraction(1, 7)}),
        _poly({3: Fraction(1), 0: Fraction(1)}),
    )
    for source in (zero, left):
        result = rational_polynomial_division(source, right)
        assert result.quotient == zero
        assert result.remainder == source


def test_refusal_does_not_become_a_false_verifier_claim() -> None:
    left, right = _family(64)
    claim = multivariate_division(right, right).model_copy(
        update={"left": left, "right": right}
    )
    with pytest.raises(OperationResourceAdmissionError):
        verify_multivariate_division(claim)


@pytest.mark.parametrize(
    "change",
    (
        {"exponents": (-1,)},
        {"coefficient": CanonicalRational.model_construct(num=2, den=2)},
    ),
)
def test_new_admission_preserves_canonical_source_shape(
    change: dict[str, object],
) -> None:
    left = _poly({1: Fraction(1)})
    forged = left.model_copy(
        update={
            "polynomial": left.polynomial.model_copy(
                update={"terms": (left.polynomial.terms[0].model_copy(update=change),)}
            )
        }
    )
    with pytest.raises(OperationDomainValidationError):
        rational_polynomial_division(forged, _poly({0: Fraction(1)}))


def test_private_scalar_work_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _division_bounds

    monkeypatch.setattr(_division_bounds, "MAX_DIVISION_WORK", 1)
    assert _division_bounds._Ledger().multiply(2, 3) == 6
    monkeypatch.setattr(_division_bounds, "MAX_DIVISION_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as error:
        _division_bounds._Ledger().multiply(2, 3)
    assert error.value.errors()[0]["type"] == "polynomial.division.work"


def test_private_intermediate_bit_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _division_bounds

    monkeypatch.setattr(_division_bounds, "MAX_DIVISION_PRIVATE_BITS", 5)
    assert _division_bounds._Ledger().multiply(3, 4) == 12
    monkeypatch.setattr(_division_bounds, "MAX_DIVISION_PRIVATE_BITS", 4)
    with pytest.raises(OperationResourceAdmissionError) as error:
        _division_bounds._Ledger().multiply(3, 4)
    assert error.value.errors()[0]["type"] == "polynomial.division.intermediate_height"


@pytest.mark.parametrize(
    "limit,code",
    (
        ("MAX_DIVISION_ALLOCATION_BITS", "allocation"),
        ("MAX_DIVISION_RESULT_DIGITS", "output_digits"),
    ),
)
def test_complete_storage_and_output_limits_remain_operational(
    monkeypatch: pytest.MonkeyPatch, limit: str, code: str
) -> None:
    from jacobian.math.polynomials import _division_bounds

    monkeypatch.setattr(_division_bounds, limit, 0)
    with pytest.raises(OperationResourceAdmissionError) as error:
        rational_polynomial_division(_poly({1: Fraction(1)}), _poly({0: Fraction(1)}))
    assert error.value.errors()[0]["type"] == f"polynomial.division.{code}"
