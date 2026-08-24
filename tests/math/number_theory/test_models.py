from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._models import (
    ChineseRemainderRequest,
    FactorialValuationRequest,
    FactorizationRequest,
    ModularValueRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    PowerfulNumberResult,
)


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    with pytest.raises(ValidationError, match="canonical"):
        ChineseRemainderRequest(residues=(residue,), moduli=(3,))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"residues": [1, 2], "moduli": [3]}, "equal length"),
        ({"residues": [0], "moduli": [1]}, "between 2 and 10,000"),
        ({"residues": [0], "moduli": [10_001]}, "between 2 and 10,000"),
    ],
)
def test_chinese_remainder_rejects_invalid_system_bounds(
    payload: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChineseRemainderRequest.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [{"prime": "2", "power": 3}],
            "violating_primes": [],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [{"prime": "2", "power": 1}],
            "violating_primes": ["2"],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [
                {"prime": "3", "power": 1},
                {"prime": "2", "power": 2},
            ],
            "violating_primes": ["3"],
        },
    ),
)
def test_powerful_number_result_rejects_inconsistent_or_noncanonical_witnesses(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"powerful|factor"):
        PowerfulNumberResult.model_validate(payload)


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 1_001}),
        (NonnegativeIntegerRequest, {"n": 1_001}),
        (ModularValueRequest, {"value": "2", "modulus": 10_001}),
        (FactorialValuationRequest, {"n": 1, "base": 1_000_001}),
        (FactorizationRequest, {"value": "1000000000000"}),
    ):
        with pytest.raises(ValidationError, match=r"less than or equal|at most"):
            model.model_validate(payload)


# ---------------------------------------------------------------------------
# Source-bound divisor and prime-factorization results (#2311)
# ---------------------------------------------------------------------------


def test_divisor_list_result_replays_source_enumeration() -> None:
    from jacobian.math.number_theory._models import DivisorListResult

    full = DivisorListResult(value="12", divisors=("1", "2", "3", "4", "6", "12"))
    assert full.convention == "ALL_POSITIVE_DIVISORS"
    proper = DivisorListResult(
        value="-12",
        divisors=("1", "2", "3", "4", "6"),
        convention="PROPER_DIVISORS",
    )
    assert proper.divisors == ("1", "2", "3", "4", "6")

    one_full = DivisorListResult(value="1", divisors=("1",))
    assert one_full.divisors == ("1",)
    one_proper = DivisorListResult(
        value="1",
        divisors=(),
        convention="PROPER_DIVISORS",
    )
    assert one_proper.divisors == ()
    minus_one = DivisorListResult(value="-1", divisors=("1",))
    assert minus_one.divisors == ("1",)


def test_divisor_list_result_rejects_mutations() -> None:
    from jacobian.math.number_theory._models import DivisorListResult

    base = {
        "value": "12",
        "divisors": ("1", "2", "3", "4", "6", "12"),
        "convention": "ALL_POSITIVE_DIVISORS",
    }
    with pytest.raises(ValidationError, match="enumerate the divisors"):
        DivisorListResult(**{**base, "value": "99"})
    with pytest.raises(ValidationError, match="enumerate the divisors"):
        DivisorListResult(**{**base, "divisors": ("1", "2", "3", "4", "6")})
    with pytest.raises(ValidationError, match="enumerate the divisors"):
        DivisorListResult(**{**base, "divisors": ("1", "2", "3", "4", "6", "8", "12")})
    with pytest.raises(ValidationError, match="ascending"):
        DivisorListResult(
            value="12",
            divisors=("12", "6", "4", "3", "2", "1"),
            convention="ALL_POSITIVE_DIVISORS",
        )
    with pytest.raises(ValidationError, match="enumerate the divisors"):
        DivisorListResult(
            value="12",
            divisors=("2", "99"),
            convention="ALL_POSITIVE_DIVISORS",
        )
    with pytest.raises(ValidationError, match="zero has infinitely many"):
        DivisorListResult(value="0", divisors=())
    with pytest.raises(ValidationError, match="enumerate the divisors"):
        DivisorListResult(
            value="12",
            divisors=("1", "2", "3", "4", "6"),
            convention="ALL_POSITIVE_DIVISORS",
        )


def test_prime_factorization_result_replays_source() -> None:
    from jacobian.math.number_theory._models import PrimeFactorizationResult, PrimePower

    result = PrimeFactorizationResult(
        value="72",
        factors=(PrimePower(prime="2", power=3), PrimePower(prime="3", power=2)),
    )
    assert result.factors[0].prime == "2"
    empty_one = PrimeFactorizationResult(value="1", factors=())
    minus_one = PrimeFactorizationResult(value="-1", factors=())
    assert not empty_one.factors and not minus_one.factors
    prime_power = PrimeFactorizationResult(
        value="-8", factors=(PrimePower(prime="2", power=3),)
    )
    assert prime_power.factors[0].power == 3


def test_prime_factorization_result_rejects_mutations() -> None:
    from jacobian.math.number_theory._models import PrimeFactorizationResult, PrimePower

    with pytest.raises(ValidationError, match="not prime"):
        PrimeFactorizationResult(value="4", factors=(PrimePower(prime="4", power=1),))
    with pytest.raises(ValidationError, match="multiply to abs"):
        PrimeFactorizationResult(
            value="12",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="3", power=1)),
        )
    with pytest.raises(ValidationError, match="strictly ascending"):
        PrimeFactorizationResult(
            value="12",
            factors=(PrimePower(prime="3", power=1), PrimePower(prime="2", power=2)),
        )
    with pytest.raises(ValidationError, match="unique"):
        PrimeFactorizationResult(
            value="4",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="2", power=2)),
        )
    with pytest.raises(ValidationError, match="zero has no finite"):
        PrimeFactorizationResult(value="0", factors=())


def test_producer_results_serialize_and_reconstruct() -> None:
    """Producer output round-trips and reconstructs its exact source."""

    import math

    from jacobian.math.number_theory._factorization_kernels import (
        enumerate_divisors,
        enumerate_proper_divisors,
        factorize_primes,
    )
    from jacobian.math.number_theory._models import (
        DivisorListResult,
        FactorizationRequest,
        PrimeFactorizationResult,
    )

    request = FactorizationRequest(value="72")
    factorization = PrimeFactorizationResult.model_validate(
        factorize_primes(request).model_dump()
    )
    assert math.prod(
        int(factor.prime) ** factor.power for factor in factorization.factors
    ) == abs(int(factorization.value))

    full = DivisorListResult.model_validate(enumerate_divisors(request).model_dump())
    proper = DivisorListResult.model_validate(
        enumerate_proper_divisors(request).model_dump()
    )
    assert len(proper.divisors) == len(full.divisors) - 1
    pairs = list(zip(full.divisors, reversed(full.divisors), strict=True))
    assert all(int(a) * int(b) == 72 for a, b in pairs)
