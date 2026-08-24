from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._models import (
    _MAX_FACTORIZATION_LENGTH,
    ChineseRemainderRequest,
    FactorialValuationRequest,
    FactorizationRequest,
    ModularValueRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    with pytest.raises(ValidationError, match="canonical"):
        ChineseRemainderRequest(residues=(residue,), moduli=(3,))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"residues": [1, 2], "moduli": [3]}, "equal length"),
        ({"residues": [0], "moduli": [1]}, "between 2 and 1,000,000"),
        ({"residues": [0], "moduli": [1_000_001]}, "between 2 and 1,000,000"),
    ],
)
def test_chinese_remainder_rejects_invalid_system_bounds(
    payload: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChineseRemainderRequest.model_validate(payload)


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 10_001}),
        (NonnegativeIntegerRequest, {"n": 10_001}),
        (ModularValueRequest, {"value": "2", "modulus": 1_000_001}),
        (FactorialValuationRequest, {"n": 1, "base": 1_000_001}),
        (FactorizationRequest, {"value": "1" + "0" * 20}),
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


def test_divisor_list_result_admits_twenty_digit_source_boundary() -> None:
    """The replayed source keeps the producing operations' 20-digit bound."""

    from sympy import isprime

    from jacobian.math.number_theory._models import DivisorListResult

    prime = 99_999_999_999_999_999_989
    assert len(str(prime)) == _MAX_FACTORIZATION_LENGTH == 20
    assert isprime(prime)
    result = DivisorListResult(value=str(prime), divisors=("1", str(prime)))
    assert result.value == str(prime)

    with pytest.raises(ValidationError, match="value"):
        DivisorListResult.model_validate(
            {
                "value": "10" + "0" * 19,
                "divisors": ["1"],
                "convention": "ALL_POSITIVE_DIVISORS",
            }
        )


def test_divisor_list_result_rejects_sources_beyond_factorization_domain() -> None:
    """A forged serialized output whose divisor list does not enumerate the
    source's divisors exactly is rejected by the source-bound replay."""

    from jacobian.math.number_theory._models import DivisorListResult

    with pytest.raises(
        ValidationError, match="divisor list must enumerate the divisors"
    ):
        DivisorListResult.model_validate(
            {
                "value": "12",
                "divisors": ["1", "2", "3", "12"],
                "convention": "ALL_POSITIVE_DIVISORS",
            }
        )


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


def test_prime_factorization_result_bounds_reconstruction_work() -> None:
    """Replay rejects reconstructions larger than the source before expanding."""

    from jacobian.math.number_theory._models import PrimeFactorizationResult, PrimePower

    with pytest.raises(ValidationError, match="multiply to abs"):
        PrimeFactorizationResult(
            value="3", factors=(PrimePower(prime="2", power=1000),)
        )
    with pytest.raises(ValidationError, match="multiply to abs"):
        PrimeFactorizationResult(
            value="6",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="3", power=1000)),
        )
    with pytest.raises(ValidationError, match="multiply to abs"):
        PrimeFactorizationResult(
            value="1024",
            factors=(
                PrimePower(prime="2", power=10),
                PrimePower(prime="3", power=999),
                PrimePower(prime="5", power=999),
            ),
        )


def test_prime_factorization_result_admits_full_width_source_power() -> None:
    """A source-width prime power still reconstructs exactly."""

    from jacobian.math.number_theory._models import PrimeFactorizationResult

    width = 256
    exponent = 849
    result = PrimeFactorizationResult.model_validate(
        {
            "value": str(2**exponent),
            "factors": [{"prime": "2", "power": exponent}],
        }
    )
    assert len(result.value) == width
    assert int(result.factors[0].prime) ** result.factors[0].power == 2**exponent


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


@pytest.mark.parametrize(
    ("operation_id", "version"),
    [
        ("integer.compute.divisors", "4"),
        ("integer.compute.proper_divisors", "4"),
        ("integer.compute.prime_factorization", "3"),
    ],
)
def test_source_bound_result_schema_changes_bump_operation_version(
    operation_id: str, version: str
) -> None:
    from jacobian.math.number_theory._factorization import FACTORIZATION_OPERATIONS

    operation = next(
        item for item in FACTORIZATION_OPERATIONS if item.operation_id == operation_id
    )
    assert operation.version == version
