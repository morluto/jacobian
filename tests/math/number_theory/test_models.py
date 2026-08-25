from __future__ import annotations

import pytest
from tests.math.number_theory._validation import expect_validation

from jacobian.math.number_theory._direct_factorization_models import (
    MAX_DIRECT_FACTORIZATION_DIGITS,
    DivisorListResult,
    FactorizationRequest,
    PrimeFactorizationResult,
)
from jacobian.math.number_theory._models import (
    _MAX_CRT_SIZE,
    _MAX_INTEGER_LENGTH,
    ChineseRemainderRequest,
    FactorialValuationRequest,
    ModularValueRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    PrimalityRequest,
)
from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
)


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    with expect_validation("number_theory."):
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
    with expect_validation("number_theory."):
        ChineseRemainderRequest.model_validate(payload)


def test_chinese_remainder_rejects_combined_modulus_beyond_result_budget() -> None:
    """64 pairwise-coprime six-digit moduli each fit the per-modulus bound
    while their LCM exceeds the declared 256-character ``BoundedInteger``
    result width: admission must bound the combined modulus, not each
    modulus alone."""
    from sympy import prevprime

    moduli: list[int] = []
    candidate = 1_000_000
    while len(moduli) < _MAX_CRT_SIZE:
        candidate = int(prevprime(candidate))
        moduli.append(candidate)

    with expect_validation("number_theory."):
        ChineseRemainderRequest(residues=(1,) * len(moduli), moduli=tuple(moduli))


def test_chinese_remainder_admits_boundary_system_and_solves_exactly() -> None:
    """A compatible system whose combined modulus fits the result budget is
    admitted and solved; the typed result carries the system's LCM exactly."""

    from math import lcm

    from sympy import prevprime

    from jacobian.math.number_theory._modular_operations import (
        solve_chinese_remainder,
    )

    moduli: list[int] = []
    combined = 1
    candidate = 1_000_000
    while True:
        candidate = int(prevprime(candidate))
        if len(str(combined * candidate)) > _MAX_INTEGER_LENGTH:
            break
        moduli.append(candidate)
        combined *= candidate
    assert len(str(combined)) >= _MAX_INTEGER_LENGTH - 6

    request = ChineseRemainderRequest(residues=(1,) * len(moduli), moduli=tuple(moduli))
    result = solve_chinese_remainder(request)

    assert result.residue == "1"
    assert result.modulus == str(combined)
    assert int(result.modulus) == lcm(*moduli)
    assert len(result.modulus) <= _MAX_INTEGER_LENGTH


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 10_001}),
        (NonnegativeIntegerRequest, {"n": 10_001}),
        (ModularValueRequest, {"value": "2", "modulus": 1_000_001}),
        (FactorialValuationRequest, {"n": 1, "base": 1_000_001}),
        (FactorizationRequest, {"value": "1" + "0" * 20}),
    ):
        with expect_validation("number_theory."):
            model.model_validate(payload)


def test_primality_keeps_its_operation_specific_input_bound() -> None:
    with expect_validation("string_too_long"):
        PrimalityRequest(value="1" + "0" * _MAX_INTEGER_LENGTH)


def test_direct_factorization_contract_schemas_preserve_their_envelopes() -> None:
    """Moving direct-factorization contracts must not widen public schemas."""

    request_value = FactorizationRequest.model_json_schema()["properties"]["value"]
    divisor_source = DivisorListResult.model_json_schema()["properties"]["value"]
    factorization_source = PrimeFactorizationResult.model_json_schema()["properties"][
        "value"
    ]

    assert request_value["maxLength"] == MAX_DIRECT_FACTORIZATION_DIGITS
    assert divisor_source["maxLength"] == MAX_DIRECT_FACTORIZATION_DIGITS
    assert factorization_source["maxLength"] == _MAX_INTEGER_LENGTH


def test_modular_residue_image_contract_replays_canonical_assignments() -> None:
    from jacobian.math.number_theory._modular_operations import (
        compute_modular_polynomial_residue_assignments,
    )

    request = ModularPolynomialResidueImageRequest.model_validate(
        {
            "modulus": 5,
            "variables": [{"name": "x", "residues": [0, 1, 2]}],
            "terms": [{"coefficient": "2", "exponents": [2]}],
        }
    )

    result = compute_modular_polynomial_residue_assignments(request)

    assert request.__class__.model_json_schema()["title"] == (
        "ModularPolynomialResidueImageRequest"
    )
    assert result.image == (0, 2, 3)
    assert tuple(row.model_dump() for row in result.table or ()) == (
        {"assignment": (0,), "residue": 0},
        {"assignment": (1,), "residue": 2},
        {"assignment": (2,), "residue": 3},
    )


# ---------------------------------------------------------------------------
# Source-bound divisor and prime-factorization results (#2311)
# ---------------------------------------------------------------------------


def test_divisor_list_result_replays_source_enumeration() -> None:
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

    prime = 99_999_999_999_999_999_989
    assert len(str(prime)) == MAX_DIRECT_FACTORIZATION_DIGITS == 20
    assert isprime(prime)
    result = DivisorListResult(value=str(prime), divisors=("1", str(prime)))
    assert result.value == str(prime)

    with expect_validation("number_theory."):
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

    with expect_validation("number_theory."):
        DivisorListResult.model_validate(
            {
                "value": "12",
                "divisors": ["1", "2", "3", "12"],
                "convention": "ALL_POSITIVE_DIVISORS",
            }
        )


def test_divisor_list_result_rejects_mutations() -> None:
    base = {
        "value": "12",
        "divisors": ("1", "2", "3", "4", "6", "12"),
        "convention": "ALL_POSITIVE_DIVISORS",
    }
    with expect_validation("number_theory."):
        DivisorListResult(**{**base, "value": "99"})
    with expect_validation("number_theory."):
        DivisorListResult(**{**base, "divisors": ("1", "2", "3", "4", "6")})
    with expect_validation("number_theory."):
        DivisorListResult(**{**base, "divisors": ("1", "2", "3", "4", "6", "8", "12")})
    with expect_validation("number_theory."):
        DivisorListResult(
            value="12",
            divisors=("12", "6", "4", "3", "2", "1"),
            convention="ALL_POSITIVE_DIVISORS",
        )
    with expect_validation("number_theory."):
        DivisorListResult(
            value="12",
            divisors=("2", "99"),
            convention="ALL_POSITIVE_DIVISORS",
        )
    with expect_validation("number_theory."):
        DivisorListResult(value="0", divisors=())
    with expect_validation("number_theory."):
        DivisorListResult(
            value="12",
            divisors=("1", "2", "3", "4", "6"),
            convention="ALL_POSITIVE_DIVISORS",
        )


def test_prime_factorization_result_replays_source() -> None:
    from jacobian.math.number_theory._models import PrimePower

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
    from jacobian.math.number_theory._models import PrimePower

    with expect_validation("number_theory."):
        PrimeFactorizationResult(value="4", factors=(PrimePower(prime="4", power=1),))
    with expect_validation("number_theory."):
        PrimeFactorizationResult(
            value="12",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="3", power=1)),
        )
    with expect_validation("number_theory."):
        PrimeFactorizationResult(
            value="12",
            factors=(PrimePower(prime="3", power=1), PrimePower(prime="2", power=2)),
        )
    with expect_validation("number_theory."):
        PrimeFactorizationResult(
            value="4",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="2", power=2)),
        )
    with expect_validation("number_theory."):
        PrimeFactorizationResult(value="0", factors=())


def test_prime_factorization_result_bounds_reconstruction_work() -> None:
    """Replay rejects reconstructions larger than the source before expanding."""

    from jacobian.math.number_theory._models import PrimePower

    with expect_validation("number_theory."):
        PrimeFactorizationResult(
            value="3", factors=(PrimePower(prime="2", power=1000),)
        )
    with expect_validation("number_theory."):
        PrimeFactorizationResult(
            value="6",
            factors=(PrimePower(prime="2", power=1), PrimePower(prime="3", power=1000)),
        )
    with expect_validation("number_theory."):
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
