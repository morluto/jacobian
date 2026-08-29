from __future__ import annotations

import math

import pytest
from tests.math.number_theory._validation import expect_validation

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._derived import (
    compute_binomial_prime_valuation,
    compute_factorial_valuation,
    compute_floor_square_root,
    compute_legendre_symbol,
)
from jacobian.math.number_theory._derived_models import (
    BinomialPrimeValuationRequest,
    FactorialValuationRequest,
    FloorSquareRootRequest,
    LegendreSymbolRequest,
)
from jacobian.math.number_theory._direct_factorization_models import (
    MAX_DIRECT_FACTORIZATION_DIGITS,
    DivisorListResult,
    FactorizationRequest,
    PrimeFactorizationResult,
)
from jacobian.math.number_theory._integer_models import (
    MAX_SAFE_INTEGER,
    ArithmeticFunctionRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS
from jacobian.math.number_theory._modular_basic_models import (
    MAX_CRT_SIZE,
    ChineseRemainderRequest,
    ModularValueRequest,
)
from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
)
from jacobian.math.number_theory._prime_models import (
    PreviousPrimeRequest,
    PrimalityRequest,
)
from jacobian.math.number_theory._primes import compute_previous_prime
from jacobian.math.number_theory.operations import chinese_remainder


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    request = ChineseRemainderRequest(residues=(residue,), moduli=(3,))
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        chinese_remainder(request.residues, request.moduli)


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
    request = ChineseRemainderRequest.model_validate(payload)
    with pytest.raises(OperationDomainValidationError, match=message):
        chinese_remainder(request.residues, request.moduli)


@pytest.mark.parametrize("prime", (3, 97, 9_999_991))
def test_legendre_request_admits_prime_denominators_without_a_backend(
    prime: int,
) -> None:
    assert LegendreSymbolRequest(a=2, prime=prime).prime == prime


@pytest.mark.parametrize("composite", (9, 99, 9_999_999))
def test_legendre_request_rejects_composite_denominators(composite: int) -> None:
    request = LegendreSymbolRequest(a=2, prime=composite)
    with pytest.raises(
        OperationDomainValidationError, match="Legendre denominator must be prime"
    ):
        compute_legendre_symbol(request)


def test_constant_work_integer_operations_admit_safe_integer_scale() -> None:
    square_root = compute_floor_square_root(FloorSquareRootRequest(n=MAX_SAFE_INTEGER))
    previous = compute_previous_prime(PreviousPrimeRequest(n=MAX_SAFE_INTEGER))
    legendre = compute_legendre_symbol(LegendreSymbolRequest(a=2, prime=1_000_000_007))

    assert square_root.root**2 <= MAX_SAFE_INTEGER < (square_root.root + 1) ** 2
    assert int(previous.value) < MAX_SAFE_INTEGER
    assert legendre.symbol == 1


def test_killable_arithmetic_function_request_admits_safe_integer_scale() -> None:
    assert ArithmeticFunctionRequest(n=MAX_SAFE_INTEGER).n == MAX_SAFE_INTEGER


def test_factorial_valuation_accepts_large_canonical_argument() -> None:
    n = 10**100
    result = compute_factorial_valuation(FactorialValuationRequest(n=str(n), base="2"))

    assert result.n == str(n)
    assert int(result.valuation) == n - n.bit_count()


@pytest.mark.parametrize(("n", "k", "prime", "expected"), ((8, 3, 2, 3), (10, 3, 3, 1)))
def test_scalar_binomial_prime_valuation(
    n: int, k: int, prime: int, expected: int
) -> None:
    result = compute_binomial_prime_valuation(
        BinomialPrimeValuationRequest(n=str(n), k=str(k), prime=str(prime))
    )

    assert result.valuation == str(expected)


def test_source_scale_binomial_valuation_matches_factorial_identity() -> None:
    n, k, prime = 99_999_937, 40_000_001, 2
    result = compute_binomial_prime_valuation(
        BinomialPrimeValuationRequest(n=str(n), k=str(k), prime=str(prime))
    )

    def factorial_exponent(value: int) -> int:
        return value - value.bit_count()

    assert int(result.valuation) == (
        factorial_exponent(n) - factorial_exponent(k) - factorial_exponent(n - k)
    )


def test_kummer_carries_match_direct_binomial_factorization() -> None:
    for n in range(21):
        for k in range(n + 1):
            for prime in (2, 3, 5, 7):
                result = compute_binomial_prime_valuation(
                    BinomialPrimeValuationRequest(n=str(n), k=str(k), prime=str(prime))
                )
                value = math.comb(n, k)
                expected = 0
                while value % prime == 0:
                    value //= prime
                    expected += 1
                assert result.valuation == str(expected)


def test_scalar_binomial_valuation_rejects_composite_base() -> None:
    request = BinomialPrimeValuationRequest(n="20", k="7", prime="4")
    with pytest.raises(OperationDomainValidationError, match="prime must be prime"):
        compute_binomial_prime_valuation(request)


def test_valuation_request_schemas_publish_semantic_bounds() -> None:
    factorial = FactorialValuationRequest.model_json_schema()["properties"]
    binomial = BinomialPrimeValuationRequest.model_json_schema()["properties"]

    assert "nonnegative" in factorial["n"]["description"].lower()
    assert "[2, 1000000]" in factorial["base"]["description"]
    assert "0 <= k <= n" in binomial["n"]["description"]
    assert "0 <= k <= n" in binomial["k"]["description"]
    assert str(2**64 - 1) in binomial["prime"]["description"]


def test_valuation_admission_follows_copied_request_fields() -> None:
    request = BinomialPrimeValuationRequest(n="8", k="3", prime="2")
    assert compute_binomial_prime_valuation(request).valuation == "3"
    copied = request.model_copy(update={"n": "20", "k": "7"})
    assert compute_binomial_prime_valuation(copied).valuation == "4"
    assert copied.n == "20"


def test_chinese_remainder_rejects_combined_modulus_beyond_result_budget() -> None:
    """64 pairwise-coprime six-digit moduli each fit the per-modulus bound
    while their LCM exceeds the declared 256-character ``BoundedInteger``
    result width: admission must bound the combined modulus, not each
    modulus alone."""
    from sympy import prevprime

    moduli: list[int] = []
    candidate = 1_000_000
    while len(moduli) < MAX_CRT_SIZE:
        candidate = int(prevprime(candidate))
        moduli.append(candidate)

    request = ChineseRemainderRequest(residues=(1,) * len(moduli), moduli=tuple(moduli))
    with pytest.raises(OperationDomainValidationError, match="combined modulus"):
        chinese_remainder(request.residues, request.moduli)


def test_chinese_remainder_admits_boundary_system_and_solves_exactly() -> None:
    """A compatible system whose combined modulus fits the result budget is
    admitted and solved; the typed result carries the system's LCM exactly."""

    from math import lcm

    from sympy import prevprime

    from jacobian.math.number_theory.operations import chinese_remainder

    moduli: list[int] = []
    combined = 1
    candidate = 1_000_000
    while True:
        candidate = int(prevprime(candidate))
        if len(str(combined * candidate)) > MAX_INTEGER_DIGITS:
            break
        moduli.append(candidate)
        combined *= candidate
    assert len(str(combined)) >= MAX_INTEGER_DIGITS - 6

    request = ChineseRemainderRequest(residues=(1,) * len(moduli), moduli=tuple(moduli))
    result = chinese_remainder(request.residues, request.moduli)

    assert result.residue == "1"
    assert result.modulus == str(combined)
    assert int(result.modulus) == lcm(*moduli)
    assert len(result.modulus) <= MAX_INTEGER_DIGITS


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 10_001}),
        (NonnegativeIntegerRequest, {"n": 10_001}),
        (ModularValueRequest, {"value": "2", "modulus": 1_000_001}),
        # Eight digits exceeds the published base digit ceiling (len("1000000")).
        (FactorialValuationRequest, {"n": "1", "base": "10000000"}),
        (FactorizationRequest, {"value": "1" + "0" * 20}),
    ):
        with expect_validation("number_theory."):
            model.model_validate(payload)


def test_primality_keeps_its_operation_specific_input_bound() -> None:
    with expect_validation("string_too_long"):
        PrimalityRequest(value="1" + "0" * MAX_INTEGER_DIGITS)


def test_direct_factorization_contract_schemas_preserve_their_envelopes() -> None:
    """Moving direct-factorization contracts must not widen public schemas."""

    request_value = FactorizationRequest.model_json_schema()["properties"]["value"]
    divisor_source = DivisorListResult.model_json_schema()["properties"]["value"]
    factorization_source = PrimeFactorizationResult.model_json_schema()["properties"][
        "value"
    ]

    assert request_value["maxLength"] == MAX_DIRECT_FACTORIZATION_DIGITS
    assert divisor_source["maxLength"] == MAX_DIRECT_FACTORIZATION_DIGITS
    assert factorization_source["maxLength"] == MAX_DIRECT_FACTORIZATION_DIGITS


def test_modular_residue_image_contract_round_trips_canonical_assignments() -> None:
    from jacobian.math.number_theory.operations import (
        modular_polynomial_residue_assignments,
    )

    request = ModularPolynomialResidueImageRequest.model_validate(
        {
            "modulus": 5,
            "variables": [{"name": "x", "residues": [0, 1, 2]}],
            "terms": [{"coefficient": "2", "exponents": [2]}],
        }
    )

    result = modular_polynomial_residue_assignments(
        request.modulus, request.variables, request.terms
    )

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


def test_divisor_list_result_preserves_structural_constraints() -> None:
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
    """The result keeps the producing operations' 20-digit source bound."""

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


def test_divisor_list_result_rejects_mutations() -> None:
    with expect_validation("number_theory."):
        DivisorListResult(
            value="12",
            divisors=("12", "6", "4", "3", "2", "1"),
            convention="ALL_POSITIVE_DIVISORS",
        )


def test_direct_factorization_admits_nonzero_values_in_models() -> None:
    from jacobian.math.number_theory._factorization_kernels import (
        enumerate_divisors,
        enumerate_proper_divisors,
        factorize_primes,
    )

    request = FactorizationRequest(value="0")
    for operation in (enumerate_divisors, enumerate_proper_divisors, factorize_primes):
        with pytest.raises(OperationDomainValidationError, match="zero"):
            operation(request)


def test_prime_factorization_result_accepts_kernel_shape() -> None:
    from jacobian.math.number_theory._integer_models import PrimePower

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


def test_prime_factorization_result_preserves_structural_constraints() -> None:
    from jacobian.math.number_theory._integer_models import PrimePower

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


def test_prime_factorization_result_rejects_source_beyond_worker_envelope() -> None:
    """The result retains the producer's bounded source envelope."""

    width = 256
    exponent = 849
    assert len(str(2**exponent)) == width > MAX_DIRECT_FACTORIZATION_DIGITS
    with expect_validation("string_too_long"):
        PrimeFactorizationResult.model_validate(
            {
                "value": str(2**exponent),
                "factors": [{"prime": "2", "power": exponent}],
            }
        )


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


def test_divisor_enumeration_uses_the_twenty_digit_source_envelope() -> None:
    from jacobian.math.number_theory._factorization_kernels import enumerate_divisors

    # This is the least integer with more than the former 4096-divisor ceiling.
    result = enumerate_divisors(FactorizationRequest(value="146659312800"))

    assert result.status == "COMPLETE"
    assert len(result.divisors) == 4320
    assert result.divisors[0] == "1"
    assert result.divisors[-1] == result.value
