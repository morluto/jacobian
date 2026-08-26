"""Named Pydantic wire contracts for exact integer number-theory operations.

These contracts cover gcd/lcm, Bezout coefficients, divisors, prime
factorization, p-adic valuation, multiplicative arithmetic functions,
primality, friable counting, modular arithmetic, and integer predicates (coprimality,
divisibility, perfect/abundant/deficient, square, squarefree).  They are
owned by the number-theory domain and intentionally exclude arithmetic-owned
operations (absolute value, sign, decimal digit sum/count, base expansion,
integer nth root).
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable semantic error owned by the number-theory domain."""

    return PydanticCustomError(f"number_theory.{reason}", message)


# ---------------------------------------------------------------------------
# Shared bounds for the current bounded integer-domain contracts.
# ---------------------------------------------------------------------------

_MAX_INTEGER_LENGTH = 256
MAX_POWERFUL_INTEGER_DIGITS = 25
MAX_POWERFUL_CUTOFF = 100_000
MAX_POWERFUL_FACTOR_ENTRIES = 42
MAX_POWERFUL_EXPONENT = 83
# Certified factoring uses SymPy's ``factorint`` (Pollard rho, Pollard p-1,
# ECM) on the input and recursively on ``p - 1`` for Pratt certificates.
# The 30-digit bound (~100 bits) is a real work bound: it keeps worst-case
# synchronous factoring of hard semiprimes (e.g., two ~15-digit primes) and
# Pratt ``p - 1`` factorization bounded to well under one second, while
# still covering the 21-digit subexponential test vector.  An 80-digit cap
# would admit inputs whose Pollard rho/ECM work is unbounded for a
# synchronous ``math.run`` worker, so the admitted domain is narrowed here
# and documented as an algorithmic budget.
_MAX_CERTIFIED_FACTORIZATION_LENGTH = 30
# ``_MAX_N_SMALL`` covers arithmetic functions that may factor their input
# (totient, Möbius, divisor sigma, square-free predicates, and
# multiplicative order).  The 10_000 bound keeps SymPy factoring safe for
# in-process execution while admitting materially larger useful cases than
# the prior 1_000 cap.  Primorial has its own request bound derived from
# the declared result digit budget (see ``_MAX_PRIMORIAL_N``).
_MAX_N_SMALL = 10_000
# primorial(n) carries n(ln n + ln ln n)/ln 10 digits.  The declared
# result budget is ``_MAX_PRIMORIAL_DIGITS`` (3_400), and primorial(1001)
# already has 3397 digits while primorial(1002) has 3401, so the exact
# admitted boundary is n <= 1001.  Defined here so ``PrimorialRequest``
# can derive its own request-side guard from the output contract.
_MAX_PRIMORIAL_N = 1001
# ``_MAX_MODULUS`` is shared across modular inverse, multiplicative order,
# quadratic residues, CRT, Jacobi symbol, and brute-force discrete log.
# Raised to 1_000_000 for non-enumeration ops (inverse, order, CRT, Jacobi
# are O(log m)).  Quadratic residues at 1M enumerates ~500k entries
# (worst case ~10 MiB JSON) and relies on existing output-size limits.
# Brute-force discrete log is O(m) — 200k ~12ms, 1M ~60ms — so the uniform
# 1M cap makes discrete log heavy; a future BSGS implementation should
# replace the brute force before further raising this bound.
_MAX_MODULUS = 1_000_000
_MAX_CRT_SIZE = 64
# CRT admission derives its input envelope from the declared output
# contract: ``ChineseRemainderResult.modulus`` is a ``BoundedInteger`` of
# at most ``_MAX_INTEGER_LENGTH`` characters, so the LCM of an admitted
# system must stay within the same width.  ``10 ** _MAX_INTEGER_LENGTH``
# is the smallest excluded combined modulus (positive values only).
_MAX_CRT_COMBINED_MODULUS = 10**_MAX_INTEGER_LENGTH

# Friable counting has two exact, result-sensitive execution regimes. The
# materialized regime uses one bytearray for primality and one for friability;
# its work bound covers both the operation and result-validation replay. The
# generated regime enumerates prime-exponent prefixes only when a conservative
# prefix-box bound covers both passes.
MAX_FRIABLE_MATERIALIZED_X = 1_000_000
MAX_FRIABLE_GENERATED_CUTOFF = 10_000
_MAX_FRIABLE_MATERIALIZED_TOTAL_STEPS = 82_000_000
_MAX_FRIABLE_MATERIALIZED_BYTES = 3_000_000
_MAX_FRIABLE_GENERATED_TOTAL_NODES = 1_000_000
# Sources are nonnegative and rejected at ``10**_MAX_FRIABLE_SOURCE_DIGITS`` or
# above, so every admitted value carries at most this many decimal digits.
_MAX_FRIABLE_SOURCE_DIGITS = 256
_MAX_FRIABLE_SOURCE_ABS = 10**_MAX_FRIABLE_SOURCE_DIGITS

BoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=_MAX_INTEGER_LENGTH,
        strict=True,
    ),
]


class PrimalityRequest(StrictModel):
    """One bounded canonical integer for the maintained primality backend."""

    value: BoundedInteger


PowerfulInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^[1-9][0-9]*$",
        max_length=MAX_POWERFUL_INTEGER_DIGITS,
        strict=True,
    ),
]
CertifiedFactorizationInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=_MAX_CERTIFIED_FACTORIZATION_LENGTH,
        strict=True,
    ),
]

type _FriableRegime = Literal["DIRECT", "MATERIALIZED", "GENERATED"]


def _primes_through(limit: int) -> tuple[int, ...]:
    """Return the primes at most a small admitted cutoff."""

    if limit < 2:
        return ()
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for candidate in range(2, math.isqrt(limit) + 1):
        if not sieve[candidate]:
            continue
        first_composite = candidate * candidate
        composite_count = (limit - first_composite) // candidate + 1
        sieve[first_composite : limit + 1 : candidate] = b"\x00" * composite_count
    return tuple(index for index, is_prime in enumerate(sieve) if is_prime)


def _maximum_exponent(x: int, prime: int) -> int:
    """Return the largest ``exponent`` with ``prime**exponent <= x``."""

    exponent = 0
    remaining = x
    while remaining >= prime:
        remaining //= prime
        exponent += 1
    return exponent


def _plan_friable_count(x: int, y: int) -> tuple[_FriableRegime, tuple[int, ...]]:
    """Validate and select one exact friable-count execution regime."""

    if x < 0 or y < 0:
        raise _validation_error(
            "friable_count_sources_must_be_nonnegative",
            "friable-count sources must be nonnegative",
        )
    if x >= _MAX_FRIABLE_SOURCE_ABS or y >= _MAX_FRIABLE_SOURCE_ABS:
        raise _validation_error(
            f"friable_count_sources_must_have_at_most_{_MAX_FRIABLE_SOURCE_DIGITS}_decimal_digits",
            f"friable-count sources must have at most {_MAX_FRIABLE_SOURCE_DIGITS} decimal digits",
        )
    if x == 0 or y <= 1 or y >= x:
        return "DIRECT", ()

    # Each materialized pass marks at most two harmonic series of multiples,
    # plus one scan. Result validation replays the full exact computation.
    per_pass_steps = x * (2 * x.bit_length() + 1)
    materialized_bytes = 2 * (x + 1) + x // 2
    if (
        x <= MAX_FRIABLE_MATERIALIZED_X
        and 2 * per_pass_steps <= _MAX_FRIABLE_MATERIALIZED_TOTAL_STEPS
        and materialized_bytes <= _MAX_FRIABLE_MATERIALIZED_BYTES
    ):
        return "MATERIALIZED", ()

    if y > MAX_FRIABLE_GENERATED_CUTOFF:
        raise _validation_error(
            "generated_friable_counting_exceeds_the_admitted_prime_cutoff",
            "generated friable counting exceeds the admitted prime cutoff",
        )

    primes = _primes_through(y)
    nodes_per_pass = 1
    prefix_box = 1
    for prime in primes:
        prefix_box *= _maximum_exponent(x, prime) + 1
        nodes_per_pass += prefix_box
        if 2 * nodes_per_pass > _MAX_FRIABLE_GENERATED_TOTAL_NODES:
            raise _validation_error(
                "generated_friable_counting_exceeds_the_search_node_budget",
                "generated friable counting exceeds the search-node budget",
            )
    return "GENERATED", primes


class PowerfulNumberRequest(StrictModel):
    """One positive canonical integer of at most 25 digits."""

    value: PowerfulInteger = Field(
        description=(
            "Positive canonical decimal integer with at most 25 digits. The "
            "kernel derives B=ceil(value^(1/5)), so B <= 100000."
        ),
        examples=["12168"],
    )


class ArithmeticFunctionRequest(StrictModel):
    """A small nonnegative integer for an exact arithmetic function."""

    n: StrictInt = Field(ge=0, le=_MAX_N_SMALL)


# ---------------------------------------------------------------------------
# Request models — bounded non-negative / positive integers
# ---------------------------------------------------------------------------


class NonnegativeIntegerRequest(StrictModel):
    """One bounded non-negative integer (0 <= n <= 10 000)."""

    n: StrictInt = Field(ge=0, le=_MAX_N_SMALL)


class PositiveIntegerRequest(StrictModel):
    """One bounded positive integer (1 <= n <= 10 000)."""

    n: StrictInt = Field(ge=1, le=_MAX_N_SMALL)


class PrimorialRequest(StrictModel):
    """One bounded positive integer whose primorial fits the result contract.

    ``primorial(n)`` grows like ``exp(n log n)``: the product of the first
    ``n`` primes carries ``n(log n + log log n) / ln 10`` digits.  The
    shared arithmetic-function bound admits values whose primorial would
    exceed the declared ``_MAX_PRIMORIAL_DIGITS``-digit result, so this
    request derives its own conservative ceiling from the digit bound.
    """

    n: StrictInt = Field(ge=1, le=_MAX_PRIMORIAL_N)


class PreviousPrimeRequest(StrictModel):
    """One bounded integer n >= 3 for previous-prime queries."""

    n: StrictInt = Field(ge=3, le=_MAX_N_SMALL)


class FloorSquareRootRequest(StrictModel):
    n: StrictInt = Field(ge=0, le=1_000_000_000_000)


class FloorSquareRootResult(StrictModel):
    """The exact floor of the nonnegative integer square root."""

    root: StrictInt = Field(ge=0, le=1_000_000)


class LegendreSymbolRequest(StrictModel):
    """Arguments for the Legendre symbol with a bounded odd prime denominator."""

    a: StrictInt = Field(ge=-(2**53 - 1), le=2**53 - 1)
    prime: StrictInt = Field(ge=3, le=10_000_000)

    @model_validator(mode="after")
    def require_prime_denominator(self) -> Self:
        from sympy import isprime

        if not isprime(self.prime):
            raise _validation_error(
                "legendre_denominator_must_be_prime",
                "Legendre denominator must be prime",
            )
        return self


class LegendreSymbolResult(StrictModel):
    a: StrictInt
    prime: StrictInt = Field(ge=3, le=10_000_000)
    symbol: Literal[-1, 0, 1]


class FactorialValuationRequest(StrictModel):
    """Arguments for the largest exponent ``e`` such that ``base**e`` divides ``n!``."""

    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)


class FactorialValuationResult(StrictModel):
    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)
    valuation: StrictInt = Field(ge=0)


class FriableCountRequest(StrictModel):
    """One exact bounded count of positive ``y``-friable integers through ``x``."""

    x: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive source bound. Easy boundary cases may "
            "use up to 256 decimal digits; other cases must fit an exact counting "
            "regime selected from the source-sensitive work bounds."
        ),
        examples=["100"],
    )
    y: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive prime-factor cutoff. Values zero and "
            "one use the convention that only 1 is friable when x is positive."
        ),
        examples=["5"],
    )

    @model_validator(mode="after")
    def require_admitted_exact_count(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        _plan_friable_count(
            parse_canonical_integer(self.x),
            parse_canonical_integer(self.y),
        )
        return self


class FriableCountResult(StrictModel):
    """An exact friable count bound to its complete source pair."""

    x: BoundedInteger
    y: BoundedInteger
    count: BoundedInteger

    @model_validator(mode="after")
    def bind_exact_count_to_sources(self) -> Self:
        from jacobian.canonical import parse_canonical_integer
        from jacobian.math.number_theory._friable_operations import count_friable

        x = parse_canonical_integer(self.x)
        y = parse_canonical_integer(self.y)
        count = parse_canonical_integer(self.count)
        if count < 0:
            raise _validation_error(
                "friable_count_must_be_nonnegative", "friable count must be nonnegative"
            )
        if count != count_friable(x, y):
            raise _validation_error(
                "friable_count_does_not_match_the_retained_sources",
                "friable count does not match the retained sources",
            )
        return self


# ---------------------------------------------------------------------------
# Request models — modular arithmetic
# ---------------------------------------------------------------------------


class ModularValueRequest(StrictModel):
    """One canonical integer and a bounded modulus (2 <= modulus <= 1 000 000)."""

    value: BoundedInteger
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)


class ModularUnitRequest(StrictModel):
    """One canonical integer and a bounded modulus where the value must be a unit."""

    value: BoundedInteger
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)

    @model_validator(mode="after")
    def require_coprime(self) -> Self:
        from math import gcd

        if gcd(int(self.value), self.modulus) != 1:
            raise _validation_error(
                "value_must_be_coprime_to_the_modulus",
                "value must be coprime to the modulus",
            )
        return self


class ModulusRequest(StrictModel):
    """A single bounded modulus (2 <= modulus <= 1 000 000)."""

    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)


class ChineseRemainderRequest(StrictModel):
    """A finite system of integer congruences with parallel residues and moduli."""

    residues: tuple[int, ...] = Field(min_length=1, max_length=_MAX_CRT_SIZE)
    moduli: tuple[int, ...] = Field(min_length=1, max_length=_MAX_CRT_SIZE)

    @model_validator(mode="after")
    def require_parallel_positive_moduli(self) -> Self:
        if len(self.residues) != len(self.moduli):
            raise _validation_error(
                "residues_and_moduli_must_have_equal_length",
                "residues and moduli must have equal length",
            )
        if any(modulus < 2 or modulus > _MAX_MODULUS for modulus in self.moduli):
            raise _validation_error(
                "every_modulus_must_be_between_2_and_1_000_000",
                "every modulus must be between 2 and 1,000,000",
            )
        # The result carries the system's combined modulus as one exact
        # ``BoundedInteger``, so admission derives its input envelope from
        # that declared output budget: reject any compatible system whose
        # LCM exceeds the result width, however small each modulus is.
        from math import gcd

        combined = 1
        for modulus in self.moduli:
            combined = combined // gcd(combined, modulus) * modulus
            if combined > _MAX_CRT_COMBINED_MODULUS:
                raise _validation_error(
                    "the_system_s_combined_modulus_must_have_at",
                    "the system's combined modulus must have at most "
                    f"{_MAX_INTEGER_LENGTH} digits; split the congruence "
                    "system into narrower subsystems",
                )
        if any(
            residue < 0 or residue >= modulus
            for residue, modulus in zip(self.residues, self.moduli, strict=True)
        ):
            raise _validation_error(
                "every_residue_must_be_canonical_for_its_modulus",
                "every residue must be canonical for its modulus",
            )
        # Check pairwise consistency: residues must agree modulo gcd(moduli).
        for i in range(len(self.moduli)):
            for j in range(i + 1, len(self.moduli)):
                g = gcd(self.moduli[i], self.moduli[j])
                if (self.residues[i] - self.residues[j]) % g != 0:
                    raise _validation_error(
                        "congruence_system_is_inconsistent",
                        "congruence system is inconsistent",
                    )
        return self


class JacobiSymbolRequest(StrictModel):
    """Arguments for the Jacobi symbol (a / n), with odd positive n."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=_MAX_MODULUS)

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise _validation_error(
                "jacobi_symbol_denominator_must_be_odd",
                "Jacobi symbol denominator must be odd",
            )
        return self


class DiscreteLogarithmRequest(StrictModel):
    """A bounded modular discrete-logarithm problem."""

    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)

    @model_validator(mode="after")
    def require_canonical_residues(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise _validation_error(
                "base_and_target_must_be_less_than_the_modulus",
                "base and target must be less than the modulus",
            )
        return self


_MAX_PRIMORIAL_DIGITS = 3_400
PrimorialInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=_MAX_PRIMORIAL_DIGITS,
        strict=True,
    ),
]


class PrimorialResult(StrictModel):
    """The primorial (product of the first n primes)."""

    value: PrimorialInteger


class PrimePower(StrictModel):
    """One prime base and its exponent in a prime factorization."""

    prime: BoundedInteger
    power: int = Field(ge=1, le=_MAX_N_SMALL)


class ResidualPerfectPower(StrictModel):
    """An exact decomposition of the stripped residual as base**exponent."""

    base: PowerfulInteger
    exponent: StrictInt = Field(ge=2, le=MAX_POWERFUL_EXPONENT)


class PowerfulNumberResult(StrictModel):
    """A source-bound, replayable exact powerful-number decision."""

    value: PowerfulInteger
    conclusion: Literal[
        "POWERFUL",
        "EXPONENT_ONE",
        "ROUGH_NOT_PERFECT_POWER",
    ]
    is_powerful: StrictBool
    cutoff: StrictInt = Field(
        ge=1,
        le=MAX_POWERFUL_CUTOFF,
        description="The canonical cutoff B=ceil(value^(1/5)); B^5 >= value.",
    )
    checked_through: StrictInt = Field(
        ge=1,
        le=MAX_POWERFUL_CUTOFF,
        description=(
            "All primes at most this bound were tested. It equals cutoff unless "
            "an exponent-one prime ended the decision early."
        ),
    )
    stripped_factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=MAX_POWERFUL_FACTOR_ENTRIES,
    )
    residual: PowerfulInteger = Field(
        description=(
            "The positive cofactor after the reported prime powers are removed."
        )
    )
    residual_perfect_power: ResidualPerfectPower | None = None

    @model_validator(mode="after")
    def bind_decision_to_source_by_exact_replay(self) -> Self:
        from jacobian.canonical import parse_canonical_integer
        from jacobian.math.number_theory._powerful_kernels import (
            decide_powerful_data,
        )

        expected = decide_powerful_data(parse_canonical_integer(self.value))
        factors = tuple(
            (parse_canonical_integer(factor.prime), factor.power)
            for factor in self.stripped_factors
        )
        perfect_power = (
            None
            if self.residual_perfect_power is None
            else (
                parse_canonical_integer(self.residual_perfect_power.base),
                self.residual_perfect_power.exponent,
            )
        )
        if (
            self.conclusion != expected.conclusion
            or self.is_powerful != (expected.conclusion == "POWERFUL")
            or self.cutoff != expected.cutoff
            or self.checked_through != expected.checked_through
            or factors != expected.stripped_factors
            or parse_canonical_integer(self.residual) != expected.residual
            or perfect_power != expected.perfect_power
        ):
            raise _validation_error(
                "powerful_number_conclusion_or_certificate_does_not_match_exact_replay",
                "powerful-number conclusion or certificate does not match exact replay",
            )
        return self


class BooleanResult(StrictModel):
    """Truth value of a number-theory predicate."""

    holds: bool


class QuadraticResiduesResult(StrictModel):
    """All quadratic residues modulo one modulus."""

    residues: tuple[BoundedInteger, ...]


class ChineseRemainderResult(StrictModel):
    """The least non-negative solution and modulus of a compatible CRT system."""

    residue: BoundedInteger
    modulus: BoundedInteger


class JacobiSymbolResult(StrictModel):
    """The exact Jacobi symbol, bound to its normalized arguments."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=_MAX_MODULUS)
    jacobi: Literal[-1, 0, 1]

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise _validation_error(
                "jacobi_symbol_denominator_must_be_odd",
                "Jacobi symbol denominator must be odd",
            )
        return self


class DiscreteLogarithmResult(StrictModel):
    """The exact result of one bounded discrete-logarithm computation."""

    status: Literal["SOLVED", "UNSOLVABLE"]
    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    discrete_log: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_conclusion(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise _validation_error(
                "base_and_target_must_be_less_than_the_modulus",
                "base and target must be less than the modulus",
            )
        if self.status == "SOLVED":
            if self.discrete_log is None:
                raise _validation_error(
                    "solved_discrete_logarithm_requires_an_exponent",
                    "solved discrete logarithm requires an exponent",
                )
            if pow(self.base, self.discrete_log, self.modulus) != self.target:
                raise _validation_error(
                    "discrete_logarithm_does_not_reproduce_the_target",
                    "discrete logarithm does not reproduce the target",
                )
        elif self.discrete_log is not None:
            raise _validation_error(
                "unsolvable_discrete_logarithm_cannot_carry_an_exponent",
                "unsolvable discrete logarithm cannot carry an exponent",
            )
        return self


# ---------------------------------------------------------------------------
# Pratt certificate and certified factorization models
# ---------------------------------------------------------------------------


def _verify_pratt_identities(p: int, witness: int, sub_primes: tuple[int, ...]) -> None:
    """Verify the Pratt identities for one certificate node.

    Checks ``witness^(p-1) ≡ 1 (mod p)``, ``witness^((p-1)/q) ≢ 1 (mod p)``
    for each prime factor ``q`` of ``p - 1``, and that ``sub_primes``
    exactly covers the distinct prime factors of ``p - 1``.

    Completeness is verified by repeatedly dividing ``p - 1`` by the
    recursively certified ``sub_primes`` and requiring residual ``1``,
    without invoking a factoring backend.  This keeps validation bounded
    and makes the Pratt certificate independently replayable.
    """

    if pow(witness, p - 1, p) != 1:
        raise _validation_error(
            "pratt_witness_fails_a_p_1_1_mod_p",
            "Pratt witness fails a^(p-1) ≡ 1 (mod p)",
        )
    for q in sub_primes:
        if (p - 1) % q != 0:
            raise _validation_error(
                "sub_certificate_prime_must_divide_p_1",
                "sub-certificate prime must divide p-1",
            )
        if pow(witness, (p - 1) // q, p) == 1:
            raise _validation_error(
                "pratt_witness_fails_a_p_1_q_1_mod_p",
                "Pratt witness fails a^((p-1)/q) ≢ 1 (mod p)",
            )
    # Verify completeness without factoring: divide out each certified
    # prime factor and require that the residual becomes 1.  Duplicate
    # primes are already rejected by the caller, and each q is a
    # recursively certified prime (validated before this node).
    residual = p - 1
    for q in sub_primes:
        while residual % q == 0:
            residual //= q
    if residual != 1:
        raise _validation_error(
            "sub_certificates_must_exactly_cover_the_distinct_prime_factors_of_p_1",
            "sub-certificates must exactly cover the distinct prime factors of p-1",
        )


class PrattCertificateNode(StrictModel):
    """One node in a Pratt primality certificate tree.

    A Pratt certificate proves that ``prime`` is prime by exhibiting a witness
    ``a`` such that ``a^(prime-1) ≡ 1 (mod prime)`` and ``a^((prime-1)/q) ≢ 1
    (mod prime)`` for every prime factor ``q`` of ``prime - 1``.  Each such
    ``q`` is itself certified by a recursive Pratt certificate.

    The base case is ``prime == 2``: it has no prime factors of ``prime - 1``
    and thus no sub-certificates and no witness.
    """

    prime: BoundedInteger
    witness: BoundedInteger | None = None
    sub_certificates: tuple[PrattCertificateNode, ...] = Field(
        default_factory=tuple,
        min_length=0,
        max_length=256,
    )

    @model_validator(mode="after")
    def require_valid_certificate(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        p = parse_canonical_integer(self.prime)
        if p < 2:
            raise _validation_error(
                "certificate_prime_must_be_at_least_2",
                "certificate prime must be at least 2",
            )
        if p == 2:
            if self.witness is not None:
                raise _validation_error(
                    "base_case_prime_2_has_no_witness",
                    "base case prime 2 has no witness",
                )
            if self.sub_certificates:
                raise _validation_error(
                    "base_case_prime_2_has_no_sub_certificates",
                    "base case prime 2 has no sub-certificates",
                )
            return self
        if self.witness is None:
            raise _validation_error(
                "non_base_case_certificate_requires_a_witness",
                "non-base-case certificate requires a witness",
            )
        sub_primes_str = [item.prime for item in self.sub_certificates]
        if len(set(sub_primes_str)) != len(self.sub_certificates):
            raise _validation_error(
                "sub_certificate_primes_must_be_unique",
                "sub-certificate primes must be unique",
            )
        w = parse_canonical_integer(self.witness)
        if w < 2 or w >= p:
            raise _validation_error(
                "witness_must_be_between_2_and_p_1", "witness must be between 2 and p-1"
            )
        sub_primes = tuple(
            parse_canonical_integer(sub.prime) for sub in self.sub_certificates
        )
        _verify_pratt_identities(p, w, sub_primes)
        return self


class CertifiedFactorizationRequest(StrictModel):
    """One positive integer for subexponential certified factorization.

    The integer is bounded to 30 digits (~100 bits) so that SymPy's
    ``factorint`` (Pollard rho, p-1, ECM) and recursive Pratt ``p - 1``
    factorization complete within a bounded synchronous budget.  See
    ``_MAX_CERTIFIED_FACTORIZATION_LENGTH`` for the work-bound rationale.
    """

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_composite_domain(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        if parse_canonical_integer(self.value) < 2:
            raise _validation_error(
                "certified_factorization_requires_an_integer_at_least_2",
                "certified factorization requires an integer at least 2",
            )
        return self


class CertifiedFactor(StrictModel):
    """One certified prime factor with its Pratt primality certificate."""

    prime: BoundedInteger
    exponent: StrictInt = Field(ge=1, le=4096)
    certificate: PrattCertificateNode


class CertifiedFactorizationResult(StrictModel):
    """The complete certified prime-power factorization of one integer."""

    status: Literal["COMPLETE"]
    value: CertifiedFactorizationInteger
    factors: tuple[CertifiedFactor, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def bind_decomposition(self) -> Self:
        import math as _math

        from jacobian.canonical import parse_canonical_integer

        product = _math.prod(
            parse_canonical_integer(item.prime) ** item.exponent
            for item in self.factors
        )
        if product != parse_canonical_integer(self.value):
            raise _validation_error(
                "factor_components_must_multiply_to_the_requested_integer",
                "factor components must multiply to the requested integer",
            )
        primes = [parse_canonical_integer(item.prime) for item in self.factors]
        if primes != sorted(primes):
            raise _validation_error(
                "factor_primes_must_be_ascending", "factor primes must be ascending"
            )
        if len(set(primes)) != len(primes):
            raise _validation_error(
                "factor_primes_must_be_unique", "factor primes must be unique"
            )
        for item in self.factors:
            cert_prime = parse_canonical_integer(item.certificate.prime)
            factor_prime = parse_canonical_integer(item.prime)
            if cert_prime != factor_prime:
                raise _validation_error(
                    "factor_certificate_prime_must_equal_the_factor_prime",
                    "factor certificate prime must equal the factor prime",
                )
        return self


class PrimalityCertificateRequest(StrictModel):
    """One positive integer to be certified as prime via a Pratt certificate."""

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_candidate_domain(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        if parse_canonical_integer(self.value) < 2:
            raise _validation_error(
                "primality_certificate_requires_an_integer_at_least_2",
                "primality certificate requires an integer at least 2",
            )
        return self


class PrimalityCertificateResult(StrictModel):
    """A Pratt primality certificate for one declared prime."""

    status: Literal["CERTIFIED", "COMPOSITE"]
    value: CertifiedFactorizationInteger
    certificate: PrattCertificateNode | None = None

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        if self.status == "CERTIFIED" and self.certificate is None:
            raise _validation_error(
                "certified_status_requires_a_certificate",
                "CERTIFIED status requires a certificate",
            )
        if self.status == "COMPOSITE" and self.certificate is not None:
            raise _validation_error(
                "composite_status_must_not_carry_a_certificate",
                "COMPOSITE status must not carry a certificate",
            )
        value_int = parse_canonical_integer(self.value)
        if self.status == "COMPOSITE":
            from sympy import isprime

            if isprime(value_int):
                raise _validation_error(
                    "composite_status_requires_a_composite_value",
                    "COMPOSITE status requires a composite value",
                )
        if self.status == "CERTIFIED":
            assert self.certificate is not None
            cert_prime = parse_canonical_integer(self.certificate.prime)
            if cert_prime != value_int:
                raise _validation_error(
                    "certificate_prime_must_match_the_candidate_value",
                    "certificate prime must match the candidate value",
                )
            from sympy import isprime

            if not isprime(value_int):
                raise _validation_error(
                    "certified_status_requires_a_prime_value",
                    "CERTIFIED status requires a prime value",
                )
        return self


PrattCertificateNode.model_rebuild()
