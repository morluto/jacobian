"""Named Pydantic wire contracts for exact integer number-theory operations.

These contracts cover gcd/lcm, Bezout coefficients, divisors, prime
factorization, p-adic valuation, multiplicative arithmetic functions,
primality, modular arithmetic, and integer predicates (coprimality,
divisibility, perfect/abundant/deficient, square, squarefree).  They are
owned by the number-theory domain and intentionally exclude arithmetic-owned
operations (absolute value, sign, decimal digit sum/count, base expansion,
integer nth root).
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import product
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian.contracts.base import ContractModel

# ---------------------------------------------------------------------------
# Shared bounds for the current bounded integer-domain contracts.
# ---------------------------------------------------------------------------

_MAX_INTEGER_LENGTH = 256
_MAX_FACTORIZATION_LENGTH = 12
# These small bounds deliberately keep arithmetic functions that may factor
# their input (totient, Möbius, divisor sigma, square-free predicates, and
# multiplicative order) safe for in-process SymPy execution.
_MAX_N_SMALL = 1_000
_MAX_MODULUS = 10_000
_MAX_CRT_SIZE = 64
_MAX_DIVISORS = 4_096
_MAX_FACTOR_ENTRIES = 256
_MAX_RESIDUE_VARIABLES = 6
_MAX_RESIDUE_DOMAIN_SIZE = 32
_MAX_RESIDUE_TERMS = 64
_MAX_RESIDUE_EXPONENT = 32
_MAX_RESIDUE_ASSIGNMENTS = 4_096
_MAX_POLYNOMIAL_RESIDUE_MODULUS = 1_000_000

BoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=_MAX_INTEGER_LENGTH,
        strict=True,
    ),
]
FactorizationInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=_MAX_FACTORIZATION_LENGTH,
        strict=True,
    ),
]
ResidueVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        max_length=32,
        strict=True,
    ),
]
ResidueDomain = Annotated[
    tuple[StrictInt, ...],
    Field(min_length=1, max_length=_MAX_RESIDUE_DOMAIN_SIZE),
]
ResidueAssignment = Annotated[
    tuple[StrictInt, ...],
    Field(min_length=1, max_length=_MAX_RESIDUE_VARIABLES),
]
CanonicalResidue = Annotated[
    StrictInt,
    Field(ge=0, lt=_MAX_POLYNOMIAL_RESIDUE_MODULUS),
]


# ---------------------------------------------------------------------------
# Request models — canonical integers (arbitrary precision, bounded string)
# ---------------------------------------------------------------------------


class IntegerValueRequest(ContractModel):
    """One canonical integer supplied to a unary number-theory operation."""

    value: BoundedInteger


class FactorizationRequest(ContractModel):
    """One small integer for direct exact factorization in the server process."""

    value: FactorizationInteger


class NonzeroFactorizationRequest(FactorizationRequest):
    """One nonzero integer with a finite divisor and prime-factorization set."""

    @model_validator(mode="after")
    def require_nonzero_value(self) -> Self:
        if int(self.value) == 0:
            raise ValueError("zero has no finite factorization or divisor enumeration")
        return self


class PowerfulNumberRequest(FactorizationRequest):
    """One positive integer for an exact powerful-number decision."""

    @model_validator(mode="after")
    def require_positive_value(self) -> Self:
        if int(self.value) < 1:
            raise ValueError("powerful-number input must be positive")
        return self


class ArithmeticFunctionRequest(ContractModel):
    """A small nonnegative integer for an exact arithmetic function."""

    n: StrictInt = Field(ge=0, le=_MAX_N_SMALL)


class IntegerPairRequest(ContractModel):
    """Two canonical integers supplied to a symmetric binary operation."""

    left: BoundedInteger
    right: BoundedInteger


class DivisibilityRequest(ContractModel):
    """A divisor and dividend supplied to a divisibility predicate."""

    divisor: BoundedInteger
    dividend: BoundedInteger


class ValuationRequest(ContractModel):
    """One integer and a prime base supplied to a p-adic valuation."""

    value: BoundedInteger
    prime: BoundedInteger


# ---------------------------------------------------------------------------
# Request models — bounded non-negative / positive integers
# ---------------------------------------------------------------------------


class NonnegativeIntegerRequest(ContractModel):
    """One bounded non-negative integer (0 <= n <= 1 000)."""

    n: StrictInt = Field(ge=0, le=_MAX_N_SMALL)


class PositiveIntegerRequest(ContractModel):
    """One bounded positive integer (1 <= n <= 1 000)."""

    n: StrictInt = Field(ge=1, le=_MAX_N_SMALL)


class FloorSquareRootRequest(ContractModel):
    n: StrictInt = Field(ge=0, le=1_000_000_000_000)


class FloorSquareRootResult(ContractModel):
    """The exact floor of the nonnegative integer square root."""

    root: StrictInt = Field(ge=0, le=1_000_000)


class LegendreSymbolRequest(ContractModel):
    """Arguments for the Legendre symbol with a bounded odd prime denominator."""

    a: StrictInt = Field(ge=-(2**53 - 1), le=2**53 - 1)
    prime: StrictInt = Field(ge=3, le=10_000_000)

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.prime % 2 == 0:
            raise ValueError("Legendre denominator must be odd")
        return self


class LegendreSymbolResult(ContractModel):
    a: StrictInt
    prime: StrictInt = Field(ge=3, le=10_000_000)
    symbol: Literal[-1, 0, 1]


class FactorialValuationRequest(ContractModel):
    """Arguments for the largest exponent ``e`` such that ``base**e`` divides ``n!``."""

    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)


class FactorialValuationResult(ContractModel):
    n: StrictInt = Field(ge=0, le=100_000)
    base: StrictInt = Field(ge=2, le=1_000_000)
    valuation: StrictInt = Field(ge=0)


# ---------------------------------------------------------------------------
# Request models — modular arithmetic
# ---------------------------------------------------------------------------


class ModularValueRequest(ContractModel):
    """One canonical integer and a bounded modulus (2 <= modulus <= 10 000)."""

    value: BoundedInteger
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)


class ModulusRequest(ContractModel):
    """A single bounded modulus (2 <= modulus <= 10 000)."""

    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)


class ModularPolynomialVariable(ContractModel):
    """One named variable and its canonical finite residue domain."""

    name: ResidueVariableName
    residues: ResidueDomain

    @model_validator(mode="after")
    def require_canonical_domain(self) -> Self:
        if any(residue < 0 for residue in self.residues):
            raise ValueError("variable residues must be nonnegative")
        if self.residues != tuple(sorted(set(self.residues))):
            raise ValueError("variable residues must be strictly increasing")
        return self


class ModularPolynomialTerm(ContractModel):
    """One nonzero sparse integer-polynomial term in canonical exponent order."""

    coefficient: BoundedInteger
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )

    @model_validator(mode="after")
    def require_nonnegative_exponents(self) -> Self:
        if any(
            exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
            for exponent in self.exponents
        ):
            raise ValueError(
                f"term exponents must be between 0 and {_MAX_RESIDUE_EXPONENT}"
            )
        return self


class ModularPolynomialResidueImageRequest(ContractModel):
    """A bounded sparse polynomial over declared finite residue domains."""

    modulus: StrictInt = Field(ge=2, le=_MAX_POLYNOMIAL_RESIDUE_MODULUS)
    variables: tuple[ModularPolynomialVariable, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    terms: tuple[ModularPolynomialTerm, ...] = Field(
        min_length=0,
        max_length=_MAX_RESIDUE_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_bounded_polynomial(self) -> Self:
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("polynomial variable names must be unique")
        if any(
            residue >= self.modulus
            for variable in self.variables
            for residue in variable.residues
        ):
            raise ValueError("every variable residue must be less than the modulus")
        assignment_count = math.prod(
            len(variable.residues) for variable in self.variables
        )
        if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
            raise ValueError(
                "declared residue domains exceed the 4,096-assignment bound"
            )
        if any(len(term.exponents) != len(self.variables) for term in self.terms):
            raise ValueError("every term exponent vector must match the variable count")
        exponent_vectors = [term.exponents for term in self.terms]
        if exponent_vectors != sorted(set(exponent_vectors)):
            raise ValueError(
                "term exponent vectors must be unique and lexicographically increasing"
            )
        if any(int(term.coefficient) % self.modulus == 0 for term in self.terms):
            raise ValueError(
                "sparse polynomial terms must have nonzero coefficient modulo m"
            )
        return self


class ChineseRemainderRequest(ContractModel):
    """A finite system of integer congruences with parallel residues and moduli."""

    residues: tuple[int, ...] = Field(min_length=1, max_length=_MAX_CRT_SIZE)
    moduli: tuple[int, ...] = Field(min_length=1, max_length=_MAX_CRT_SIZE)

    @model_validator(mode="after")
    def require_parallel_positive_moduli(self) -> Self:
        if len(self.residues) != len(self.moduli):
            raise ValueError("residues and moduli must have equal length")
        if any(modulus < 2 or modulus > _MAX_MODULUS for modulus in self.moduli):
            raise ValueError("every modulus must be between 2 and 10,000")
        if any(
            residue < 0 or residue >= modulus
            for residue, modulus in zip(self.residues, self.moduli, strict=True)
        ):
            raise ValueError("every residue must be canonical for its modulus")
        return self


class JacobiSymbolRequest(ContractModel):
    """Arguments for the Jacobi symbol (a / n), with odd positive n."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=_MAX_MODULUS)

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise ValueError("Jacobi symbol denominator must be odd")
        return self


class DiscreteLogarithmRequest(ContractModel):
    """A bounded modular discrete-logarithm problem."""

    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)

    @model_validator(mode="after")
    def require_canonical_residues(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise ValueError("base and target must be less than the modulus")
        return self


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class IntegerValueResult(ContractModel):
    """One exact integer value produced by a number-theory operation."""

    value: BoundedInteger


class ExtendedGcdResult(ContractModel):
    """A gcd together with exact Bezout coefficients."""

    gcd: BoundedInteger
    left_coefficient: BoundedInteger
    right_coefficient: BoundedInteger


class DivisorListResult(ContractModel):
    """An ordered list of positive divisors of one nonzero integer.

    The list may be empty: ``proper_divisors(±1)`` has no positive proper
    divisors.  Zero remains not-applicable (handled at the operation layer).
    """

    divisors: tuple[BoundedInteger, ...] = Field(
        min_length=0,
        max_length=_MAX_DIVISORS,
    )

    @model_validator(mode="after")
    def require_positive_ascending_unique(self) -> Self:
        values = [int(divisor) for divisor in self.divisors]
        if any(value < 1 for value in values):
            raise ValueError("divisors must be positive")
        if values != sorted(values):
            raise ValueError("divisors must be ascending")
        if len(set(values)) != len(values):
            raise ValueError("divisors must be unique")
        return self


class PrimePower(ContractModel):
    """One prime base and its exponent in a prime factorization."""

    prime: BoundedInteger
    power: int = Field(ge=1, le=_MAX_N_SMALL)


class PrimeFactorizationResult(ContractModel):
    """The complete prime-power factorization of one nonzero integer.

    The factor list may be empty: ``±1`` has no prime factors.  Zero remains
    not-applicable (handled at the operation layer).
    """

    factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=_MAX_FACTOR_ENTRIES,
    )

    @model_validator(mode="after")
    def require_unique_primes(self) -> Self:
        primes = [factor.prime for factor in self.factors]
        if len(set(primes)) != len(primes):
            raise ValueError("prime factors must be unique")
        return self


class PowerfulNumberResult(ContractModel):
    """A powerful-number decision with its complete factor witness."""

    semantics_version: Literal["powerful-number.prime-exponents-at-least-two.v1"]
    is_powerful: StrictBool
    factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=_MAX_FACTOR_ENTRIES,
    )
    violating_primes: tuple[BoundedInteger, ...] = Field(
        min_length=0,
        max_length=_MAX_FACTOR_ENTRIES,
    )

    @model_validator(mode="after")
    def bind_decision_to_canonical_factor_witness(self) -> Self:
        primes = [int(factor.prime) for factor in self.factors]
        if any(prime < 2 for prime in primes):
            raise ValueError("factor bases must be greater than one")
        if primes != sorted(set(primes)):
            raise ValueError("factor bases must be strictly increasing")
        expected_violations = tuple(
            factor.prime for factor in self.factors if factor.power < 2
        )
        if self.violating_primes != expected_violations:
            raise ValueError(
                "violating primes must be exactly the factors with exponent below two"
            )
        if self.is_powerful != (not expected_violations):
            raise ValueError("powerful decision does not match the factor exponents")
        return self


class BooleanResult(ContractModel):
    """Truth value of a number-theory predicate."""

    holds: bool


class QuadraticResiduesResult(ContractModel):
    """All quadratic residues modulo one modulus."""

    residues: tuple[BoundedInteger, ...]


class NormalizedModularPolynomialTerm(ContractModel):
    """One sparse term with its coefficient reduced to the canonical residue."""

    coefficient: StrictInt = Field(ge=1, lt=_MAX_POLYNOMIAL_RESIDUE_MODULUS)
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )


class ModularPolynomialResidueCount(ContractModel):
    """Multiplicity of one reachable residue in the declared assignment table."""

    residue: CanonicalResidue
    count: StrictInt = Field(ge=1, le=_MAX_RESIDUE_ASSIGNMENTS)


class ModularPolynomialResidueWitness(ContractModel):
    """The first lexicographic assignment reaching one residue."""

    residue: CanonicalResidue
    assignment: ResidueAssignment


class ModularPolynomialResidueTableRow(ContractModel):
    """One exact assignment-to-residue evaluation."""

    assignment: ResidueAssignment
    residue: CanonicalResidue


class ModularPolynomialResidueImageResult(ContractModel):
    """Exact residue-image summary with an optional complete assignment table."""

    semantics_version: Literal["modular-polynomial-residue-image.v1"]
    modulus: StrictInt = Field(ge=2, le=_MAX_POLYNOMIAL_RESIDUE_MODULUS)
    variable_order: tuple[ResidueVariableName, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    domains: tuple[ResidueDomain, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    normalized_terms: tuple[NormalizedModularPolynomialTerm, ...] = Field(
        min_length=0,
        max_length=_MAX_RESIDUE_TERMS,
    )
    enumeration_scope: Literal["COMPLETE_DECLARED_CARTESIAN_PRODUCT"]
    total_assignments: StrictInt = Field(ge=1, le=_MAX_RESIDUE_ASSIGNMENTS)
    image: tuple[CanonicalResidue, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    residue_counts: tuple[ModularPolynomialResidueCount, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    witnesses: tuple[ModularPolynomialResidueWitness, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    table: tuple[ModularPolynomialResidueTableRow, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )

    @model_validator(mode="after")
    def bind_complete_residue_image(self) -> Self:
        assignments = _validate_residue_image_shape(self)
        residues = _validate_residue_image_table(self, assignments)
        _validate_residue_image_summaries(self, assignments, residues)
        return self


def _evaluate_normalized_modular_polynomial(
    terms: tuple[NormalizedModularPolynomialTerm, ...],
    assignment: tuple[int, ...],
    modulus: int,
) -> int:
    value = 0
    for term in terms:
        monomial = term.coefficient
        for coordinate, exponent in zip(
            assignment,
            term.exponents,
            strict=True,
        ):
            monomial = monomial * pow(coordinate, exponent, modulus) % modulus
        value = (value + monomial) % modulus
    return value


def _validate_residue_image_shape(
    result: ModularPolynomialResidueImageResult,
) -> tuple[tuple[int, ...], ...]:
    if len(set(result.variable_order)) != len(result.variable_order):
        raise ValueError("result variable names must be unique")
    if len(result.domains) != len(result.variable_order):
        raise ValueError("result domains must match the variable count")
    if any(
        domain != tuple(sorted(set(domain)))
        or any(residue < 0 or residue >= result.modulus for residue in domain)
        for domain in result.domains
    ):
        raise ValueError("result domains must contain canonical increasing residues")
    if any(
        len(term.exponents) != len(result.variable_order)
        or term.coefficient >= result.modulus
        or any(
            exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
            for exponent in term.exponents
        )
        for term in result.normalized_terms
    ):
        raise ValueError("normalized terms do not match the result scope")
    exponent_vectors = [term.exponents for term in result.normalized_terms]
    if exponent_vectors != sorted(set(exponent_vectors)):
        raise ValueError("normalized term exponents must be canonical")
    assignment_count = math.prod(len(domain) for domain in result.domains)
    if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
        raise ValueError("result domains exceed the 4,096-assignment bound")
    if result.total_assignments != assignment_count:
        raise ValueError("total assignments do not match the declared domains")
    if result.table is not None and len(result.table) != assignment_count:
        raise ValueError("complete table length does not match the declared domains")
    assignments = tuple(product(*result.domains))
    return assignments


def _validate_residue_image_table(
    result: ModularPolynomialResidueImageResult,
    assignments: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    expected_residues = tuple(
        _evaluate_normalized_modular_polynomial(
            result.normalized_terms,
            assignment,
            result.modulus,
        )
        for assignment in assignments
    )
    if result.table is not None:
        if tuple(row.assignment for row in result.table) != assignments:
            raise ValueError(
                "complete table must enumerate the declared Cartesian product in order"
            )
        if tuple(row.residue for row in result.table) != expected_residues:
            raise ValueError(
                "complete table contains an incorrect polynomial evaluation"
            )
    return expected_residues


def _validate_residue_image_summaries(
    result: ModularPolynomialResidueImageResult,
    assignments: tuple[tuple[int, ...], ...],
    residues: tuple[int, ...],
) -> None:
    image = tuple(sorted(set(residues)))
    if result.image != image:
        raise ValueError("residue image does not match the complete table")
    counts = Counter(residues)
    expected_counts = tuple(
        ModularPolynomialResidueCount(residue=residue, count=counts[residue])
        for residue in image
    )
    if result.residue_counts != expected_counts:
        raise ValueError("residue counts do not match the complete table")
    first_assignments: dict[int, tuple[int, ...]] = {}
    for assignment, residue in zip(assignments, residues, strict=True):
        first_assignments.setdefault(residue, assignment)
    expected_witnesses = tuple(
        ModularPolynomialResidueWitness(
            residue=residue,
            assignment=first_assignments[residue],
        )
        for residue in image
    )
    if result.witnesses != expected_witnesses:
        raise ValueError("residue witnesses must be the first table assignments")


class ChineseRemainderResult(ContractModel):
    """The least non-negative solution and modulus of a compatible CRT system."""

    residue: BoundedInteger
    modulus: BoundedInteger


class JacobiSymbolResult(ContractModel):
    """The exact Jacobi symbol, bound to its normalized arguments."""

    a: BoundedInteger
    n: StrictInt = Field(ge=3, le=_MAX_MODULUS)
    jacobi: Literal[-1, 0, 1]

    @model_validator(mode="after")
    def require_odd_denominator(self) -> Self:
        if self.n % 2 == 0:
            raise ValueError("Jacobi symbol denominator must be odd")
        return self


class DiscreteLogarithmResult(ContractModel):
    """The exact result of one bounded discrete-logarithm computation."""

    status: Literal["SOLVED", "UNSOLVABLE"]
    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    discrete_log: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_conclusion(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise ValueError("base and target must be less than the modulus")
        if self.status == "SOLVED":
            if self.discrete_log is None:
                raise ValueError("solved discrete logarithm requires an exponent")
            if pow(self.base, self.discrete_log, self.modulus) != self.target:
                raise ValueError("discrete logarithm does not reproduce the target")
        elif self.discrete_log is not None:
            raise ValueError("unsolvable discrete logarithm cannot carry an exponent")
        return self
