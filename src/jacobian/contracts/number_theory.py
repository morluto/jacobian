"""Named Pydantic wire contracts for exact integer number-theory capabilities.

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

from jacobian.contracts.results import ContractModel

# ---------------------------------------------------------------------------
# Shared bounds for the current bounded integer-domain contracts.
# ---------------------------------------------------------------------------

_MAX_INTEGER_LENGTH = 256
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
_MAX_FINITE_GROUP_ORDER = 4_096
_MAX_FINITE_GROUP_RANK = 6
_MAX_FINITE_GROUP_FACTOR_SIZE = 256

BoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=_MAX_INTEGER_LENGTH,
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


class FactorizationResourceBudget(ContractModel):
    """Execution budget for complete integer factorization-derived operations."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=30)


class FactorizationRequest(ContractModel):
    """One integer and an explicit budget for an isolated SymPy computation."""

    value: BoundedInteger
    resource_budget: FactorizationResourceBudget = Field(
        default_factory=FactorizationResourceBudget
    )


class PowerfulNumberRequest(FactorizationRequest):
    """One positive integer for an exact powerful-number decision."""

    @model_validator(mode="after")
    def require_positive_value(self) -> Self:
        if int(self.value) < 1:
            raise ValueError("powerful-number input must be positive")
        return self


class ArithmeticFunctionRequest(ContractModel):
    """A small nonnegative integer with an explicit factorization budget."""

    n: StrictInt = Field(ge=0, le=_MAX_N_SMALL)
    resource_budget: FactorizationResourceBudget = Field(
        default_factory=FactorizationResourceBudget
    )


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


class FiniteAbelianGroupFactorizationRequest(ContractModel):
    """Two bounded integer-vector factors in a product of cyclic groups."""

    moduli: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    left: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_FACTOR_SIZE
    )
    right: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_FACTOR_SIZE
    )

    @model_validator(mode="after")
    def require_bounded_product_group(self) -> Self:
        if any(modulus < 2 or modulus > 1_000_000 for modulus in self.moduli):
            raise ValueError("cyclic moduli must be between 2 and 1,000,000")
        if math.prod(self.moduli) > _MAX_FINITE_GROUP_ORDER:
            raise ValueError("finite abelian group exceeds the 4,096-element bound")
        if len(self.left) * len(self.right) > _MAX_FINITE_GROUP_ORDER:
            raise ValueError("factor Cartesian product exceeds the 4,096-pair bound")
        if any(
            len(element) != len(self.moduli)
            for factor in (self.left, self.right)
            for element in factor
        ):
            raise ValueError("every factor element must match the group rank")
        if any(
            abs(coordinate) > 1_000_000
            for factor in (self.left, self.right)
            for element in factor
            for coordinate in element
        ):
            raise ValueError("factor coordinates exceed the input bound")
        normalized_factors = tuple(
            tuple(
                tuple(
                    coordinate % modulus
                    for coordinate, modulus in zip(element, self.moduli, strict=True)
                )
                for element in factor
            )
            for factor in (self.left, self.right)
        )
        if any(len(factor) != len(set(factor)) for factor in normalized_factors):
            raise ValueError("factor elements must be distinct after normalization")
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


class DiscreteLogarithmBudget(ContractModel):
    """Total wall-clock budget for one isolated SymPy computation."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=30)


class DiscreteLogarithmRequest(ContractModel):
    """A bounded modular discrete-logarithm problem."""

    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    resource_budget: DiscreteLogarithmBudget = Field(
        default_factory=DiscreteLogarithmBudget
    )

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
    """Inline residue-image summary with an optional durable assignment ledger."""

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


class FiniteAbelianRepresentationCount(ContractModel):
    representation_count: StrictInt = Field(ge=0, le=_MAX_FINITE_GROUP_ORDER)
    element_count: StrictInt = Field(ge=1, le=_MAX_FINITE_GROUP_ORDER)


class FiniteAbelianRepresentationWitness(ContractModel):
    element: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    left: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_FINITE_GROUP_RANK)
    right: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    other_left: tuple[StrictInt, ...] | None = Field(
        default=None, min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    other_right: tuple[StrictInt, ...] | None = Field(
        default=None, min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )

    @model_validator(mode="after")
    def require_two_representations(self) -> Self:
        if self.other_left is None or self.other_right is None:
            raise ValueError("duplicate witnesses require two complete representations")
        return self


class FiniteAbelianGroupFactorizationResult(ContractModel):
    """Complete unique-representation summary for ``G = left + right``."""

    semantics_version: Literal["finite-abelian-group-factorization.v1"]
    moduli: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    normalized_left: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_FACTOR_SIZE
    )
    normalized_right: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_FACTOR_SIZE
    )
    group_order: StrictInt = Field(ge=2, le=_MAX_FINITE_GROUP_ORDER)
    pair_count: StrictInt = Field(ge=1, le=_MAX_FINITE_GROUP_ORDER)
    distinct_sum_count: StrictInt = Field(ge=1, le=_MAX_FINITE_GROUP_ORDER)
    representation_histogram: tuple[FiniteAbelianRepresentationCount, ...] = Field(
        min_length=1, max_length=_MAX_FINITE_GROUP_ORDER
    )
    is_exact_factorization: StrictBool
    first_missing: tuple[StrictInt, ...] | None = Field(
        default=None, min_length=1, max_length=_MAX_FINITE_GROUP_RANK
    )
    first_duplicate: FiniteAbelianRepresentationWitness | None = None

    @model_validator(mode="after")
    def bind_factorization_summary(self) -> Self:
        self._require_bounded_group_shape()
        self._require_histogram_invariants()
        self._require_witness_invariants()
        self._require_complete_replay()
        return self

    def _require_bounded_group_shape(self) -> None:
        if any(modulus < 2 or modulus > 1_000_000 for modulus in self.moduli):
            raise ValueError("cyclic moduli must be between 2 and 1,000,000")
        if self.group_order != math.prod(self.moduli):
            raise ValueError("group order must equal the product of cyclic moduli")
        factors = (self.normalized_left, self.normalized_right)
        if any(
            len(element) != len(self.moduli) for factor in factors for element in factor
        ):
            raise ValueError(
                "every normalized factor element must match the group rank"
            )
        if any(
            coordinate < 0 or coordinate >= modulus
            for factor in factors
            for element in factor
            for coordinate, modulus in zip(element, self.moduli, strict=True)
        ):
            raise ValueError("normalized factor coordinates must be canonical residues")
        if any(len(factor) != len(set(factor)) for factor in factors):
            raise ValueError("normalized factor elements must be unique")
        if self.pair_count != len(self.normalized_left) * len(self.normalized_right):
            raise ValueError("pair count must equal the factor Cartesian-product size")

    def _require_histogram_invariants(self) -> None:
        counts = tuple(
            item.representation_count for item in self.representation_histogram
        )
        if counts != tuple(sorted(set(counts))):
            raise ValueError(
                "representation histogram counts must be unique and increasing"
            )
        if (
            sum(item.element_count for item in self.representation_histogram)
            != self.group_order
        ):
            raise ValueError("representation histogram must cover the complete group")
        if (
            sum(
                item.representation_count * item.element_count
                for item in self.representation_histogram
            )
            != self.pair_count
        ):
            raise ValueError("representation histogram must cover every factor pair")
        distinct_sum_count = sum(
            item.element_count
            for item in self.representation_histogram
            if item.representation_count > 0
        )
        if self.distinct_sum_count != distinct_sum_count:
            raise ValueError(
                "distinct sum count must match the positive histogram entries"
            )
        expected = self.pair_count == self.group_order and all(
            item.representation_count == 1 for item in self.representation_histogram
        )
        if self.is_exact_factorization != expected:
            raise ValueError(
                "factorization decision does not match the complete histogram"
            )

    def _require_witness_invariants(self) -> None:
        if self.is_exact_factorization and (self.first_missing or self.first_duplicate):
            raise ValueError("exact factorizations cannot carry failure witnesses")
        has_missing = any(
            item.representation_count == 0 for item in self.representation_histogram
        )
        has_duplicate = any(
            item.representation_count > 1 for item in self.representation_histogram
        )
        if (self.first_missing is not None) != has_missing:
            raise ValueError("missing witness presence must match the histogram")
        if (self.first_duplicate is not None) != has_duplicate:
            raise ValueError("duplicate witness presence must match the histogram")
        if self.first_missing is not None:
            self._require_canonical_group_element(
                self.first_missing, field="missing witness"
            )
        if self.first_duplicate is not None:
            duplicate = self.first_duplicate
            for field, element in (
                ("duplicate element", duplicate.element),
                ("duplicate left", duplicate.left),
                ("duplicate right", duplicate.right),
                ("duplicate other_left", duplicate.other_left),
                ("duplicate other_right", duplicate.other_right),
            ):
                assert element is not None
                self._require_canonical_group_element(element, field=field)
            first_pair = (duplicate.left, duplicate.right)
            second_pair = (duplicate.other_left, duplicate.other_right)
            if first_pair == second_pair:
                raise ValueError("duplicate witness representations must be distinct")
            for left, right in (first_pair, second_pair):
                assert left is not None and right is not None
                total = tuple(
                    (left_coordinate + right_coordinate) % modulus
                    for left_coordinate, right_coordinate, modulus in zip(
                        left, right, self.moduli, strict=True
                    )
                )
                if total != duplicate.element:
                    raise ValueError(
                        "duplicate witness representations must produce its element"
                    )

    def _require_canonical_group_element(
        self, element: tuple[int, ...], *, field: str
    ) -> None:
        if len(element) != len(self.moduli) or any(
            coordinate < 0 or coordinate >= modulus
            for coordinate, modulus in zip(element, self.moduli, strict=True)
        ):
            raise ValueError(f"{field} must be a canonical group element")

    def _require_complete_replay(self) -> None:
        representations: dict[
            tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
        ] = {}
        for left in self.normalized_left:
            for right in self.normalized_right:
                total = tuple(
                    (left_coordinate + right_coordinate) % modulus
                    for left_coordinate, right_coordinate, modulus in zip(
                        left, right, self.moduli, strict=True
                    )
                )
                representations.setdefault(total, []).append((left, right))
        group = tuple(product(*(range(modulus) for modulus in self.moduli)))
        expected_histogram = Counter(
            len(representations.get(element, ())) for element in group
        )
        actual_histogram = {
            item.representation_count: item.element_count
            for item in self.representation_histogram
        }
        if actual_histogram != dict(expected_histogram):
            raise ValueError("representation histogram must match exact replay")
        if self.distinct_sum_count != len(representations):
            raise ValueError("distinct sum count must match exact replay")
        expected_missing = next(
            (element for element in group if element not in representations), None
        )
        if self.first_missing != expected_missing:
            raise ValueError("missing witness must be the first missing group element")
        duplicate_element = next(
            (element for element in group if len(representations.get(element, ())) > 1),
            None,
        )
        expected_duplicate = None
        if duplicate_element is not None:
            first, second = representations[duplicate_element][:2]
            expected_duplicate = FiniteAbelianRepresentationWitness(
                element=duplicate_element,
                left=first[0],
                right=first[1],
                other_left=second[0],
                other_right=second[1],
            )
        if self.first_duplicate != expected_duplicate:
            raise ValueError("duplicate witness must match the first exact replay")


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
    """A completed discrete-log result; interruption has a separate envelope."""

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


class DiscreteLogarithmObligation(ContractModel):
    """Independent checks still open for a completed producer result."""

    obligation_schema_version: Literal["1"] = "1"
    predicate: Literal["MODULAR_DISCRETE_LOGARITHM"] = "MODULAR_DISCRETE_LOGARITHM"
    base: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=_MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=_MAX_MODULUS)
    status: Literal["SOLVED", "UNSOLVABLE"]
    discrete_log: StrictInt | None = Field(default=None, ge=0)
    required_checks: tuple[
        Literal[
            "DISCRETE_LOG_WITNESS_REPLAY",
            "DISCRETE_LOG_NONSOLVABILITY",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def require_status_specific_check(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise ValueError("base and target must be less than the modulus")
        expected = (
            ("DISCRETE_LOG_WITNESS_REPLAY",)
            if self.status == "SOLVED"
            else ("DISCRETE_LOG_NONSOLVABILITY",)
        )
        if self.required_checks != expected:
            raise ValueError("required checks must match the discrete-log status")
        if (self.discrete_log is None) != (self.status == "UNSOLVABLE"):
            raise ValueError("candidate exponent must match the discrete-log status")
        return self
