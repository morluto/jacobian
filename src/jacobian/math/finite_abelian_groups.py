"""Exact operations in explicit products of finite cyclic groups."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import comb, lcm, prod
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

if TYPE_CHECKING:
    from sympy import Poly, Symbol

MAX_FINITE_GROUP_ORDER = 4_096
MAX_FINITE_GROUP_RANK = 6
MAX_FINITE_GROUP_FACTOR_SIZE = 256
MAX_FINITE_GROUP_MODULUS = (1 << 53) - 1
MAX_FINITE_GROUP_COORDINATE = (1 << 53) - 1
MAX_SPECTRAL_SET_SIZE = 4_096
MAX_SPECTRAL_CYCLOTOMIC_DEGREE = 60
MAX_SPECTRAL_CHARACTER_TERMS = 258_048
MAX_SPECTRAL_CYCLOTOMIC_REDUCTIONS = 4_032
MAX_SPECTRAL_CYCLOTOMIC_DENSE_OPS = 524_288
MAX_SPECTRAL_CYCLOTOMIC_COEFFICIENT_BITS = 64
MAX_SPECTRAL_CYCLOTOMIC_INTERMEDIATE_BITS = 256
MAX_SPECTRAL_REMAINDER_COEFFICIENT_BITS = 2_048
MAX_SPECTRAL_REMAINDER_COEFFICIENT_DIGITS = (
    MAX_SPECTRAL_REMAINDER_COEFFICIENT_BITS * 30_103 + 99_999
) // 100_000 + 1
MAX_SPECTRAL_RESULT_BYTES = 32_768


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_abelian_group.{reason}", message)


GroupElement = tuple[StrictInt, ...]
FiniteGroupModulus = Annotated[
    StrictInt,
    Field(ge=2, le=MAX_FINITE_GROUP_MODULUS),
]
BoundedGroupCoordinate = Annotated[
    StrictInt,
    Field(
        ge=-MAX_FINITE_GROUP_COORDINATE,
        le=MAX_FINITE_GROUP_COORDINATE,
    ),
]
BoundedGroupElement = tuple[BoundedGroupCoordinate, ...]
CanonicalGroupCoordinate = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_FINITE_GROUP_COORDINATE),
]
CanonicalGroupElement = tuple[CanonicalGroupCoordinate, ...]
BoundedRemainderInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_SPECTRAL_REMAINDER_COEFFICIENT_DIGITS + 1),
]


class FiniteAbelianProductGroup(StrictModel):
    """An ordered product of cyclic moduli between 2 and the safe integer 2**53 - 1.

    Moduli and canonical residues serialize as raw JSON integers inside exact
    results, so this reusable value is bounded by the interoperable
    safe-integer range rather than any operation's work envelope. The axis
    count carries no ceiling of its own: consuming operations derive their
    execution envelope from the supplied rows, the group exponent, and the
    serialized-result size that already scales with the rank.
    """

    moduli: tuple[FiniteGroupModulus, ...] = Field(min_length=1)

    @property
    def order(self) -> int:
        """Return the product-group order."""

        return prod(self.moduli)

    @property
    def exponent(self) -> int:
        """Return the least common multiple of the ordered cyclic moduli."""

        return lcm(*self.moduli)


class FiniteAbelianGroupFactorizationRequest(StrictModel):
    """Two bounded integer-vector factors in a product of cyclic groups.

    The kernel enumerates every element of the ambient group, so this
    operation keeps its own order bound at 4,096 independent of the reusable
    group value's domain.
    """

    moduli: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    left: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    right: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )

    @model_validator(mode="after")
    def require_bounded_product_group(self) -> Self:
        FiniteAbelianProductGroup(moduli=self.moduli)
        if prod(self.moduli) > MAX_FINITE_GROUP_ORDER:
            raise _validation_error(
                "factorization_group_order",
                "finite abelian group exceeds the 4,096-element bound",
            )
        if len(self.left) * len(self.right) > MAX_FINITE_GROUP_ORDER:
            raise _validation_error(
                "factorization_pair_count",
                "factor Cartesian product exceeds the 4,096-pair bound",
            )
        if any(
            len(element) != len(self.moduli)
            for factor in (self.left, self.right)
            for element in factor
        ):
            raise _validation_error(
                "factorization_element_rank",
                "every factor element must match the group rank",
            )
        if any(
            abs(coordinate) > MAX_FINITE_GROUP_COORDINATE
            for factor in (self.left, self.right)
            for element in factor
            for coordinate in element
        ):
            raise _validation_error(
                "factorization_coordinate_bound",
                "factor coordinates exceed the input bound",
            )
        for factor in (self.left, self.right):
            normalized = {
                tuple(
                    coordinate % modulus
                    for coordinate, modulus in zip(element, self.moduli, strict=True)
                )
                for element in factor
            }
            if len(normalized) != len(factor):
                raise _validation_error(
                    "factorization_duplicate_element",
                    "factor elements must be distinct after normalization",
                )
        return self


class FiniteAbelianSpectralPairSource(StrictModel):
    """A point set and candidate frequency set in one explicit product group.

    Coordinates are reduced on their declared axes and rows are sorted
    lexicographically. Distinctness is checked after reduction, so this value
    has one canonical residue-tuple representation. The per-set row cap is a
    defensive materialization fallback; operation admission is governed by
    serialized result bytes and the derived reduction budgets.
    """

    group: FiniteAbelianProductGroup
    points: tuple[BoundedGroupElement, ...] = Field(
        min_length=0,
        max_length=MAX_SPECTRAL_SET_SIZE,
    )
    frequencies: tuple[BoundedGroupElement, ...] = Field(
        min_length=0,
        max_length=MAX_SPECTRAL_SET_SIZE,
    )

    @model_validator(mode="after")
    def canonicalize_residue_sets(self) -> Self:
        rank = len(self.group.moduli)

        def canonicalize(
            rows: tuple[BoundedGroupElement, ...], label: str
        ) -> tuple[tuple[int, ...], ...]:
            if any(len(row) != rank for row in rows):
                raise _validation_error(
                    "source_row_rank", f"every {label} row must match the group rank"
                )
            normalized = tuple(
                tuple(
                    coordinate % modulus
                    for coordinate, modulus in zip(row, self.group.moduli, strict=True)
                )
                for row in rows
            )
            if len(set(normalized)) != len(normalized):
                raise _validation_error(
                    "source_duplicate_row",
                    f"{label} rows must be distinct after residue normalization",
                )
            return tuple(sorted(normalized))

        object.__setattr__(self, "points", canonicalize(self.points, "point"))
        object.__setattr__(
            self,
            "frequencies",
            canonicalize(self.frequencies, "frequency"),
        )
        return self


class FiniteAbelianSpectralPairRequest(StrictModel):
    """Decide whether the supplied frequencies are a spectrum of the points.

    Equal-size sources with at least two frequencies are admitted through the
    derived reduction envelope: 60-degree cyclotomics, 258,048 point-character
    terms, 4,032 exact reductions, bounded construction intermediates, and a
    32,768-byte serialized result; those counts include the result model's
    independent replay. Cardinality mismatches, singleton pairs, and the
    equal-empty pair need no cyclotomic reduction and are admitted by source
    and result size alone.
    """

    source: FiniteAbelianSpectralPairSource

    @model_validator(mode="after")
    def require_bounded_exact_decision(self) -> Self:
        try:
            _spectral_pair_work(self.source)
        except ValueError as error:
            raise _validation_error("spectral_admission", str(error)) from error
        return self


class FiniteAbelianNonorthogonalityWitness(StrictModel):
    """First nonorthogonal frequency pair and its exact cyclotomic remainder.

    ``remainder_coefficients[k]`` is the coefficient of ``X^k`` in the
    remainder of the character-sum polynomial modulo the group-exponent
    cyclotomic polynomial. The tuple is dense and has length ``phi(N)``.
    """

    left_frequency: CanonicalGroupElement = Field(min_length=1)
    right_frequency: CanonicalGroupElement = Field(min_length=1)
    difference: CanonicalGroupElement = Field(min_length=1)
    remainder_coefficients: tuple[BoundedRemainderInteger, ...] = Field(
        min_length=1,
        max_length=MAX_SPECTRAL_CYCLOTOMIC_DEGREE,
    )

    @model_validator(mode="after")
    def require_bounded_nonzero_remainder(self) -> Self:
        if any(
            len(coefficient.lstrip("-")) > MAX_SPECTRAL_REMAINDER_COEFFICIENT_DIGITS
            for coefficient in self.remainder_coefficients
        ):
            raise _validation_error(
                "remainder_digit_bound",
                "cyclotomic remainder coefficient exceeds its digit bound",
            )
        if all(coefficient == "0" for coefficient in self.remainder_coefficients):
            raise _validation_error(
                "zero_remainder",
                "a nonorthogonality witness must have nonzero remainder",
            )
        return self


SpectralPairDecisionReason = Literal[
    "SPECTRAL",
    "CARDINALITY_MISMATCH",
    "NONORTHOGONAL_FREQUENCIES",
]


class FiniteAbelianSpectralPairResult(StrictModel):
    """Exact, source-bound decision for a finite-Abelian spectral pair.

    The fixed dual pairing is
    ``chi_lambda(a) = exp(2*pi*i*sum(lambda_j*a_j/m_j))``. For an equal-size
    pair, every distinct frequency pair is checked in lexicographic source
    order. The restricted characters all have nonzero norm, so ``|A|``
    pairwise-orthogonal characters form a basis of ``C^A``. Result validation
    repeats the exact cyclotomic reductions and binds the decision, first
    witness, convention, and retained source together.
    """

    source: FiniteAbelianSpectralPairSource
    is_spectral: StrictBool
    reason: SpectralPairDecisionReason
    first_nonorthogonal_pair: FiniteAbelianNonorthogonalityWitness | None = None
    character_convention: Literal["POSITIVE_PRODUCT_DUAL_PAIRING"] = (
        "POSITIVE_PRODUCT_DUAL_PAIRING"
    )

    @model_validator(mode="after")
    def replay_exact_decision(self) -> Self:
        expected = _finite_abelian_spectral_pair_decision_data(self.source)
        observed = (
            self.is_spectral,
            self.reason,
            self.first_nonorthogonal_pair,
        )
        required = (
            expected.is_spectral,
            expected.reason,
            expected.first_nonorthogonal_pair,
        )
        if observed != required:
            raise _validation_error(
                "spectral_replay",
                "spectral-pair result must equal the replayed exact decision",
            )
        return self


@dataclass(frozen=True, slots=True)
class _SpectralPairWork:
    group_exponent: int
    cyclotomic_degree: int | None
    character_terms: int
    cyclotomic_reductions: int
    cyclotomic_dense_ops: int
    cyclotomic_coefficient_bits: int
    cyclotomic_intermediate_bits: int
    remainder_coefficient_bits: int
    predicted_result_bytes: int


@dataclass(frozen=True, slots=True)
class _SpectralPairDecisionData:
    is_spectral: bool
    reason: SpectralPairDecisionReason
    first_nonorthogonal_pair: FiniteAbelianNonorthogonalityWitness | None


def _euler_totient(value: int) -> int:
    """Return phi(value) by bounded integer trial division for preflight."""

    result = value
    remaining = value
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            while remaining % prime == 0:
                remaining //= prime
            result -= result // prime
        prime += 1
    if remaining > 1:
        result -= result // remaining
    return result


def _decimal_digits_from_bits(bits: int) -> int:
    """Conservatively convert a positive binary bit bound to decimal digits."""

    return (bits * 30_103 + 99_999) // 100_000 + 1


def _predicted_spectral_source_bytes(source: FiniteAbelianSpectralPairSource) -> int:
    """Bound the serialized canonical source retained in any result."""

    coordinate_digits = max(len(str(modulus - 1)) for modulus in source.group.moduli)
    rank = len(source.group.moduli)
    row_bytes = 4 + rank * (coordinate_digits + 3)
    return (
        2_048
        + rank * (coordinate_digits + 3)
        + (len(source.points) + len(source.frequencies)) * row_bytes
    )


def _predicted_spectral_witness_bytes(
    *,
    rank: int,
    coordinate_digits: int,
    cyclotomic_degree: int,
    remainder_coefficient_bits: int,
) -> int:
    """Bound the largest canonical failure-witness serialization."""

    return (
        512
        + 3 * rank * (coordinate_digits + 3)
        + cyclotomic_degree
        * (_decimal_digits_from_bits(remainder_coefficient_bits) + 4)
    )


def _spectral_pair_work(source: FiniteAbelianSpectralPairSource) -> _SpectralPairWork:
    """Preflight exact work, intermediate growth, and output obligations.

    A cardinality mismatch, an equal singleton pair, and the equal empty pair
    are decided without a cyclotomic backend call; their admission is the
    serialized source-plus-decision byte bound. Otherwise, the public
    operation plus source-bound result validation perform two complete passes.
    Each pass checks at most ``C(|Lambda|, 2)`` pairs and ``|A|`` character
    terms per pair. No check depends on the ambient group order or modulus
    size, only on the supplied rows, the rank, and the group exponent.

    The serialized-source bound is enforced before any exponent arithmetic.
    It depends only on the declared moduli and supplied rows, so an oversized
    request never runs superlinear preflight work: once it holds, the axis
    count fits the result budget and every intermediate least common multiple
    stays below the product of the admitted moduli.

    Every coefficient of ``Phi_N`` is at most ``2**phi(N)`` in absolute value:
    it is an elementary symmetric sum of ``phi(N)`` unit-modulus roots. SymPy's
    inflate/exact-quotient construction stays within a conservative
    ``(N + 1) * 2**(2*N)`` coefficient-height envelope. Two constructions use
    at most ``10*bit_length(N)*(N+1)^2`` conservative dense coefficient
    operations, including inflation and monic exact division. Monic long
    division of a degree-``< N`` character polynomial starts at height ``|A|``
    and has at most ``N - phi(N)`` eliminations, each growing height by at most
    ``1 + 2**phi(N)``. The bit bounds below are the corresponding integer upper
    bounds, computed before SymPy is invoked. The dense-op and intermediate
    budgets grow only with ``N``, so they are enforced before ``Phi_N``'s
    totient trial division; every rejected exponent is already over one of
    those derived budgets, and every surviving exponent makes that preflight
    itself trivially bounded.
    """

    source_bytes = _predicted_spectral_source_bytes(source)
    if source_bytes > MAX_SPECTRAL_RESULT_BYTES:
        raise ValueError("spectral-pair result exceeds its serialized byte bound")
    coordinate_digits = max(len(str(modulus - 1)) for modulus in source.group.moduli)
    rank = len(source.group.moduli)

    exponent = source.group.exponent
    needs_reduction = (
        len(source.points) == len(source.frequencies) and len(source.frequencies) > 1
    )
    if not needs_reduction:
        work = _SpectralPairWork(
            group_exponent=exponent,
            cyclotomic_degree=None,
            character_terms=0,
            cyclotomic_reductions=0,
            cyclotomic_dense_ops=0,
            cyclotomic_coefficient_bits=0,
            cyclotomic_intermediate_bits=0,
            remainder_coefficient_bits=0,
            predicted_result_bytes=(
                source_bytes
                + _predicted_spectral_witness_bytes(
                    rank=rank,
                    coordinate_digits=coordinate_digits,
                    cyclotomic_degree=0,
                    remainder_coefficient_bits=0,
                )
            ),
        )
        if work.predicted_result_bytes > MAX_SPECTRAL_RESULT_BYTES:
            raise ValueError("spectral-pair result exceeds its serialized byte bound")
        return work

    cyclotomic_dense_ops = 10 * exponent.bit_length() * (exponent + 1) * (exponent + 1)
    cyclotomic_intermediate_bits = 2 * exponent + (exponent + 1).bit_length() + 1
    if cyclotomic_dense_ops > MAX_SPECTRAL_CYCLOTOMIC_DENSE_OPS:
        raise ValueError("cyclotomic construction work exceeds its dense-op bound")
    if cyclotomic_intermediate_bits > MAX_SPECTRAL_CYCLOTOMIC_INTERMEDIATE_BITS:
        raise ValueError("cyclotomic construction intermediate exceeds its bit bound")

    degree = _euler_totient(exponent)
    pair_count = comb(len(source.frequencies), 2)
    character_terms = 2 * pair_count * len(source.points)
    reductions = 2 * pair_count
    cyclotomic_coefficient_bits = degree + 1
    remainder_coefficient_bits = len(source.points).bit_length() + (degree + 1) * (
        exponent - degree
    )
    predicted_result_bytes = source_bytes + _predicted_spectral_witness_bytes(
        rank=rank,
        coordinate_digits=coordinate_digits,
        cyclotomic_degree=degree,
        remainder_coefficient_bits=remainder_coefficient_bits,
    )

    if degree > MAX_SPECTRAL_CYCLOTOMIC_DEGREE:
        raise ValueError("cyclotomic degree exceeds the exact reduction bound")
    if character_terms > MAX_SPECTRAL_CHARACTER_TERMS:
        raise ValueError("spectral-pair character-term work exceeds its bound")
    if reductions > MAX_SPECTRAL_CYCLOTOMIC_REDUCTIONS:
        raise ValueError("spectral-pair cyclotomic reductions exceed their bound")
    if cyclotomic_coefficient_bits > MAX_SPECTRAL_CYCLOTOMIC_COEFFICIENT_BITS:
        raise ValueError("cyclotomic coefficient exceeds its bit bound")
    if remainder_coefficient_bits > MAX_SPECTRAL_REMAINDER_COEFFICIENT_BITS:
        raise ValueError("cyclotomic remainder intermediate exceeds its bit bound")
    if predicted_result_bytes > MAX_SPECTRAL_RESULT_BYTES:
        raise ValueError("spectral-pair result exceeds its serialized byte bound")
    return _SpectralPairWork(
        group_exponent=exponent,
        cyclotomic_degree=degree,
        character_terms=character_terms,
        cyclotomic_reductions=reductions,
        cyclotomic_dense_ops=cyclotomic_dense_ops,
        cyclotomic_coefficient_bits=cyclotomic_coefficient_bits,
        cyclotomic_intermediate_bits=cyclotomic_intermediate_bits,
        remainder_coefficient_bits=remainder_coefficient_bits,
        predicted_result_bytes=predicted_result_bytes,
    )


def _character_sum_remainder(
    source: FiniteAbelianSpectralPairSource,
    left_frequency: tuple[int, ...],
    right_frequency: tuple[int, ...],
    *,
    generator: Symbol,
    cyclotomic: Poly,
    cyclotomic_degree: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Reduce one exact character sum modulo the exponent cyclotomic."""

    from sympy import Poly
    from sympy.polys.domains import ZZ

    exponent = source.group.exponent
    difference = tuple(
        (left - right) % modulus
        for left, right, modulus in zip(
            left_frequency,
            right_frequency,
            source.group.moduli,
            strict=True,
        )
    )
    counts: Counter[int] = Counter()
    for point in source.points:
        power = (
            sum(
                (exponent // modulus) * coordinate * frequency_difference
                for coordinate, frequency_difference, modulus in zip(
                    point,
                    difference,
                    source.group.moduli,
                    strict=True,
                )
            )
            % exponent
        )
        counts[power] += 1
    polynomial = Poly.from_dict(
        {(power,): coefficient for power, coefficient in counts.items()},
        generator,
        domain=ZZ,
    )
    remainder = polynomial.rem(cyclotomic, auto=False)
    coefficients = tuple(
        format_canonical_integer(int(remainder.nth(power)))
        for power in range(cyclotomic_degree)
    )
    return difference, coefficients


def _finite_abelian_spectral_pair_decision_data(
    source: FiniteAbelianSpectralPairSource,
) -> _SpectralPairDecisionData:
    """Compute replayable decision data without constructing the result model."""

    work = _spectral_pair_work(source)
    if len(source.points) != len(source.frequencies):
        return _SpectralPairDecisionData(
            is_spectral=False,
            reason="CARDINALITY_MISMATCH",
            first_nonorthogonal_pair=None,
        )
    if len(source.frequencies) <= 1:
        return _SpectralPairDecisionData(
            is_spectral=True,
            reason="SPECTRAL",
            first_nonorthogonal_pair=None,
        )

    from sympy import Symbol, cyclotomic_poly
    from sympy.polys.domains import ZZ

    degree = cast(int, work.cyclotomic_degree)
    generator = Symbol("_finite_abelian_character")
    cyclotomic = cast(
        "Poly",
        cyclotomic_poly(work.group_exponent, generator, polys=True),
    )
    if cyclotomic.domain != ZZ or cyclotomic.LC() != 1 or cyclotomic.degree() != degree:
        raise RuntimeError("SymPy returned an incompatible cyclotomic polynomial")

    for left_index, left_frequency in enumerate(source.frequencies):
        for right_frequency in source.frequencies[left_index + 1 :]:
            difference, coefficients = _character_sum_remainder(
                source,
                left_frequency,
                right_frequency,
                generator=generator,
                cyclotomic=cyclotomic,
                cyclotomic_degree=degree,
            )
            if any(coefficient != "0" for coefficient in coefficients):
                return _SpectralPairDecisionData(
                    is_spectral=False,
                    reason="NONORTHOGONAL_FREQUENCIES",
                    first_nonorthogonal_pair=FiniteAbelianNonorthogonalityWitness(
                        left_frequency=left_frequency,
                        right_frequency=right_frequency,
                        difference=difference,
                        remainder_coefficients=coefficients,
                    ),
                )
    return _SpectralPairDecisionData(
        is_spectral=True,
        reason="SPECTRAL",
        first_nonorthogonal_pair=None,
    )


def decide_finite_abelian_spectral_pair(
    source: FiniteAbelianSpectralPairSource,
) -> FiniteAbelianSpectralPairResult:
    """Decide exact finite-Abelian spectrality under the positive pairing."""

    decision = _finite_abelian_spectral_pair_decision_data(source)
    return FiniteAbelianSpectralPairResult(
        source=source,
        is_spectral=decision.is_spectral,
        reason=decision.reason,
        first_nonorthogonal_pair=decision.first_nonorthogonal_pair,
    )


def _run_finite_abelian_spectral_pair(
    request: FiniteAbelianSpectralPairRequest,
) -> FiniteAbelianSpectralPairResult:
    """Adapt the catalog request to the native source-value function."""

    return decide_finite_abelian_spectral_pair(request.source)


class FiniteAbelianRepresentationCount(StrictModel):
    """Number of group elements having one representation count."""

    representation_count: StrictInt = Field(ge=0, le=MAX_FINITE_GROUP_ORDER)
    element_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)


class FiniteAbelianRepresentationWitness(StrictModel):
    """The first element with two distinct displayed representations."""

    element: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    left: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    right: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    other_left: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    other_right: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)


class FiniteAbelianGroupFactorizationResult(StrictModel):
    """Complete unique-representation summary for ``G = left + right``."""

    moduli: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    normalized_left: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    normalized_right: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    group_order: StrictInt = Field(ge=2, le=MAX_FINITE_GROUP_ORDER)
    pair_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)
    distinct_sum_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)
    representation_histogram: tuple[FiniteAbelianRepresentationCount, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_ORDER,
    )
    is_exact_factorization: StrictBool
    first_missing: GroupElement | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    first_duplicate: FiniteAbelianRepresentationWitness | None = None
    convention: Literal["UNIQUE_SUM_REPRESENTATION_IN_PRODUCT_OF_CYCLIC_GROUPS"] = (
        "UNIQUE_SUM_REPRESENTATION_IN_PRODUCT_OF_CYCLIC_GROUPS"
    )

    @model_validator(mode="after")
    def bind_structural_summary(self) -> Self:
        self._validate_group_structure()
        self._validate_factor_sets()
        self._validate_histogram()
        self._validate_decision_witness_presence()
        self._validate_witnesses()
        return self

    def _validate_group_structure(self) -> None:
        if self.group_order != prod(self.moduli):
            raise ValueError("group order must equal the product of cyclic moduli")

    def _validate_factor_sets(self) -> None:
        factors = (self.normalized_left, self.normalized_right)
        for factor in factors:
            if factor != tuple(sorted(set(factor))):
                raise ValueError("normalized factors must be unique and sorted")
            if any(len(element) != len(self.moduli) for element in factor):
                raise ValueError("normalized factor elements must match the group rank")
            if any(
                coordinate < 0 or coordinate >= modulus
                for element in factor
                for coordinate, modulus in zip(element, self.moduli, strict=True)
            ):
                raise ValueError("normalized factor coordinates must be residues")
        if self.pair_count != len(self.normalized_left) * len(self.normalized_right):
            raise ValueError("pair count must equal the factor product size")

    def _validate_histogram(self) -> None:
        counts = tuple(
            item.representation_count for item in self.representation_histogram
        )
        if counts != tuple(sorted(set(counts))):
            raise ValueError("histogram representation counts must be increasing")
        if (
            sum(item.element_count for item in self.representation_histogram)
            != self.group_order
        ):
            raise ValueError("representation histogram must cover the group")
        if (
            sum(
                item.representation_count * item.element_count
                for item in self.representation_histogram
            )
            != self.pair_count
        ):
            raise ValueError("representation histogram must cover every factor pair")
        positive_count = sum(
            item.element_count
            for item in self.representation_histogram
            if item.representation_count > 0
        )
        if positive_count != self.distinct_sum_count:
            raise ValueError("distinct sum count must match the histogram")

    def _validate_decision_witness_presence(self) -> None:
        exact = (
            self.pair_count == self.group_order
            and self.representation_histogram
            == (
                FiniteAbelianRepresentationCount(
                    representation_count=1,
                    element_count=self.group_order,
                ),
            )
        )
        if self.is_exact_factorization != exact:
            raise ValueError("factorization decision must match the complete histogram")
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
        if self.is_exact_factorization and (
            self.first_missing is not None or self.first_duplicate is not None
        ):
            raise ValueError("exact factorizations cannot carry failure witnesses")

    def _validate_witnesses(self) -> None:
        def canonical(element: tuple[int, ...]) -> bool:
            return len(element) == len(self.moduli) and all(
                0 <= coordinate < modulus
                for coordinate, modulus in zip(element, self.moduli, strict=True)
            )

        if self.first_missing is not None and not canonical(self.first_missing):
            raise ValueError("missing witness must be a canonical group element")
        duplicate = self.first_duplicate
        if duplicate is None:
            return
        elements = (
            duplicate.element,
            duplicate.left,
            duplicate.right,
            duplicate.other_left,
            duplicate.other_right,
        )
        if not all(canonical(element) for element in elements):
            raise ValueError("duplicate witness elements must be canonical")
        first_pair = (duplicate.left, duplicate.right)
        second_pair = (duplicate.other_left, duplicate.other_right)
        if first_pair == second_pair:
            raise ValueError("duplicate witness representations must be distinct")
        for left, right in (first_pair, second_pair):
            total = tuple(
                (left_coordinate + right_coordinate) % modulus
                for left_coordinate, right_coordinate, modulus in zip(
                    left, right, self.moduli, strict=True
                )
            )
            if total != duplicate.element:
                raise ValueError("duplicate witness representations must sum correctly")


def finite_abelian_group_factorization(
    request: FiniteAbelianGroupFactorizationRequest,
) -> FiniteAbelianGroupFactorizationResult:
    """Exhaustively test unique representation in a product of cyclic groups."""

    moduli = request.moduli

    def normalize(element: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            coordinate % modulus
            for coordinate, modulus in zip(element, moduli, strict=True)
        )

    left = tuple(sorted(normalize(element) for element in request.left))
    right = tuple(sorted(normalize(element) for element in request.right))
    representations: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ] = {}
    for left_element in left:
        for right_element in right:
            total = tuple(
                (left_coordinate + right_coordinate) % modulus
                for left_coordinate, right_coordinate, modulus in zip(
                    left_element, right_element, moduli, strict=True
                )
            )
            representations.setdefault(total, []).append((left_element, right_element))
    group = tuple(product(*(range(modulus) for modulus in moduli)))
    histogram = Counter(len(representations.get(element, ())) for element in group)
    first_missing = next(
        (element for element in group if element not in representations),
        None,
    )
    duplicate_element = next(
        (element for element in group if len(representations.get(element, ())) > 1),
        None,
    )
    duplicate = None
    if duplicate_element is not None:
        first, second = representations[duplicate_element][:2]
        duplicate = FiniteAbelianRepresentationWitness(
            element=duplicate_element,
            left=first[0],
            right=first[1],
            other_left=second[0],
            other_right=second[1],
        )
    group_order = prod(moduli)
    exact = len(left) * len(right) == group_order and histogram == {1: group_order}
    return FiniteAbelianGroupFactorizationResult(
        moduli=moduli,
        normalized_left=left,
        normalized_right=right,
        group_order=group_order,
        pair_count=len(left) * len(right),
        distinct_sum_count=len(representations),
        representation_histogram=tuple(
            FiniteAbelianRepresentationCount(
                representation_count=count,
                element_count=histogram[count],
            )
            for count in sorted(histogram)
        ),
        is_exact_factorization=exact,
        first_missing=None if exact else first_missing,
        first_duplicate=None if exact else duplicate,
    )


__all__ = [
    "FiniteAbelianGroupFactorizationResult",
    "FiniteAbelianNonorthogonalityWitness",
    "FiniteAbelianProductGroup",
    "FiniteAbelianRepresentationCount",
    "FiniteAbelianRepresentationWitness",
    "FiniteAbelianSpectralPairResult",
    "FiniteAbelianSpectralPairSource",
    "decide_finite_abelian_spectral_pair",
    "finite_abelian_group_factorization",
]
