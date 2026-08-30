"""Affine-map word collision profile kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.algebra.affine_map_word_collision._models import (
    MAX_DEPTH,
    MAX_GENERATORS,
    CollisionRow,
    WordCollisionProfileResult,
)

__all__ = ["compute_word_collision_profile"]

MAX_COMPOSITION_WORK = 5_000_000


@dataclass(frozen=True, slots=True)
class WordCollisionAdmission:
    """Derived work, coefficient, and output bounds for one invocation."""

    generators: tuple[tuple[Fraction, Fraction], ...]
    word_count: int
    coefficient_height: RationalHeight
    result_bytes: int


def _decimal_digits_from_bits(bits: int) -> int:
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _fraction_height(value: Fraction) -> RationalHeight:
    return RationalHeight(
        _decimal_digits_from_bits(abs(value.numerator).bit_length()),
        _decimal_digits_from_bits(value.denominator.bit_length()),
    )


def _string_size(digits: int) -> int:
    # Reserve one byte for a possible leading minus sign in addition to the
    # JSON string quotes.  Heights intentionally track magnitude only.
    return digits + 3


def _max_height(values: Sequence[RationalHeight], default: RationalHeight) -> RationalHeight:
    if not values:
        return default
    return RationalHeight(
        max(value.numerator_digits for value in values),
        max(value.denominator_digits for value in values),
    )


def _coefficient_height(
    generators: tuple[tuple[Fraction, Fraction], ...], depth: int
) -> RationalHeight:
    """Bound coefficients while preserving zero-slope reset points."""
    nonzero = [(slope, intercept) for slope, intercept in generators if slope]
    zero = [intercept for slope, intercept in generators if not slope]
    nonzero_slope = RationalHeight(1, 1)
    nonzero_intercept = RationalHeight(1, 1)
    constant_intercept: RationalHeight | None = None
    max_nonzero_slope = _max_height(
        tuple(_fraction_height(slope) for slope, _ in nonzero), RationalHeight(1, 1)
    )
    max_nonzero_intercept = _max_height(
        tuple(_fraction_height(intercept) for _, intercept in nonzero),
        RationalHeight(1, 1),
    )
    max_zero_intercept = _max_height(
        tuple(_fraction_height(intercept) for intercept in zero), RationalHeight(1, 1)
    )
    for _ in range(depth):
        next_nonzero_slope: RationalHeight | None = None
        next_nonzero_intercept: RationalHeight | None = None
        if nonzero:
            next_nonzero_slope = nonzero_slope.product(max_nonzero_slope)
            next_nonzero_intercept = sum_heights(
                (
                    max_nonzero_slope.product(nonzero_intercept),
                    max_nonzero_intercept,
                )
            )
        next_constant = max_zero_intercept if zero else None
        if constant_intercept is not None and nonzero:
            grown_constant = sum_heights(
                (
                    max_nonzero_slope.product(constant_intercept),
                    max_nonzero_intercept,
                )
            )
            next_constant = (
                grown_constant
                if next_constant is None
                else _max_height((next_constant, grown_constant), next_constant)
            )
        if next_nonzero_slope is not None:
            assert next_nonzero_intercept is not None
            nonzero_slope = next_nonzero_slope
            nonzero_intercept = next_nonzero_intercept
        constant_intercept = next_constant
    return _max_height(
        tuple(
            height
            for height in (nonzero_slope, nonzero_intercept, constant_intercept)
            if height is not None
        ),
        RationalHeight(1, 1),
    )


def _rational_size(height: RationalHeight) -> int:
    return strict_json_object_size(
        (
            ("num", _string_size(height.numerator_digits)),
            ("den", _string_size(height.denominator_digits)),
        )
    )


def _array_size(value_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(value_sizes) - 1, 0) + sum(value_sizes)


def _admit_word_collision_profile(
    generators: Sequence[tuple[Fraction, Fraction]],
    depth: int,
    *,
    enforce_transport: bool = False,
) -> WordCollisionAdmission:
    """Admit native and catalog calls before product or exact composition."""

    if not isinstance(generators, Sequence) or not generators:
        raise OperationDomainValidationError(
            location=("generators",),
            code="affine_map.invalid_generators",
            message="generators must be a nonempty sequence of affine maps",
        )
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
        raise OperationDomainValidationError(
            location=("depth",),
            code="affine_map.invalid_depth",
            message=(f"depth must be an integer between 1 and {MAX_DEPTH}"),
        )
    if len(generators) > MAX_GENERATORS:
        raise OperationDomainValidationError(
            location=("generators",),
            code="affine_map.too_many_generators",
            message=f"at most {MAX_GENERATORS} generators are supported",
        )
    normalized: list[tuple[Fraction, Fraction]] = []
    for index, generator in enumerate(generators):
        if not isinstance(generator, (tuple, list)) or len(generator) != 2:
            raise OperationDomainValidationError(
                location=("generators", index),
                code="affine_map.invalid_generator",
                message="each generator must contain a slope and intercept",
            )
        slope, intercept = generator
        if not isinstance(slope, Fraction) or not isinstance(intercept, Fraction):
            raise OperationDomainValidationError(
                location=("generators", index),
                code="affine_map.invalid_generator",
                message="slopes and intercepts must be exact Fractions",
            )
        normalized.append((slope, intercept))
    canonical_generators = tuple(normalized)
    word_count = len(canonical_generators) ** depth
    composition_work = word_count * depth
    if composition_work > MAX_COMPOSITION_WORK:
        raise OperationDomainValidationError(
            location=("depth",),
            code="affine_map.composition_work_exceeds_bound",
            message=(
                f"the complete word profile exceeds the {MAX_COMPOSITION_WORK:,}-step "
                "composition work limit"
            ),
        )

    slope_heights = tuple(_fraction_height(slope) for slope, _ in canonical_generators)
    intercept_heights = tuple(
        _fraction_height(intercept) for _, intercept in canonical_generators
    )
    if any(
        height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS)
        for height in (*slope_heights, *intercept_heights)
    ):
        raise OperationDomainValidationError(
            location=("generators",),
            code="affine_map.generator_height_exceeded",
            message="generator coefficients exceed the canonical rational digit limit",
        )
    coefficient_height = _coefficient_height(canonical_generators, depth)
    if coefficient_height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
        raise OperationDomainValidationError(
            location=("depth",),
            code="affine_map.rational_growth_exceeded",
            message="composed affine coefficients exceed the canonical rational digit limit",
        )

    generator_value_sizes = tuple(
        strict_json_object_size(
            (
                ("slope", _rational_size(_fraction_height(slope))),
                ("intercept", _rational_size(_fraction_height(intercept))),
            )
        )
        for slope, intercept in canonical_generators
    )
    generators_bytes = _array_size(generator_value_sizes)
    index_digits = len(str(len(canonical_generators) - 1))
    word_bytes = _array_size((len(encode_strict_json(0)),) * depth)
    word_bytes += depth * max(index_digits - 1, 0)
    total_words_bytes = 2 * word_count + word_count * word_bytes
    row_fixed_bytes = strict_json_object_size(
        (
            ("slope", _rational_size(coefficient_height)),
            ("intercept", _rational_size(coefficient_height)),
            ("multiplicity", len(encode_strict_json(word_count))),
            ("words", 0),
        )
    )
    distinct_generator_count = len(set(canonical_generators))
    possible_rows = min(
        word_count,
        distinct_generator_count
        if all(slope == 0 for slope, _ in canonical_generators)
        else distinct_generator_count**depth,
    )
    rows_bytes = _array_size((row_fixed_bytes,) * possible_rows) + total_words_bytes
    result_bytes = strict_json_object_size(
        (
            ("generators", generators_bytes),
            ("depth", len(encode_strict_json(depth))),
            ("rows", rows_bytes),
        )
    )
    if enforce_transport and result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("generators",),
            code="affine_map.result_bytes_exceeded",
            message="word collision profile exceeds the canonical output-byte limit",
        )
    return WordCollisionAdmission(
        generators=canonical_generators,
        word_count=word_count,
        coefficient_height=coefficient_height,
        result_bytes=result_bytes,
    )


def compute_word_collision_profile(
    generators: Sequence[tuple[Fraction, Fraction]],
    depth: int,
    *,
    enforce_transport: bool = False,
) -> WordCollisionProfileResult:
    """Compute the complete word collision profile of an affine-map family.

    For each generator word of length ``depth``, compose the corresponding
    affine maps and group words by their exact composed map.
    Convention: word (i_1,...,i_d) represents f_{i_d} o ... o f_{i_1}.
    """
    admission = _admit_word_collision_profile(
        generators, depth, enforce_transport=enforce_transport
    )
    generators = admission.generators

    from jacobian.math.algebra.affine_map_word_collision._models import AffineMapSpec

    gen_specs = [
        AffineMapSpec._from_kernel(slope, intercept)
        for slope, intercept in generators
    ]

    class_to_words: dict[tuple[Fraction, Fraction], list[tuple[int, ...]]] = {}

    for word in product(range(len(generators)), repeat=depth):
        a, b = _compose_word(generators, word)
        key = (a, b)
        if key not in class_to_words:
            class_to_words[key] = []
        class_to_words[key].append(word)

    rows: list[CollisionRow] = []
    for (slope, intercept), words in sorted(
        class_to_words.items(),
        key=lambda kv: (kv[0][0], kv[0][1]),
    ):
        rows.append(
            CollisionRow._from_kernel(
                slope, intercept, len(words), tuple(sorted(words))
            )
        )

    return WordCollisionProfileResult._from_kernel(
        generators=tuple(gen_specs),
        depth=depth,
        rows=tuple(rows),
    )


def _compose_word(
    generators: tuple[tuple[Fraction, Fraction], ...],
    word: tuple[int, ...],
) -> tuple[Fraction, Fraction]:
    """Compose affine maps according to the convention f_{i_d} o ... o f_{i_1}.

    Each generator is x -> a*x + b. Composition (A,B) o (C,D) = (A*C, A*D+B).
    """
    a = Fraction(1)
    b = Fraction(0)
    for idx in word:
        ga, gb = generators[idx]
        a, b = ga * a, ga * b + gb
    return a, b
