"""Domain-owned combinatorics on words operations."""

from __future__ import annotations

import math

from jacobian.math.words._models import (
    ConjugatesRequest,
    ConjugatesResult,
    FactorOccurrencesRequest,
    FactorOccurrencesResult,
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    MorphismApplyRequest,
    MorphismApplyResult,
    MorphismComposeRequest,
    MorphismComposeResult,
    ParikhRequest,
    ParikhResult,
    PeriodsRequest,
    PeriodsResult,
    PrefixFunctionRequest,
    PrefixFunctionResult,
    PrimitiveRootRequest,
    PrimitiveRootResult,
)


def compute_factors_length(request: FactorsLengthRequest) -> FactorsLengthResult:
    """Find all distinct factors of a given length in a word."""
    word = request.word
    n = request.factor_length
    word_len = len(word)

    if n > word_len:
        return FactorsLengthResult(
            factor_length=n,
            factors=(),
            occurrences=(),
            multiplicities=(),
            first_occurrence=(),
            distinct_count=0,
        )

    factor_map: dict[tuple[str, ...], list[int]] = {}
    for i in range(word_len - n + 1):
        factor = tuple(word[i : i + n])
        if factor not in factor_map:
            factor_map[factor] = []
        factor_map[factor].append(i)

    factors = tuple(factor_map.keys())
    occurrences = tuple(tuple(factor_map[f]) for f in factors)
    multiplicities = tuple(len(o) for o in occurrences)
    first_occurrence = tuple(o[0] for o in occurrences)

    return FactorsLengthResult(
        factor_length=n,
        factors=factors,
        occurrences=occurrences,
        multiplicities=multiplicities,
        first_occurrence=first_occurrence,
        distinct_count=len(factors),
    )


def compute_factor_occurrences(
    request: FactorOccurrencesRequest,
) -> FactorOccurrencesResult:
    """Find all occurrences of a pattern in a word (including overlaps)."""
    word = request.word
    pattern = request.pattern
    word_len = len(word)
    pat_len = len(pattern)

    if pat_len == 0:
        return FactorOccurrencesResult(
            pattern=pattern,
            occurrences=tuple(range(word_len + 1)),
            count=word_len + 1,
        )

    if pat_len > word_len:
        return FactorOccurrencesResult(
            pattern=pattern, occurrences=(), count=0
        )

    occurrences: list[int] = []
    for i in range(word_len - pat_len + 1):
        if word[i : i + pat_len] == pattern:
            occurrences.append(i)

    return FactorOccurrencesResult(
        pattern=pattern,
        occurrences=tuple(occurrences),
        count=len(occurrences),
    )


def compute_periods(request: PeriodsRequest) -> PeriodsResult:
    """Compute all periods of a finite word."""
    word = request.word
    n = len(word)
    if n == 0:
        return PeriodsResult(periods=(), least_period=0, is_primitive=True)

    periods: list[int] = []
    for p in range(1, n):
        if all(word[i] == word[i + p] for i in range(n - p)):
            periods.append(p)
    # n is always a period
    periods.append(n)

    least = periods[0]
    is_primitive = least == n

    return PeriodsResult(
        periods=tuple(periods),
        least_period=least,
        is_primitive=is_primitive,
    )


def compute_primitive_root(request: PrimitiveRootRequest) -> PrimitiveRootResult:
    """Compute the primitive root of a finite word."""
    word = request.word
    n = len(word)
    if n == 0:
        return PrimitiveRootResult(root=(), exponent=1)

    for k in range(1, n + 1):
        if n % k != 0:
            continue
        root = word[:k]
        if all(word[i] == root[i % k] for i in range(n)):
            return PrimitiveRootResult(root=tuple(root), exponent=n // k)

    # Should never reach here
    return PrimitiveRootResult(root=tuple(word), exponent=1)


def compute_conjugates(request: ConjugatesRequest) -> ConjugatesResult:
    """Compute all cyclic conjugates of a finite word."""
    word = request.word
    n = len(word)

    if n == 0:
        return ConjugatesResult(
            conjugates=((),),
            least_lexicographic=(),
            rotation_index=(0,),
        )

    conjugates: list[tuple[str, ...]] = []
    for i in range(n):
        conj = tuple(word[i:] + word[:i])
        conjugates.append(conj)

    least = min(conjugates)
    least_idx = conjugates.index(least)

    # Find rotation indices
    rotation_index = tuple(
        i for i, c in enumerate(conjugates) if c == least
    )

    return ConjugatesResult(
        conjugates=tuple(conjugates),
        least_lexicographic=least,
        rotation_index=rotation_index,
    )


def compute_parikh_vector(request: ParikhRequest) -> ParikhResult:
    """Compute the Parikh vector of a finite word."""
    alphabet = request.alphabet
    word = request.word

    counts = {letter: 0 for letter in alphabet}
    for letter in word:
        counts[letter] += 1

    parikh = tuple(counts[letter] for letter in alphabet)
    support = tuple(letter for letter in alphabet if counts[letter] > 0)

    return ParikhResult(
        parikh_vector=parikh,
        length=len(word),
        support=support,
    )


def compute_prefix_function(request: PrefixFunctionRequest) -> PrefixFunctionResult:
    """Compute the Knuth-Morris-Pratt prefix function (border table)."""
    word = request.word
    n = len(word)

    if n == 0:
        return PrefixFunctionResult(
            prefix_function=(),
            border_lengths=(),
        )

    pi: list[int] = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and word[i] != word[j]:
            j = pi[j - 1]
        if word[i] == word[j]:
            j += 1
        pi[i] = j

    # border_lengths: the border length of each prefix
    border_lengths = tuple(pi)

    return PrefixFunctionResult(
        prefix_function=tuple(pi),
        border_lengths=border_lengths,
    )


def apply_morphism(request: MorphismApplyRequest) -> MorphismApplyResult:
    """Apply a word morphism to a finite word."""
    source_alphabet = request.source_alphabet
    image_map = {
        source_alphabet[i]: request.images[i] for i in range(len(source_alphabet))
    }

    result: list[str] = []
    for letter in request.word:
        result.extend(image_map[letter])

    return MorphismApplyResult(
        image=tuple(result),
        length=len(result),
    )


def compose_morphisms(request: MorphismComposeRequest) -> MorphismComposeResult:
    """Compose two word morphisms: tau ∘ sigma."""
    source_alphabet = request.source_alphabet
    middle_alphabet = request.middle_alphabet
    sigma_images = request.sigma_images
    tau_images = request.tau_images

    tau_map = {
        middle_alphabet[i]: tau_images[i] for i in range(len(middle_alphabet))
    }

    composed: list[tuple[str, ...]] = []
    for i, _ in enumerate(source_alphabet):
        sigma_image = sigma_images[i]
        result: list[str] = []
        for letter in sigma_image:
            result.extend(tau_map[letter])
        composed.append(tuple(result))

    return MorphismComposeResult(images=tuple(composed))


def compute_incidence_matrix(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    """Compute the incidence matrix of a word morphism."""
    source_alphabet = request.source_alphabet
    target_alphabet = request.target_alphabet
    images = request.images

    n_src = len(source_alphabet)
    n_tgt = len(target_alphabet)

    matrix: list[list[int]] = [
        [0] * n_src for _ in range(n_tgt)
    ]

    for j, image in enumerate(images):
        for letter in image:
            i = target_alphabet.index(letter)
            matrix[i][j] += 1

    return IncidenceMatrixResult(
        matrix=tuple(tuple(row) for row in matrix),
        source_alphabet=source_alphabet,
        target_alphabet=target_alphabet,
    )


__all__ = [
    "apply_morphism",
    "compose_morphisms",
    "compute_conjugates",
    "compute_factor_occurrences",
    "compute_factors_length",
    "compute_incidence_matrix",
    "compute_parikh_vector",
    "compute_periods",
    "compute_prefix_function",
    "compute_primitive_root",
]
