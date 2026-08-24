"""Exact bounded native kernels for combinatorics on words."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from jacobian.math.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    FiniteWord,
    ProlongableSubstitution,
    Substitution,
    SubstitutionDependencyEdge,
    SubstitutionDependencyGraph,
    WordMorphism,
    _require_dependency_occurrence_bound,
    _require_prolongable_source_occurrence_bound,
)

MAX_FIXED_POINT_GENERATION_WORK = 1_000_000
MAX_FIXED_POINT_RESULT_BYTES = 512_000


@dataclass(frozen=True, slots=True)
class FactorAnalysis:
    factor_length: int
    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PeriodAnalysis:
    periods: tuple[int, ...]
    least_period: int
    primitive: bool


@dataclass(frozen=True, slots=True)
class FixedPointPrefixAnalysis:
    prefix: FiniteWord
    least_iterate_depth: int
    retained_prefix_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrimitivityAnalysis:
    strongly_connected_components: tuple[tuple[str, ...], ...]
    irreducible: bool
    aperiodic: bool | None
    primitive: bool
    least_positive_power: int | None
    exponent_upper_bound: int
    obstruction: Literal[
        "NONE", "REDUCIBLE_DEPENDENCY_GRAPH", "PERIODIC_DEPENDENCY_GRAPH"
    ]


def factors_of_length(word: FiniteWord, factor_length: int) -> FactorAnalysis:
    if not 0 <= factor_length <= len(word.letters):
        raise ValueError("factor length must be between zero and the word length")
    positions: dict[tuple[str, ...], list[int]] = {}
    for start in range(len(word.letters) - factor_length + 1):
        factor = word.letters[start : start + factor_length]
        positions.setdefault(factor, []).append(start)
    factors = tuple(positions)
    return FactorAnalysis(
        factor_length=factor_length,
        factors=factors,
        occurrences=tuple(tuple(positions[factor]) for factor in factors),
    )


def factor_occurrences(word: FiniteWord, pattern: tuple[str, ...]) -> tuple[int, ...]:
    if any(letter not in word.alphabet for letter in pattern):
        raise ValueError("pattern letter is outside the declared alphabet")
    if not pattern:
        return tuple(range(len(word.letters) + 1))
    return tuple(
        start
        for start in range(len(word.letters) - len(pattern) + 1)
        if word.letters[start : start + len(pattern)] == pattern
    )


def periods(word: FiniteWord) -> PeriodAnalysis:
    length = len(word.letters)
    if length == 0:
        return PeriodAnalysis(periods=(), least_period=0, primitive=False)
    values = tuple(
        period
        for period in range(1, length + 1)
        if all(
            word.letters[index] == word.letters[index + period]
            for index in range(length - period)
        )
    )
    _, exponent = primitive_root(word)
    return PeriodAnalysis(
        periods=values,
        least_period=values[0],
        primitive=exponent == 1,
    )


def primitive_root(word: FiniteWord) -> tuple[tuple[str, ...], int]:
    length = len(word.letters)
    if length == 0:
        return ((), 1)
    for root_length in range(1, length + 1):
        if length % root_length == 0:
            root = word.letters[:root_length]
            if root * (length // root_length) == word.letters:
                return (root, length // root_length)
    raise RuntimeError("finite word did not admit itself as a primitive root")


def conjugates(word: FiniteWord) -> tuple[tuple[str, ...], ...]:
    if not word.letters:
        return ((),)
    rotations = {
        word.letters[index:] + word.letters[:index]
        for index in range(len(word.letters))
    }
    rank = {symbol: index for index, symbol in enumerate(word.alphabet)}
    return tuple(
        sorted(rotations, key=lambda value: tuple(rank[item] for item in value))
    )


def parikh_vector(word: FiniteWord) -> tuple[int, ...]:
    return tuple(word.letters.count(symbol) for symbol in word.alphabet)


def prefix_function(word: FiniteWord) -> tuple[int, ...]:
    result = [0] * len(word.letters)
    for index in range(1, len(word.letters)):
        border = result[index - 1]
        while border and word.letters[index] != word.letters[border]:
            border = result[border - 1]
        if word.letters[index] == word.letters[border]:
            border += 1
        result[index] = border
    return tuple(result)


def apply_morphism(morphism: WordMorphism, word: FiniteWord) -> FiniteWord:
    if word.alphabet != morphism.source_alphabet:
        raise ValueError("word alphabet must equal the morphism source alphabet")
    image_map = dict(zip(morphism.source_alphabet, morphism.images, strict=True))
    output_length = sum(len(image_map[letter]) for letter in word.letters)
    if output_length > MAX_MORPHISM_OUTPUT_LENGTH:
        raise ValueError("morphism output exceeds the length bound")
    letters = tuple(output for letter in word.letters for output in image_map[letter])
    return FiniteWord(alphabet=morphism.target_alphabet, letters=letters)


def compose_morphisms(first: WordMorphism, second: WordMorphism) -> WordMorphism:
    if first.target_alphabet != second.source_alphabet:
        raise ValueError("first target alphabet must equal second source alphabet")
    second_map = dict(zip(second.source_alphabet, second.images, strict=True))
    images = tuple(
        tuple(output for letter in image for output in second_map[letter])
        for image in first.images
    )
    if any(len(image) > MAX_MORPHISM_OUTPUT_LENGTH for image in images):
        raise ValueError("composed morphism image exceeds the length bound")
    return WordMorphism(
        source_alphabet=first.source_alphabet,
        target_alphabet=second.target_alphabet,
        images=images,
    )


def incidence_matrix(morphism: WordMorphism) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(image.count(target) for image in morphism.images)
        for target in morphism.target_alphabet
    )


def substitution_dependency_graph(
    substitution: Substitution,
) -> SubstitutionDependencyGraph:
    """Return every nonzero letter dependency in alphabet row order."""

    _require_dependency_occurrence_bound(substitution)
    morphism = substitution.morphism
    edges = tuple(
        SubstitutionDependencyEdge(
            source=source,
            target=target,
            multiplicity=image.count(target),
            positions=tuple(
                position for position, letter in enumerate(image) if letter == target
            ),
        )
        for source, image in zip(morphism.source_alphabet, morphism.images, strict=True)
        for target in morphism.target_alphabet
        if target in image
    )
    return SubstitutionDependencyGraph(substitution=substitution, edges=edges)


def fixed_point_prefix(
    source: ProlongableSubstitution, prefix_length: int
) -> FixedPointPrefixAnalysis:
    """Return the requested prefix from the least sufficient seed iterate."""

    _require_fixed_point_prefix_budget(source, prefix_length)
    morphism = source.substitution.morphism
    image_map = dict(zip(morphism.source_alphabet, morphism.images, strict=True))
    current: tuple[str, ...] = (source.seed,)
    retained_prefix_lengths = [min(1, prefix_length)]
    depth = 0
    while len(current) < prefix_length:
        next_prefix: list[str] = []
        for letter in current:
            for output in image_map[letter]:
                next_prefix.append(output)
                if len(next_prefix) == prefix_length:
                    break
            if len(next_prefix) == prefix_length:
                break
        current = tuple(next_prefix)
        depth += 1
        retained_prefix_lengths.append(len(current))
    return FixedPointPrefixAnalysis(
        prefix=FiniteWord(
            alphabet=morphism.target_alphabet,
            letters=current[:prefix_length],
        ),
        least_iterate_depth=depth,
        retained_prefix_lengths=tuple(retained_prefix_lengths),
    )


def _fixed_point_result_byte_bound(
    source: ProlongableSubstitution, prefix_length: int
) -> int:
    alphabet = source.substitution.morphism.target_alphabet
    encoded_symbols = tuple(
        len(json.dumps(symbol, ensure_ascii=True).encode("utf-8"))
        for symbol in alphabet
    )
    prefix_bytes = (
        128
        + sum(encoded_symbols)
        + len(encoded_symbols)
        + prefix_length * (max(encoded_symbols) + 1)
    )
    ledger_length = max(1, prefix_length)
    ledger_bytes = 128 + ledger_length * (len(str(max(1, prefix_length))) + 1)
    source_bytes = len(source.model_dump_json().encode("utf-8"))
    return 4_096 + source_bytes + prefix_bytes + ledger_bytes


def _require_fixed_point_prefix_budget(
    source: ProlongableSubstitution, prefix_length: int
) -> None:
    if not 0 <= prefix_length <= MAX_MORPHISM_OUTPUT_LENGTH:
        raise ValueError(f"prefix length must be in 0..{MAX_MORPHISM_OUTPUT_LENGTH}")
    _require_prolongable_source_occurrence_bound(source)
    # One capped generation inspects/appends fewer than 2N cells, there are at
    # most N generations, and a public result replays the kernel once.
    generation_work = 4 * prefix_length * prefix_length
    if generation_work > MAX_FIXED_POINT_GENERATION_WORK:
        raise ValueError(
            "fixed-point generation exceeds the work bound "
            f"({generation_work} > {MAX_FIXED_POINT_GENERATION_WORK})"
        )
    result_bytes = _fixed_point_result_byte_bound(source, prefix_length)
    if result_bytes > MAX_FIXED_POINT_RESULT_BYTES:
        raise ValueError(
            "fixed-point result exceeds the byte bound "
            f"({result_bytes} > {MAX_FIXED_POINT_RESULT_BYTES})"
        )


def _boolean_product(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    product: list[int] = []
    for left_row in left:
        row = 0
        remaining = left_row
        while remaining:
            bit = remaining & -remaining
            row |= right[bit.bit_length() - 1]
            remaining ^= bit
        product.append(row)
    return tuple(product)


def substitution_primitivity_profile(
    dependency_graph: SubstitutionDependencyGraph,
) -> PrimitivityAnalysis:
    """Decide primitivity by Boolean powers through the Wielandt bound."""

    _require_dependency_occurrence_bound(dependency_graph.substitution)
    import networkx as nx

    alphabet = dependency_graph.substitution.morphism.source_alphabet
    index = {symbol: position for position, symbol in enumerate(alphabet)}
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(alphabet)
    graph.add_edges_from((edge.source, edge.target) for edge in dependency_graph.edges)

    components = tuple(
        sorted(
            (
                tuple(sorted(component, key=index.__getitem__))
                for component in nx.strongly_connected_components(graph)
            ),
            key=lambda component: index[component[0]],
        )
    )
    irreducible = len(components) == 1
    aperiodic = nx.is_aperiodic(graph) if irreducible else None

    order = len(alphabet)
    exponent_upper_bound = 1 if order == 1 else (order - 1) ** 2 + 1
    adjacency_rows = [0] * order
    for edge in dependency_graph.edges:
        adjacency_rows[index[edge.source]] |= 1 << index[edge.target]
    adjacency = tuple(adjacency_rows)
    full_row = (1 << order) - 1
    power = adjacency
    least_positive_power = None
    for exponent in range(1, exponent_upper_bound + 1):
        if all(row == full_row for row in power):
            least_positive_power = exponent
            break
        power = _boolean_product(power, adjacency)

    primitive = least_positive_power is not None
    if primitive != (irreducible and aperiodic is True):
        raise RuntimeError("graph and Boolean-power primitivity criteria disagree")
    obstruction: Literal[
        "NONE", "REDUCIBLE_DEPENDENCY_GRAPH", "PERIODIC_DEPENDENCY_GRAPH"
    ]
    if primitive:
        obstruction = "NONE"
    elif not irreducible:
        obstruction = "REDUCIBLE_DEPENDENCY_GRAPH"
    else:
        obstruction = "PERIODIC_DEPENDENCY_GRAPH"

    return PrimitivityAnalysis(
        strongly_connected_components=components,
        irreducible=irreducible,
        aperiodic=aperiodic,
        primitive=primitive,
        least_positive_power=least_positive_power,
        exponent_upper_bound=exponent_upper_bound,
        obstruction=obstruction,
    )


__all__ = [
    "FactorAnalysis",
    "FixedPointPrefixAnalysis",
    "PeriodAnalysis",
    "PrimitivityAnalysis",
    "apply_morphism",
    "compose_morphisms",
    "conjugates",
    "factor_occurrences",
    "factors_of_length",
    "fixed_point_prefix",
    "incidence_matrix",
    "parikh_vector",
    "periods",
    "prefix_function",
    "primitive_root",
    "substitution_dependency_graph",
    "substitution_primitivity_profile",
]
