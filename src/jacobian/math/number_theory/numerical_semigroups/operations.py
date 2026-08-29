"""Native numerical-semigroup operations on ordinary Python values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import gcd

from jacobian._exact import format_canonical_rational
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.numerical_semigroups._algorithms import (
    apery_set,
    belongs,
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorization_count,
    factorization_length_extrema,
    factorization_lengths,
    factorization_predecessors,
    factorizations,
    minimal_generating_system,
    reconstruct_factorization,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeResult,
    ElementDeltaSetResult,
    ElementElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._factorization_models import (
    FactorizationComputeResult,
    FactorizationDistanceResult,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeResult,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_models import (
    BettiCatenaryDegree,
    BettiElementsResult,
    CatenaryDegreeResult,
    DeltaSetResult,
    ElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._models import (
    MAX_ELEMENT,
    MAX_GLOBAL_BETTI_ELEMENT,
    MAX_GLOBAL_DELTA_CHECK,
    MAX_GRAPH_FACTORIZATIONS,
    MAX_MATERIALIZED_FACTORIZATIONS,
)
from jacobian.math.number_theory.numerical_semigroups._presentation_models import (
    MinimalPresentationRelation,
    MinimalPresentationResult,
    PresentationBinomial,
    PresentationBinomialsResult,
)
from jacobian.math.number_theory.numerical_semigroups._summary_models import (
    NumericalSemigroupSummaryResult,
    SemigroupMembershipResult,
)


@dataclass(frozen=True)
class FactorizationGraph:
    """The edges and connected components of a factorization graph."""

    edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]


def _generators(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("generators must be positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("generators must be strictly increasing")
    if gcd(*values) != 1:
        raise ValueError("generators must have gcd 1")
    if minimal_generating_system(values) != values:
        raise ValueError("generators must be a minimal generating system")
    return values


def factorization_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    """Return the standard distance between equal-degree factorizations."""
    if len(first) != len(second) or not first:
        raise ValueError("factorizations must have the same positive dimension")
    if any(type(value) is not int or value < 0 for value in (*first, *second)):
        raise ValueError("factorization coordinates must be nonnegative integers")
    common_length = sum(
        min(left, right) for left, right in zip(first, second, strict=True)
    )
    return max(sum(first) - common_length, sum(second) - common_length)


def factorization_graph(family: tuple[tuple[int, ...], ...]) -> FactorizationGraph:
    """Build the graph joining factorizations that share a generator."""
    if not family:
        return FactorizationGraph(edges=(), components=())
    dimension = len(family[0])
    if not dimension or any(len(item) != dimension for item in family):
        raise ValueError("factorizations must have one common positive dimension")
    if any(type(value) is not int or value < 0 for item in family for value in item):
        raise ValueError("factorization coordinates must be nonnegative integers")
    edges = tuple(
        (left, right)
        for left in range(len(family))
        for right in range(left + 1, len(family))
        if any(min(a, b) > 0 for a, b in zip(family[left], family[right], strict=True))
    )
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(family))}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(family)):
        if start in seen:
            continue
        pending = [start]
        seen.add(start)
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return FactorizationGraph(edges=edges, components=tuple(components))


def element_delta_set(generators: tuple[int, ...], value: int) -> tuple[int, ...]:
    """Return the successive factorization-length gaps of one element."""
    lengths = factorization_lengths(_generators(generators), value)
    return tuple(sorted({right - left for left, right in pairwise(lengths)}))


def element_elasticity(generators: tuple[int, ...], value: int) -> Fraction:
    """Return the exact elasticity of one nonzero semigroup element."""
    lengths = factorization_lengths(_generators(generators), value)
    if not lengths:
        raise ValueError("value is not in the numerical semigroup")
    if lengths[0] == 0:
        raise ValueError("elasticity is undefined for zero")
    return Fraction(lengths[-1], lengths[0])


def element_catenary_degree(generators: tuple[int, ...], value: int) -> int:
    """Return the exact catenary degree of one semigroup element."""
    family = factorizations(_generators(generators), value)
    if not family:
        raise ValueError("value is not in the numerical semigroup")
    return catenary_degree_from_factorizations(family)


def elasticity(generators: tuple[int, ...]) -> Fraction:
    """Return the exact global elasticity of a numerical semigroup."""
    values = _generators(generators)
    return Fraction(values[-1], values[0])


def _bounded_value(generators: tuple[int, ...], value: int) -> int:
    if type(value) is not int:
        raise ValueError("value must be an integer")
    if generators != (1,) and value > MAX_ELEMENT:
        raise ValueError(f"value must be at most {MAX_ELEMENT}")
    return value


def _require_materializable(
    generators: tuple[int, ...], value: int, maximum: int
) -> None:
    if value >= 0 and (count := factorization_count(generators, value)) > maximum:
        raise ValueError(
            f"factorization family has {count} members, exceeding the exact materialization bound {maximum}"
        )


def _bounded_betti_data(
    generators: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, tuple[tuple[int, ...], ...]]]:
    apery, candidates, disconnected = betti_data(generators)
    if generators == (1,):
        return apery, candidates, disconnected
    maximum = max(apery[1:]) + generators[-1]
    if maximum > MAX_GLOBAL_BETTI_ELEMENT:
        raise ValueError(
            "complete Apéry candidate range ends at "
            f"{maximum}, exceeding the global invariant bound {MAX_GLOBAL_BETTI_ELEMENT}"
        )
    return apery, candidates, disconnected


def summary(generators: tuple[int, ...]) -> NumericalSemigroupSummaryResult:
    """Return the exact summary of a canonical numerical semigroup."""
    values = _generators(generators)
    multiplicity = values[0]
    if multiplicity == 1:
        return NumericalSemigroupSummaryResult._from_kernel(
            minimal_generators=("1",),
            multiplicity="1",
            embedding_dimension=1,
            frobenius_number="-1",
            conductor="0",
            genus=0,
            gaps=(),
        )

    limit = (multiplicity - 1) * max(values)
    in_semigroup = [False] * (limit + 1)
    in_semigroup[0] = True
    run = 0
    conductor = limit + 1
    for value in range(1, limit + 1):
        in_semigroup[value] = any(
            value >= generator and in_semigroup[value - generator]
            for generator in values
        )
        if in_semigroup[value]:
            run += 1
            if run == multiplicity:
                conductor = value - multiplicity + 1
                break
        else:
            run = 0

    gaps = [
        value
        for value in range(1, conductor)
        if value <= limit and not in_semigroup[value]
    ]
    frobenius = max(gaps) if gaps else -1
    return NumericalSemigroupSummaryResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(value) for value in values),
        multiplicity=format_canonical_integer(multiplicity),
        embedding_dimension=len(values),
        frobenius_number=format_canonical_integer(frobenius),
        conductor=format_canonical_integer(conductor),
        genus=len(gaps),
        gaps=tuple(format_canonical_integer(gap) for gap in gaps),
    )


def membership(generators: tuple[int, ...], value: int) -> SemigroupMembershipResult:
    """Return whether a canonical integer belongs to a numerical semigroup."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    return SemigroupMembershipResult(
        value=format_canonical_integer(target),
        in_semigroup=belongs(target, apery_set(values)),
    )


def factorization_profile(
    generators: tuple[int, ...], value: int
) -> FactorizationComputeResult:
    """Return the complete factorization family of a canonical element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    _require_materializable(values, target, MAX_MATERIALIZED_FACTORIZATIONS)
    family = factorizations(values, target)
    return FactorizationComputeResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        in_semigroup=bool(family),
        factorizations=family,
    )


def factorization_lengths_profile(
    generators: tuple[int, ...], value: int
) -> FactorizationLengthsComputeResult:
    """Return the complete factorization-length set of a canonical element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    lengths = factorization_lengths(values, target)
    return FactorizationLengthsComputeResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        in_semigroup=bool(lengths),
        lengths=lengths,
    )


def factorization_distance_profile(
    generators: tuple[int, ...],
    value: int,
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> FactorizationDistanceResult:
    """Return the distance between two factorizations of one element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    if len(first) != len(values) or len(second) != len(values):
        raise ValueError(
            "factorization coordinates must match the minimal generating system"
        )
    if any(
        type(coordinate) is not int or coordinate < 0
        for coordinate in (*first, *second)
    ):
        raise ValueError("factorization coordinates must be non-negative")
    if any(
        sum(
            coordinate * generator
            for coordinate, generator in zip(item, values, strict=True)
        )
        != target
        for item in (first, second)
    ):
        raise ValueError("both factorizations must evaluate to the declared value")
    return FactorizationDistanceResult(
        value=format_canonical_integer(target),
        distance=factorization_distance(first, second),
        first_length=sum(first),
        second_length=sum(second),
    )


def factorization_graph_profile(
    generators: tuple[int, ...], value: int
) -> FactorizationGraphComputeResult:
    """Return the exact shared-support graph of one canonical element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    _require_materializable(values, target, MAX_GRAPH_FACTORIZATIONS)
    family = factorizations(values, target)
    graph = factorization_graph(family)
    return FactorizationGraphComputeResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        in_semigroup=bool(family),
        factorizations=family,
        edges=graph.edges,
        connected_components=graph.components,
        is_connected=len(graph.components) <= 1,
    )


def element_delta_set_profile(
    generators: tuple[int, ...], value: int
) -> ElementDeltaSetResult:
    """Return the complete factorization-length delta set of one element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    if not belongs(target, apery_set(values)):
        raise ValueError("value must belong to the numerical semigroup")
    lengths = factorization_lengths(values, target)
    return ElementDeltaSetResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        factorization_lengths=lengths,
        delta_set=tuple(sorted({right - left for left, right in pairwise(lengths)})),
    )


def element_elasticity_profile(
    generators: tuple[int, ...], value: int
) -> ElementElasticityResult:
    """Return exact factorization-length elasticity of one element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    if target <= 0:
        raise ValueError("elasticity is defined here only for positive elements")
    if not belongs(target, apery_set(values)):
        raise ValueError("value must belong to the numerical semigroup")
    minimum, maximum = factorization_length_extrema(values, target)
    return ElementElasticityResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        minimum_length=minimum,
        maximum_length=maximum,
        elasticity=format_canonical_rational(Fraction(maximum, minimum)),
    )


def element_catenary_degree_profile(
    generators: tuple[int, ...], value: int
) -> ElementCatenaryDegreeResult:
    """Return the catenary degree of one canonical semigroup element."""
    values = _generators(generators)
    target = _bounded_value(values, value)
    if not belongs(target, apery_set(values)):
        raise ValueError("value must belong to the numerical semigroup")
    _require_materializable(values, target, MAX_GRAPH_FACTORIZATIONS)
    family = factorizations(values, target)
    return ElementCatenaryDegreeResult._from_kernel(
        value=format_canonical_integer(target),
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        factorization_count=len(family),
        catenary_degree=catenary_degree_from_factorizations(family),
    )


def betti_elements(generators: tuple[int, ...]) -> BettiElementsResult:
    """Return the complete Betti-element profile of a canonical semigroup."""
    values = _generators(generators)
    apery, candidates, disconnected = _bounded_betti_data(values)
    return BettiElementsResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        apery_set=tuple(format_canonical_integer(item) for item in apery),
        candidate_count=len(candidates),
        betti_elements=tuple(format_canonical_integer(item) for item in disconnected),
    )


def delta_set(generators: tuple[int, ...]) -> DeltaSetResult:
    """Return the complete global delta set of a canonical semigroup."""
    values = _generators(generators)
    periodicity_bound = delta_periodicity_bound(values)
    checked_through = periodicity_bound + values[-1] - 1
    if checked_through > MAX_GLOBAL_DELTA_CHECK:
        raise ValueError(
            "complete delta-set check requires elements through "
            f"{checked_through}, exceeding the bound {MAX_GLOBAL_DELTA_CHECK}"
        )
    all_deltas: set[int] = set()
    length_sets: list[set[int]] = [set() for _ in range(values[-1])]
    length_sets[0].add(0)
    for target in range(1, checked_through + 1):
        lengths: set[int] = set()
        for atom in values:
            if target >= atom:
                lengths.update(
                    length + 1 for length in length_sets[(target - atom) % values[-1]]
                )
        ordered = sorted(lengths)
        all_deltas.update(right - left for left, right in pairwise(ordered))
        length_sets[target % values[-1]] = lengths
    return DeltaSetResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        delta_set=tuple(sorted(all_deltas)),
        periodicity_bound=periodicity_bound,
        checked_through=checked_through,
    )


def global_elasticity(generators: tuple[int, ...]) -> ElasticityResult:
    """Return exact global elasticity of a canonical semigroup."""
    values = _generators(generators)
    return ElasticityResult(
        elasticity=format_canonical_rational(Fraction(values[-1], values[0])),
        smallest_generator=format_canonical_integer(values[0]),
        largest_generator=format_canonical_integer(values[-1]),
    )


def global_catenary_degree(
    generators: tuple[int, ...],
) -> CatenaryDegreeResult:
    """Return the global catenary degree and Betti witnesses."""
    values = _generators(generators)
    _, _, disconnected = _bounded_betti_data(values)
    for target in disconnected:
        _require_materializable(values, target, MAX_GRAPH_FACTORIZATIONS)
    degrees = tuple(
        BettiCatenaryDegree(
            betti_element=format_canonical_integer(target),
            catenary_degree=catenary_degree_from_factorizations(
                factorizations(values, target)
            ),
        )
        for target in disconnected
    )
    maximum = max((record.catenary_degree for record in degrees), default=0)
    return CatenaryDegreeResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        catenary_degree=maximum,
        betti_degrees=degrees,
        witness_betti_elements=tuple(
            record.betti_element
            for record in degrees
            if record.catenary_degree == maximum and maximum > 0
        ),
    )


def minimal_presentation(
    generators: tuple[int, ...],
) -> MinimalPresentationResult:
    """Return one exact minimal presentation of a canonical semigroup."""
    values = _generators(generators)
    _, _, disconnected = _bounded_betti_data(values)
    predecessors = factorization_predecessors(values, max(disconnected, default=0))
    relations: list[MinimalPresentationRelation] = []
    for betti_value, components in disconnected.items():
        representatives: list[tuple[int, ...]] = []
        for component in components:
            generator_index = component[0]
            residual = reconstruct_factorization(
                values, predecessors, betti_value - values[generator_index]
            )
            if residual is None:
                raise RuntimeError("Betti component has no factorization witness")
            coordinates = list(residual)
            coordinates[generator_index] += 1
            representatives.append(tuple(coordinates))
        for target_representative in representatives[1:]:
            relations.append(
                MinimalPresentationRelation(
                    first=representatives[0], second=target_representative
                )
            )
    return MinimalPresentationResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        betti_elements=tuple(format_canonical_integer(item) for item in disconnected),
        relations=tuple(relations),
    )


def presentation_binomials(
    generators: tuple[int, ...],
    relations: tuple[MinimalPresentationRelation, ...],
) -> PresentationBinomialsResult:
    """Convert homogeneous relations on a canonical axis to binomials."""
    values = _generators(generators)
    for relation in relations:
        if len(relation.first) != len(values) or len(relation.second) != len(values):
            raise ValueError(
                "relation coordinates must match the minimal generating system"
            )
        first_degree = sum(
            coordinate * generator
            for coordinate, generator in zip(relation.first, values, strict=True)
        )
        second_degree = sum(
            coordinate * generator
            for coordinate, generator in zip(relation.second, values, strict=True)
        )
        if first_degree != second_degree:
            raise ValueError(
                "relation factorizations must have the same semigroup degree"
            )
    return PresentationBinomialsResult(
        minimal_generators=tuple(format_canonical_integer(item) for item in values),
        binomials=tuple(
            PresentationBinomial(
                left_exponents=tuple(relation.first),
                right_exponents=tuple(relation.second),
            )
            for relation in relations
        ),
    )


__all__ = [
    "FactorizationGraph",
    "apery_set",
    "belongs",
    "betti_elements",
    "delta_set",
    "elasticity",
    "element_catenary_degree",
    "element_catenary_degree_profile",
    "element_delta_set",
    "element_delta_set_profile",
    "element_elasticity",
    "element_elasticity_profile",
    "factorization_count",
    "factorization_distance",
    "factorization_distance_profile",
    "factorization_graph",
    "factorization_graph_profile",
    "factorization_length_extrema",
    "factorization_lengths",
    "factorization_lengths_profile",
    "factorization_profile",
    "factorizations",
    "global_catenary_degree",
    "global_elasticity",
    "membership",
    "minimal_generating_system",
    "minimal_presentation",
    "presentation_binomials",
    "summary",
]
