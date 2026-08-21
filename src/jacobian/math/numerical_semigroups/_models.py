"""Typed wire contracts for numerical semigroup operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    apery_set,
    belongs,
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorization_count,
    factorization_length_extrema,
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)

MAX_GENERATORS = 20
MAX_GENERATOR = 500
MAX_ELEMENT = 10_000
MAX_MATERIALIZED_FACTORIZATIONS = 20_000
MAX_GRAPH_FACTORIZATIONS = 1_000
MAX_GLOBAL_BETTI_ELEMENT = 100_000
MAX_GLOBAL_DELTA_CHECK = 20_000


def _require_positive_bounded_generators(generators: tuple[str, ...]) -> None:
    values: list[int] = []
    for generator in generators:
        value = parse_canonical_integer(generator)
        if value <= 0:
            raise ValueError("generators must be positive integers")
        if value > MAX_GENERATOR:
            raise ValueError(f"generators must be at most {MAX_GENERATOR}")
        values.append(value)
    gcd = values[0]
    for value in values[1:]:
        while value:
            gcd, value = value, gcd % value
    if gcd != 1:
        raise ValueError(f"generators must have gcd 1, got gcd {gcd}")


def _require_minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    _require_positive_bounded_generators(generators)
    values = tuple(parse_canonical_integer(value) for value in generators)
    if values != tuple(sorted(set(values))):
        raise ValueError("generators must be distinct and strictly increasing")
    minimal = minimal_generating_system(values)
    if values != minimal:
        raise ValueError(
            "generators must be the minimal generating system; "
            f"expected {list(minimal)}"
        )
    return values


def _require_bounded_value(value: str) -> int:
    parsed = parse_canonical_integer(value)
    if parsed > MAX_ELEMENT:
        raise ValueError(f"value must be at most {MAX_ELEMENT}")
    return parsed


def _require_member(generators: tuple[int, ...], value: int) -> None:
    if not belongs(value, apery_set(generators)):
        raise ValueError("value must belong to the numerical semigroup")


def _require_materializable_factorizations(
    generators: tuple[int, ...], value: int, maximum: int
) -> None:
    if value < 0:
        return
    count = factorization_count(generators, value)
    if count > maximum:
        raise ValueError(
            f"factorization family has {count} members, exceeding the exact "
            f"materialization bound {maximum}"
        )


def _require_global_betti_bound(generators: tuple[int, ...]) -> None:
    if generators == (1,):
        return
    apery = apery_set(generators)
    maximum_candidate = max(apery[1:]) + generators[-1]
    if maximum_candidate > MAX_GLOBAL_BETTI_ELEMENT:
        raise ValueError(
            "complete Apéry candidate range ends at "
            f"{maximum_candidate}, exceeding the global invariant bound "
            f"{MAX_GLOBAL_BETTI_ELEMENT}"
        )


def _require_global_catenary_bound(generators: tuple[int, ...]) -> None:
    _require_global_betti_bound(generators)
    _, _, disconnected = betti_data(generators)
    for betti_element in disconnected:
        _require_materializable_factorizations(
            generators, betti_element, MAX_GRAPH_FACTORIZATIONS
        )


def _require_global_delta_bound(generators: tuple[int, ...]) -> None:
    checked_through = delta_periodicity_bound(generators) + generators[-1] - 1
    if checked_through > MAX_GLOBAL_DELTA_CHECK:
        raise ValueError(
            f"complete delta-set check requires elements through {checked_through}, "
            f"exceeding the bound {MAX_GLOBAL_DELTA_CHECK}"
        )


def _betti_component_index(
    factorization: tuple[int, ...], components: tuple[tuple[int, ...], ...]
) -> int:
    support = {
        index for index, coordinate in enumerate(factorization) if coordinate > 0
    }
    matches = [
        index for index, component in enumerate(components) if support <= set(component)
    ]
    if len(matches) != 1:
        raise ValueError("relation factorization is not bound to one Betti component")
    return matches[0]


def _edges_span(component_count: int, edges: list[tuple[int, int]]) -> bool:
    reached = {0}
    while True:
        expanded = reached | {
            right if left in reached else left
            for left, right in edges
            if left in reached or right in reached
        }
        if expanded == reached:
            return reached == set(range(component_count))
        reached = expanded


def _require_exact_factorization_family(
    generators: tuple[int, ...],
    value: int,
    family: tuple[tuple[int, ...], ...],
) -> None:
    if len(set(family)) != len(family):
        raise ValueError("factorizations must be unique")
    for factorization in family:
        if len(factorization) != len(generators) or any(
            coordinate < 0 for coordinate in factorization
        ):
            raise ValueError("factorization has invalid coordinates")
        degree = sum(
            coordinate * generator
            for coordinate, generator in zip(factorization, generators, strict=True)
        )
        if degree != value:
            raise ValueError("factorization does not evaluate to the result value")
    if len(family) != factorization_count(generators, value):
        raise ValueError("factorizations do not form the complete family")


def _factorization_graph_data(
    family: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    edges = tuple(
        (left, right)
        for left in range(len(family))
        for right in range(left + 1, len(family))
        if any(
            a > 0 and b > 0 for a, b in zip(family[left], family[right], strict=True)
        )
    )
    unseen = set(range(len(family)))
    components: list[tuple[int, ...]] = []
    while unseen:
        reached = {min(unseen)}
        while True:
            expanded = reached | {
                right if left in reached else left
                for left, right in edges
                if left in reached or right in reached
            }
            if expanded == reached:
                break
            reached = expanded
        unseen.difference_update(reached)
        components.append(tuple(sorted(reached)))
    return edges, tuple(components)


class NumericalSemigroupRequest(StrictModel):
    """A numerical semigroup defined by a finite set of positive generators."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"membership value must be at most {MAX_ELEMENT}")
        return self


class SemigroupMembershipResult(StrictModel):
    """Whether the value is in the semigroup."""

    value: CanonicalInteger
    in_semigroup: bool


# ---------------------------------------------------------------------------
# Extended operations: factorization, elasticity, catenary degree, etc.
# ---------------------------------------------------------------------------


class FactorizationComputeRequest(StrictModel):
    """Compute the complete factorization family Z(s) for one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_materializable_factorizations(
            generators, value, MAX_MATERIALIZED_FACTORIZATIONS
        )
        return self


class FactorizationComputeResult(StrictModel):
    """Complete factorization family Z(s) for one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_exact_family(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.factorizations):
            raise ValueError("membership must agree with the factorization family")
        _require_exact_factorization_family(generators, value, self.factorizations)
        return self


class FactorizationLengthsComputeRequest(StrictModel):
    """Compute the complete sorted length set of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_minimal_generators(self.generators)
        _require_bounded_value(self.value)
        return self


class FactorizationLengthsComputeResult(StrictModel):
    """Sorted length set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    in_semigroup: bool
    lengths: tuple[int, ...]

    @model_validator(mode="after")
    def require_length_set(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.lengths):
            raise ValueError("membership must agree with the factorization lengths")
        if self.lengths != tuple(sorted(set(self.lengths))):
            raise ValueError("lengths must be strictly increasing and duplicate-free")
        if any(length < 0 for length in self.lengths):
            raise ValueError("factorization lengths must be non-negative")
        if self.lengths != factorization_lengths(generators, value):
            raise ValueError("lengths do not form the complete length set")
        return self


class FactorizationDistanceRequest(StrictModel):
    """Distance between two factorizations of the same element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger
    first: tuple[int, ...] = Field(min_length=1)
    second: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        if len(self.first) != len(generators) or len(self.second) != len(generators):
            raise ValueError(
                "factorization coordinates must match the minimal generating system"
            )
        if any(c < 0 for c in self.first) or any(c < 0 for c in self.second):
            raise ValueError("factorization coordinates must be non-negative")
        first_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.first, generators, strict=True)
        )
        second_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.second, generators, strict=True)
        )
        if first_value != value or second_value != value:
            raise ValueError("both factorizations must evaluate to the declared value")
        return self


class FactorizationDistanceResult(StrictModel):
    """Distance between two factorizations."""

    value: CanonicalInteger
    distance: int = Field(ge=0)
    first_length: int = Field(ge=0)
    second_length: int = Field(ge=0)


class FactorizationGraphComputeRequest(StrictModel):
    """Compute the standard factorization graph of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_materializable_factorizations(
            generators, value, MAX_GRAPH_FACTORIZATIONS
        )
        return self


class FactorizationGraphComputeResult(StrictModel):
    """Standard factorization graph with connected components."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    connected_components: tuple[tuple[int, ...], ...]
    is_connected: bool

    @model_validator(mode="after")
    def require_graph_partition(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        value = parse_canonical_integer(self.value)
        vertex_count = len(self.factorizations)
        if self.in_semigroup != bool(self.factorizations):
            raise ValueError("membership must agree with graph vertices")
        vertices = tuple(
            index for component in self.connected_components for index in component
        )
        if tuple(sorted(vertices)) != tuple(range(vertex_count)):
            raise ValueError("connected components must partition all graph vertices")
        if self.is_connected != (len(self.connected_components) <= 1):
            raise ValueError("is_connected must agree with connected components")
        for left, right in self.edges:
            if not 0 <= left < right < vertex_count:
                raise ValueError("graph edge has invalid vertex indices")
        _require_exact_factorization_family(generators, value, self.factorizations)
        expected_edges, expected_components = _factorization_graph_data(
            self.factorizations
        )
        if self.edges != expected_edges:
            raise ValueError("edges do not match shared-support adjacency")
        if self.connected_components != expected_components:
            raise ValueError("connected components do not match the graph")
        return self


class ElementDeltaSetRequest(StrictModel):
    """Delta set of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_semigroup_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_member(generators, value)
        return self


class ElementDeltaSetResult(StrictModel):
    """Delta set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    factorization_lengths: tuple[int, ...]
    delta_set: tuple[int, ...]

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        value = parse_canonical_integer(self.value)
        expected_lengths = factorization_lengths(generators, value)
        if self.factorization_lengths != expected_lengths:
            raise ValueError("factorization_lengths do not match the element")
        expected_delta = tuple(
            sorted({right - left for left, right in pairwise(expected_lengths)})
        )
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise ValueError("delta_set must be strictly increasing and duplicate-free")
        if any(delta <= 0 for delta in self.delta_set):
            raise ValueError("delta values must be positive")
        if self.delta_set != expected_delta:
            raise ValueError("delta_set does not match the complete length set")
        return self


class ElementElasticityRequest(StrictModel):
    """Elasticity of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Distinct, strictly increasing minimal generators, each at most "
            f"{MAX_GENERATOR}."
        ),
    )
    value: CanonicalInteger = Field(
        description=f"Positive semigroup element at most {MAX_ELEMENT}."
    )

    @model_validator(mode="after")
    def require_nonzero_semigroup_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        if value <= 0:
            raise ValueError("elasticity is defined here only for positive elements")
        _require_member(generators, value)
        return self


class ElementElasticityResult(StrictModel):
    """Elasticity of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    minimum_length: int = Field(ge=1)
    maximum_length: int = Field(ge=1)
    elasticity: str

    @model_validator(mode="after")
    def require_length_ratio(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        expected_extrema = factorization_length_extrema(
            generators, parse_canonical_integer(self.value)
        )
        if (self.minimum_length, self.maximum_length) != expected_extrema:
            raise ValueError("length extrema do not match the element")
        if Fraction(self.elasticity) != Fraction(
            self.maximum_length, self.minimum_length
        ):
            raise ValueError("elasticity does not match the length ratio")
        return self


class ElementCatenaryDegreeRequest(StrictModel):
    """Catenary degree of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_exact_bounded_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_member(generators, value)
        _require_materializable_factorizations(
            generators, value, MAX_GRAPH_FACTORIZATIONS
        )
        return self


class ElementCatenaryDegreeResult(StrictModel):
    """Catenary degree of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    factorization_count: int = Field(ge=1)
    catenary_degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_exact_degree(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        family = factorizations(generators, parse_canonical_integer(self.value))
        if self.factorization_count != len(family):
            raise ValueError("factorization_count does not match the element")
        if self.catenary_degree != catenary_degree_from_factorizations(family):
            raise ValueError("catenary_degree does not match the factorization graph")
        return self


class BettiElementsRequest(StrictModel):
    """Betti elements of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_complete_candidate_range(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_betti_bound(generators)
        return self


class BettiElementsResult(StrictModel):
    """Betti elements of a semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    apery_set: tuple[CanonicalInteger, ...]
    candidate_count: int = Field(ge=0)
    betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_complete_betti_data(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        apery, candidates, disconnected = betti_data(generators)
        if tuple(map(parse_canonical_integer, self.apery_set)) != apery:
            raise ValueError("apery_set does not match the minimal generators")
        if self.candidate_count != len(candidates):
            raise ValueError("candidate_count does not match the complete range")
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise ValueError("betti_elements do not match disconnected candidates")
        return self


class MinimalPresentationRequest(StrictModel):
    """One minimal presentation of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_complete_candidate_range(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_betti_bound(generators)
        return self


class MinimalPresentationRelation(StrictModel):
    """One relation (pair of distinct factorizations) in a presentation."""

    first: tuple[int, ...]
    second: tuple[int, ...]

    @model_validator(mode="after")
    def require_distinct_nonnegative_factorizations(self) -> Self:
        if len(self.first) != len(self.second):
            raise ValueError("relation factorizations must have equal arity")
        if any(value < 0 for value in (*self.first, *self.second)):
            raise ValueError("relation factorizations must be non-negative")
        if self.first == self.second:
            raise ValueError("relation factorizations must be distinct")
        return self


class MinimalPresentationResult(StrictModel):
    """One minimal presentation of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    betti_elements: tuple[CanonicalInteger, ...]
    relations: tuple[MinimalPresentationRelation, ...]

    @model_validator(mode="after")
    def require_minimal_relation_counts(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        _, _, disconnected = betti_data(generators)
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise ValueError("betti_elements do not match the minimal generators")
        relation_components: dict[int, list[tuple[int, int]]] = {
            betti: [] for betti in disconnected
        }
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise ValueError("relation arity does not match minimal generators")
            first_degree = sum(
                coordinate * generator
                for coordinate, generator in zip(
                    relation.first, generators, strict=True
                )
            )
            second_degree = sum(
                coordinate * generator
                for coordinate, generator in zip(
                    relation.second, generators, strict=True
                )
            )
            if first_degree != second_degree or first_degree not in relation_components:
                raise ValueError("relation is not bound to a Betti element")
            components = disconnected[first_degree]
            left_component = _betti_component_index(relation.first, components)
            right_component = _betti_component_index(relation.second, components)
            if left_component == right_component:
                raise ValueError("relation must connect distinct Betti components")
            relation_components[first_degree].append((left_component, right_component))
        expected = {
            betti: len(components) - 1 for betti, components in disconnected.items()
        }
        if {
            betti: len(edges) for betti, edges in relation_components.items()
        } != expected:
            raise ValueError("relations do not have minimal per-Betti cardinality")
        for betti, edges in relation_components.items():
            if not _edges_span(len(disconnected[betti]), edges):
                raise ValueError("relations must span all Betti components")
        return self


class PresentationBinomialsRequest(StrictModel):
    """Convert a minimal presentation to sparse binomial form."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    relations: tuple[MinimalPresentationRelation, ...]

    @model_validator(mode="after")
    def require_kernel_relations(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise ValueError(
                    "relation coordinates must match the minimal generating system"
                )
            first_degree = sum(
                coefficient * generator
                for coefficient, generator in zip(
                    relation.first, generators, strict=True
                )
            )
            second_degree = sum(
                coefficient * generator
                for coefficient, generator in zip(
                    relation.second, generators, strict=True
                )
            )
            if first_degree != second_degree:
                raise ValueError(
                    "relation factorizations must have the same semigroup degree"
                )
        return self


class PresentationBinomial(StrictModel):
    """One sparse binomial (aX - bX) arising from a presentation relation."""

    left_coefficient: Literal["1"] = "1"
    left_exponents: tuple[int, ...]
    right_coefficient: Literal["-1"] = "-1"
    right_exponents: tuple[int, ...]


class PresentationBinomialsResult(StrictModel):
    """Presentation converted to sparse binomials."""

    minimal_generators: tuple[CanonicalInteger, ...]
    binomials: tuple[PresentationBinomial, ...]


class DeltaSetRequest(StrictModel):
    """Global delta set of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_complete_periodicity_range(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_delta_bound(generators)
        return self


class DeltaSetResult(StrictModel):
    """Global delta set of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    delta_set: tuple[int, ...]
    periodicity_bound: int = Field(ge=0)
    checked_through: int = Field(ge=0)
    completeness_basis: Literal["EVENTUAL_PERIODICITY_BOUND"] = (
        "EVENTUAL_PERIODICITY_BOUND"
    )

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise ValueError("delta_set must be strictly increasing and duplicate-free")
        if any(delta <= 0 for delta in self.delta_set):
            raise ValueError("delta values must be positive")
        expected_bound = delta_periodicity_bound(generators)
        if self.periodicity_bound != expected_bound:
            raise ValueError("periodicity_bound does not match the generators")
        if self.checked_through != expected_bound + generators[-1] - 1:
            raise ValueError("checked_through does not match the completeness theorem")
        return self


class ElasticityRequest(StrictModel):
    """Global elasticity of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Distinct, strictly increasing minimal generators, each at most "
            f"{MAX_GENERATOR}."
        ),
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_minimal_generators(self.generators)
        return self


class ElasticityResult(StrictModel):
    """Global elasticity of the semigroup."""

    elasticity: str
    smallest_generator: CanonicalInteger
    largest_generator: CanonicalInteger

    @model_validator(mode="after")
    def require_generator_ratio(self) -> Self:
        smallest = parse_canonical_integer(self.smallest_generator)
        largest = parse_canonical_integer(self.largest_generator)
        if smallest > largest:
            raise ValueError("generator extrema are reversed")
        expected = Fraction(largest, smallest)
        if Fraction(self.elasticity) != expected:
            raise ValueError("elasticity must equal largest/smallest generator")
        return self


class CatenaryDegreeRequest(StrictModel):
    """Global catenary degree of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_complete_betti_graphs(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_catenary_bound(generators)
        return self


class BettiCatenaryDegree(StrictModel):
    """Catenary degree witnessed at one Betti element."""

    betti_element: CanonicalInteger
    catenary_degree: int = Field(ge=0)


class CatenaryDegreeResult(StrictModel):
    """Global catenary degree of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    catenary_degree: int = Field(ge=0)
    betti_degrees: tuple[BettiCatenaryDegree, ...]
    witness_betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_maximizing_witnesses(self) -> Self:
        generators = tuple(map(parse_canonical_integer, self.minimal_generators))
        _, _, disconnected = betti_data(generators)
        expected_records = tuple(
            BettiCatenaryDegree(
                betti_element=str(betti_element),
                catenary_degree=catenary_degree_from_factorizations(
                    factorizations(generators, betti_element)
                ),
            )
            for betti_element in disconnected
        )
        if self.betti_degrees != expected_records:
            raise ValueError("betti_degrees do not match the complete Betti set")
        maximum = max(
            (record.catenary_degree for record in self.betti_degrees), default=0
        )
        if self.catenary_degree != maximum:
            raise ValueError("global catenary degree must be the Betti maximum")
        expected = tuple(
            record.betti_element
            for record in self.betti_degrees
            if maximum > 0 and record.catenary_degree == maximum
        )
        if self.witness_betti_elements != expected:
            raise ValueError("witnesses must be exactly the maximizing Betti elements")
        return self


__all__ = [
    "BettiCatenaryDegree",
    "BettiElementsRequest",
    "BettiElementsResult",
    "CatenaryDegreeRequest",
    "CatenaryDegreeResult",
    "DeltaSetRequest",
    "DeltaSetResult",
    "ElasticityRequest",
    "ElasticityResult",
    "ElementCatenaryDegreeRequest",
    "ElementCatenaryDegreeResult",
    "ElementDeltaSetRequest",
    "ElementDeltaSetResult",
    "ElementElasticityRequest",
    "ElementElasticityResult",
    "FactorizationComputeRequest",
    "FactorizationComputeResult",
    "FactorizationDistanceRequest",
    "FactorizationDistanceResult",
    "FactorizationGraphComputeRequest",
    "FactorizationGraphComputeResult",
    "FactorizationLengthsComputeRequest",
    "FactorizationLengthsComputeResult",
    "MinimalPresentationRelation",
    "MinimalPresentationRequest",
    "MinimalPresentationResult",
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "PresentationBinomial",
    "PresentationBinomialsRequest",
    "PresentationBinomialsResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
]
