"""Typed wire contracts for numerical semigroup operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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


def _validation_error(message: str) -> PydanticCustomError:
    """Build a stable owner-local error for a failed semigroup invariant."""

    semantic_reasons = (
        ("elasticity is defined", "elasticity_undefined_for_zero"),
        ("delta_set must", "delta_set_not_canonical"),
        ("delta values", "delta_values_invalid"),
        ("is_connected", "graph_connectivity_mismatch"),
        ("candidate_count", "betti_candidate_count_mismatch"),
        ("factorization_lengths", "factorization_lengths_mismatch"),
        ("binomial exponents must match", "binomial_axis_mismatch"),
        ("relation arity", "relation_dimension_mismatch"),
        ("relation is not", "relation_betti_binding_mismatch"),
        ("relations do not", "relation_cardinality_mismatch"),
        ("relations must", "relation_connectivity_mismatch"),
        ("binomial terms", "binomial_degree_mismatch"),
        ("same semigroup degree", "binomial_degree_mismatch"),
        ("relation coordinates", "relation_dimension_mismatch"),
        ("delta_set does not", "delta_set_mismatch"),
        ("length extrema", "length_extrema_mismatch"),
        ("global catenary", "global_catenary_mismatch"),
        ("betti_degrees", "betti_degrees_mismatch"),
        ("positive", "generators_not_positive"),
        ("at most", "value_exceeds_bound"),
        ("gcd", "generators_not_coprime"),
        ("canonical minimal", "generator_axis_not_canonical"),
        ("belong", "value_not_in_semigroup"),
        ("materialization bound", "factorization_materialization_exceeded"),
        ("candidate range", "betti_candidate_bound_exceeded"),
        ("delta-set check", "delta_check_bound_exceeded"),
        ("Betti component", "factorization_component_mismatch"),
        ("factorizations must be unique", "factorizations_not_unique"),
        ("invalid coordinates", "factorization_coordinates_invalid"),
        ("does not evaluate", "factorization_value_mismatch"),
        ("complete family", "factorization_family_incomplete"),
        ("membership must agree", "membership_mismatch"),
        ("lengths must", "factorization_lengths_not_canonical"),
        ("lengths do not", "factorization_lengths_mismatch"),
        ("coordinates must be", "factorization_coordinates_negative"),
        ("coordinates must", "factorization_dimension_mismatch"),
        ("both factorizations", "factorization_value_mismatch"),
        ("multiplicity", "summary_multiplicity_mismatch"),
        ("embedding_dimension", "summary_embedding_dimension_mismatch"),
        ("frobenius_number", "summary_frobenius_mismatch"),
        ("conductor", "summary_conductor_mismatch"),
        ("genus", "summary_genus_mismatch"),
        ("gaps", "summary_gaps_mismatch"),
        ("graph vertices", "graph_vertex_membership_mismatch"),
        ("connected components", "graph_components_mismatch"),
        ("graph edge", "graph_edge_invalid"),
        ("shared-support adjacency", "graph_adjacency_mismatch"),
        ("strictly increasing", "factorization_lengths_not_canonical"),
        ("periodicity_bound", "delta_periodicity_bound_mismatch"),
        ("checked_through", "delta_checked_range_mismatch"),
        ("elasticity", "elasticity_mismatch"),
        ("factorization_count", "factorization_count_mismatch"),
        ("catenary_degree", "catenary_degree_mismatch"),
        ("apery_set", "apery_set_mismatch"),
        ("betti_elements", "betti_elements_mismatch"),
        ("relation factorizations", "relation_factorizations_invalid"),
        ("binomial exponents", "binomial_exponents_invalid"),
        ("generator extrema", "generator_extrema_reversed"),
        ("witnesses", "catenary_witnesses_mismatch"),
    )
    reason = next(
        (reason for marker, reason in semantic_reasons if marker in message),
        "invariant_mismatch",
    )
    return PydanticCustomError(f"numerical_semigroup.{reason}", message)


def _require_positive_bounded_generators(generators: tuple[str, ...]) -> None:
    values: list[int] = []
    for generator in generators:
        value = parse_canonical_integer(generator)
        if value <= 0:
            raise _validation_error("generators must be positive integers")
        if value > MAX_GENERATOR:
            raise _validation_error(f"generators must be at most {MAX_GENERATOR}")
        values.append(value)
    gcd = values[0]
    for value in values[1:]:
        while value:
            gcd, value = value, gcd % value
    if gcd != 1:
        raise _validation_error(f"generators must have gcd 1, got gcd {gcd}")


def _require_minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    """Normalize a valid presentation to its increasing minimal atom axis."""

    _require_positive_bounded_generators(generators)
    values = tuple(sorted({parse_canonical_integer(value) for value in generators}))
    return minimal_generating_system(values)


def _require_canonical_minimal_axis(
    generators: tuple[str, ...],
) -> tuple[int, ...]:
    """Reject results whose coordinate axis is not the semigroup's atom axis."""

    _require_positive_bounded_generators(generators)
    values = tuple(parse_canonical_integer(value) for value in generators)
    if values != _require_minimal_generators(generators):
        raise _validation_error(
            "minimal_generators must use the canonical minimal generator axis"
        )
    return values


def _summary_invariants(
    generators: tuple[int, ...],
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Replay the exact numerical-semigroup summary from its atom axis."""

    apery = apery_set(generators)
    conductor = max(apery) - generators[0] + 1
    gaps = tuple(value for value in range(1, conductor) if not belongs(value, apery))
    return (
        generators[0],
        len(generators),
        conductor - 1,
        conductor,
        len(gaps),
        gaps,
    )


def _require_bounded_value(value: str) -> int:
    parsed = parse_canonical_integer(value)
    if parsed > MAX_ELEMENT:
        raise _validation_error(f"value must be at most {MAX_ELEMENT}")
    return parsed


def _require_member(generators: tuple[int, ...], value: int) -> None:
    if not belongs(value, apery_set(generators)):
        raise _validation_error("value must belong to the numerical semigroup")


def _require_materializable_factorizations(
    generators: tuple[int, ...], value: int, maximum: int
) -> None:
    if value < 0:
        return
    count = factorization_count(generators, value)
    if count > maximum:
        raise _validation_error(
            f"factorization family has {count} members, exceeding the exact "
            f"materialization bound {maximum}"
        )


def _require_global_betti_bound(generators: tuple[int, ...]) -> None:
    if generators == (1,):
        return
    apery = apery_set(generators)
    maximum_candidate = max(apery[1:]) + generators[-1]
    if maximum_candidate > MAX_GLOBAL_BETTI_ELEMENT:
        raise _validation_error(
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
        raise _validation_error(
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
        raise _validation_error(
            "relation factorization is not bound to one Betti component"
        )
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
        raise _validation_error("factorizations must be unique")
    for factorization in family:
        if len(factorization) != len(generators) or any(
            coordinate < 0 for coordinate in factorization
        ):
            raise _validation_error("factorization has invalid coordinates")
        degree = sum(
            coordinate * generator
            for coordinate, generator in zip(factorization, generators, strict=True)
        )
        if degree != value:
            raise _validation_error(
                "factorization does not evaluate to the result value"
            )
    if len(family) != factorization_count(generators, value):
        raise _validation_error("factorizations do not form the complete family")


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

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_exact_summary(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        (
            multiplicity,
            embedding_dimension,
            frobenius_number,
            conductor,
            genus,
            gaps,
        ) = _summary_invariants(generators)
        if parse_canonical_integer(self.multiplicity) != multiplicity:
            raise _validation_error(
                "multiplicity does not match the minimal generators"
            )
        if self.embedding_dimension != embedding_dimension:
            raise _validation_error(
                "embedding_dimension does not match the minimal generators"
            )
        if parse_canonical_integer(self.frobenius_number) != frobenius_number:
            raise _validation_error(
                "frobenius_number does not match the minimal generators"
            )
        if parse_canonical_integer(self.conductor) != conductor:
            raise _validation_error("conductor does not match the minimal generators")
        if self.genus != genus:
            raise _validation_error("genus does not match the minimal generators")
        if tuple(map(parse_canonical_integer, self.gaps)) != gaps:
            raise _validation_error("gaps do not match the minimal generators")
        return self


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
            raise _validation_error(f"membership value must be at most {MAX_ELEMENT}")
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
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; returned "
            "coordinates use its increasing minimal generator axis."
        ),
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
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_exact_family(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error(
                "membership must agree with the factorization family"
            )
        _require_exact_factorization_family(generators, value, self.factorizations)
        return self


class FactorizationLengthsComputeRequest(StrictModel):
    """Compute the complete sorted length set of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; factorization "
            "lengths use its increasing minimal generator axis."
        ),
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
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    lengths: tuple[int, ...]

    @model_validator(mode="after")
    def require_length_set(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.lengths):
            raise _validation_error(
                "membership must agree with the factorization lengths"
            )
        if self.lengths != tuple(sorted(set(self.lengths))):
            raise _validation_error(
                "lengths must be strictly increasing and duplicate-free"
            )
        if any(length < 0 for length in self.lengths):
            raise _validation_error("factorization lengths must be non-negative")
        if self.lengths != factorization_lengths(generators, value):
            raise _validation_error("lengths do not form the complete length set")
        return self


class FactorizationDistanceRequest(StrictModel):
    """Distance between two factorizations of the same element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant, but both "
            "factorization coordinate tuples must use its increasing minimal "
            "generator axis."
        ),
    )
    value: CanonicalInteger
    first: tuple[int, ...] = Field(min_length=1)
    second: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        if any(c < 0 for c in self.first) or any(c < 0 for c in self.second):
            raise _validation_error("factorization coordinates must be non-negative")
        if len(self.first) != len(generators) or len(self.second) != len(generators):
            raise _validation_error(
                "factorization coordinates must match the minimal generating system"
            )
        first_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.first, generators, strict=True)
        )
        second_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.second, generators, strict=True)
        )
        if first_value != value or second_value != value:
            raise _validation_error(
                "both factorizations must evaluate to the declared value"
            )
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
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; graph vertices use "
            "its increasing minimal generator axis."
        ),
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
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    connected_components: tuple[tuple[int, ...], ...]
    is_connected: bool

    @model_validator(mode="after")
    def require_graph_partition(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        vertex_count = len(self.factorizations)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error("membership must agree with graph vertices")
        vertices = tuple(
            index for component in self.connected_components for index in component
        )
        if tuple(sorted(vertices)) != tuple(range(vertex_count)):
            raise _validation_error(
                "connected components must partition all graph vertices"
            )
        if self.is_connected != (len(self.connected_components) <= 1):
            raise _validation_error("is_connected must agree with connected components")
        for left, right in self.edges:
            if not 0 <= left < right < vertex_count:
                raise _validation_error("graph edge has invalid vertex indices")
        _require_exact_factorization_family(generators, value, self.factorizations)
        expected_edges, expected_components = _factorization_graph_data(
            self.factorizations
        )
        if self.edges != expected_edges:
            raise _validation_error("edges do not match shared-support adjacency")
        if self.connected_components != expected_components:
            raise _validation_error("connected components do not match the graph")
        return self


class ElementDeltaSetRequest(StrictModel):
    """Delta set of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; factorization "
            "lengths use its increasing minimal generator axis."
        ),
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
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_lengths: tuple[int, ...]
    delta_set: tuple[int, ...]

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        expected_lengths = factorization_lengths(generators, value)
        if self.factorization_lengths != expected_lengths:
            raise _validation_error("factorization_lengths do not match the element")
        expected_delta = tuple(
            sorted({right - left for left, right in pairwise(expected_lengths)})
        )
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
        if self.delta_set != expected_delta:
            raise _validation_error("delta_set does not match the complete length set")
        return self


class ElementElasticityRequest(StrictModel):
    """Elasticity of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; results use its "
            "increasing minimal generator axis."
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
            raise _validation_error(
                "elasticity is defined here only for positive elements"
            )
        _require_member(generators, value)
        return self


class ElementElasticityResult(StrictModel):
    """Elasticity of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    minimum_length: int = Field(ge=1)
    maximum_length: int = Field(ge=1)
    elasticity: str

    @model_validator(mode="after")
    def require_length_ratio(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        expected_extrema = factorization_length_extrema(
            generators, parse_canonical_integer(self.value)
        )
        if (self.minimum_length, self.maximum_length) != expected_extrema:
            raise _validation_error("length extrema do not match the element")
        if Fraction(self.elasticity) != Fraction(
            self.maximum_length, self.minimum_length
        ):
            raise _validation_error("elasticity does not match the length ratio")
        return self


class ElementCatenaryDegreeRequest(StrictModel):
    """Catenary degree of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; factorization "
            "coordinates use its increasing minimal generator axis."
        ),
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
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_count: int = Field(ge=1)
    catenary_degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_exact_degree(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        family = factorizations(generators, parse_canonical_integer(self.value))
        if self.factorization_count != len(family):
            raise _validation_error("factorization_count does not match the element")
        if self.catenary_degree != catenary_degree_from_factorizations(family):
            raise _validation_error(
                "catenary_degree does not match the factorization graph"
            )
        return self


class BettiElementsRequest(StrictModel):
    """Betti elements of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; derived data uses "
            "its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_candidate_range(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_betti_bound(generators)
        return self


class BettiElementsResult(StrictModel):
    """Betti elements of a semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    apery_set: tuple[CanonicalInteger, ...]
    candidate_count: int = Field(ge=0)
    betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_complete_betti_data(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        apery, candidates, disconnected = betti_data(generators)
        if tuple(map(parse_canonical_integer, self.apery_set)) != apery:
            raise _validation_error("apery_set does not match the minimal generators")
        if self.candidate_count != len(candidates):
            raise _validation_error("candidate_count does not match the complete range")
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise _validation_error(
                "betti_elements do not match disconnected candidates"
            )
        return self


class MinimalPresentationRequest(StrictModel):
    """One minimal presentation of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; returned relations "
            "use its increasing minimal generator axis."
        ),
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
            raise _validation_error("relation factorizations must have equal arity")
        if any(value < 0 for value in (*self.first, *self.second)):
            raise _validation_error("relation factorizations must be non-negative")
        if self.first == self.second:
            raise _validation_error("relation factorizations must be distinct")
        return self


class MinimalPresentationResult(StrictModel):
    """One minimal presentation of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    betti_elements: tuple[CanonicalInteger, ...]
    relations: tuple[MinimalPresentationRelation, ...]

    @model_validator(mode="after")
    def require_minimal_relation_counts(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _, _, disconnected = betti_data(generators)
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise _validation_error(
                "betti_elements do not match the minimal generators"
            )
        relation_components: dict[int, list[tuple[int, int]]] = {
            betti: [] for betti in disconnected
        }
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise _validation_error(
                    "relation arity does not match minimal generators"
                )
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
                raise _validation_error("relation is not bound to a Betti element")
            components = disconnected[first_degree]
            left_component = _betti_component_index(relation.first, components)
            right_component = _betti_component_index(relation.second, components)
            if left_component == right_component:
                raise _validation_error(
                    "relation must connect distinct Betti components"
                )
            relation_components[first_degree].append((left_component, right_component))
        expected = {
            betti: len(components) - 1 for betti, components in disconnected.items()
        }
        if {
            betti: len(edges) for betti, edges in relation_components.items()
        } != expected:
            raise _validation_error(
                "relations do not have minimal per-Betti cardinality"
            )
        for betti, edges in relation_components.items():
            if not _edges_span(len(disconnected[betti]), edges):
                raise _validation_error("relations must span all Betti components")
        return self


class PresentationBinomialsRequest(StrictModel):
    """Convert a minimal presentation to sparse binomial form."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; relation "
            "coordinates must use its increasing minimal generator axis."
        ),
    )
    relations: tuple[MinimalPresentationRelation, ...]

    @model_validator(mode="after")
    def require_kernel_relations(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise _validation_error(
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
                raise _validation_error(
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

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    binomials: tuple[PresentationBinomial, ...]

    @model_validator(mode="after")
    def require_canonical_axis_and_homogeneous_binomials(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        for binomial in self.binomials:
            if len(binomial.left_exponents) != len(generators) or len(
                binomial.right_exponents
            ) != len(generators):
                raise _validation_error(
                    "binomial exponents must match the minimal generator axis"
                )
            if any(
                exponent < 0
                for exponent in (*binomial.left_exponents, *binomial.right_exponents)
            ):
                raise _validation_error("binomial exponents must be non-negative")
            left_degree = sum(
                exponent * generator
                for exponent, generator in zip(
                    binomial.left_exponents, generators, strict=True
                )
            )
            right_degree = sum(
                exponent * generator
                for exponent, generator in zip(
                    binomial.right_exponents, generators, strict=True
                )
            )
            if left_degree != right_degree:
                raise _validation_error(
                    "binomial terms must have the same semigroup degree"
                )
        return self


class DeltaSetRequest(StrictModel):
    """Global delta set of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; the complete delta "
            "set uses its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_periodicity_range(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_global_delta_bound(generators)
        return self


class DeltaSetResult(StrictModel):
    """Global delta set of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    delta_set: tuple[int, ...]
    periodicity_bound: int = Field(ge=0)
    checked_through: int = Field(ge=0)
    completeness_basis: Literal["EVENTUAL_PERIODICITY_BOUND"] = (
        "EVENTUAL_PERIODICITY_BOUND"
    )

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
        expected_bound = delta_periodicity_bound(generators)
        if self.periodicity_bound != expected_bound:
            raise _validation_error("periodicity_bound does not match the generators")
        if self.checked_through != expected_bound + generators[-1] - 1:
            raise _validation_error(
                "checked_through does not match the completeness theorem"
            )
        return self


class ElasticityRequest(StrictModel):
    """Global elasticity of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; the ratio uses its "
            "increasing minimal generator axis."
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
            raise _validation_error("generator extrema are reversed")
        expected = Fraction(largest, smallest)
        if Fraction(self.elasticity) != expected:
            raise _validation_error("elasticity must equal largest/smallest generator")
        return self


class CatenaryDegreeRequest(StrictModel):
    """Global catenary degree of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; factorization "
            "coordinates use its increasing minimal generator axis."
        ),
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

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    catenary_degree: int = Field(ge=0)
    betti_degrees: tuple[BettiCatenaryDegree, ...]
    witness_betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_maximizing_witnesses(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
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
            raise _validation_error("betti_degrees do not match the complete Betti set")
        maximum = max(
            (record.catenary_degree for record in self.betti_degrees), default=0
        )
        if self.catenary_degree != maximum:
            raise _validation_error("global catenary degree must be the Betti maximum")
        expected = tuple(
            record.betti_element
            for record in self.betti_degrees
            if maximum > 0 and record.catenary_degree == maximum
        )
        if self.witness_betti_elements != expected:
            raise _validation_error(
                "witnesses must be exactly the maximizing Betti elements"
            )
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
