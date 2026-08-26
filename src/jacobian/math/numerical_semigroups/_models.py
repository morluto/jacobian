"""Shared canonical numerical-semigroup values and contract helpers."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    apery_set,
    belongs,
    betti_data,
    delta_periodicity_bound,
    factorization_count,
    minimal_generating_system,
)

MAX_GENERATORS = 20
MAX_GENERATOR = 500
MAX_ELEMENT = 10_000
MAX_MATERIALIZED_FACTORIZATIONS = 20_000
MAX_GRAPH_FACTORIZATIONS = 1_000
MAX_GLOBAL_BETTI_ELEMENT = 100_000
MAX_GLOBAL_DELTA_CHECK = 20_000
_GENERAL_GENERATOR_ENVELOPE = f"General-path generators are each at most {MAX_GENERATOR}; a presentation containing 1 canonicalizes to (1,) and is admitted by its constant-size free-semigroup path. "
_GENERAL_ELEMENT_ENVELOPE = f"General-path elements are at most {MAX_ELEMENT}; on the free axis (1,), their exact results remain constant-cardinality."


class NumericalSemigroupRequest(StrictModel):
    """Canonical positive-generator presentation shared by semigroup owners."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


def _validation_error(message: str) -> PydanticCustomError:
    """Build a stable owner-local error for a failed semigroup invariant."""
    reasons = (
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
    return PydanticCustomError(
        f"numerical_semigroup.{next((code for marker, code in reasons if marker in message), 'invariant_mismatch')}",
        message,
    )


def _require_positive_bounded_generators(generators: tuple[str, ...]) -> None:
    values = [parse_canonical_integer(value) for value in generators]
    if any(value <= 0 for value in values):
        raise _validation_error("generators must be positive integers")
    if 1 in values:
        return
    if any(value > MAX_GENERATOR for value in values):
        raise _validation_error(f"generators must be at most {MAX_GENERATOR}")
    gcd = values[0]
    for value in values[1:]:
        while value:
            gcd, value = value, gcd % value
    if gcd != 1:
        raise _validation_error(f"generators must have gcd 1, got gcd {gcd}")


def _require_minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    _require_positive_bounded_generators(generators)
    return minimal_generating_system(
        tuple(sorted({parse_canonical_integer(value) for value in generators}))
    )


def _require_canonical_minimal_axis(generators: tuple[str, ...]) -> tuple[int, ...]:
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
    apery = apery_set(generators)
    conductor = max(apery) - generators[0] + 1
    gaps = tuple(value for value in range(1, conductor) if not belongs(value, apery))
    return generators[0], len(generators), conductor - 1, conductor, len(gaps), gaps


def _require_bounded_value(generators: tuple[int, ...], value: str) -> int:
    parsed = parse_canonical_integer(value)
    if generators != (1,) and parsed > MAX_ELEMENT:
        raise _validation_error(f"value must be at most {MAX_ELEMENT}")
    return parsed


def _require_member(generators: tuple[int, ...], value: int) -> None:
    if not belongs(value, apery_set(generators)):
        raise _validation_error("value must belong to the numerical semigroup")


def _require_materializable_factorizations(
    generators: tuple[int, ...], value: int, maximum: int
) -> None:
    if value >= 0 and (count := factorization_count(generators, value)) > maximum:
        raise _validation_error(
            f"factorization family has {count} members, exceeding the exact materialization bound {maximum}"
        )


def _require_global_betti_bound(generators: tuple[int, ...]) -> None:
    if generators == (1,):
        return
    maximum = max(apery_set(generators)[1:]) + generators[-1]
    if maximum > MAX_GLOBAL_BETTI_ELEMENT:
        raise _validation_error(
            f"complete Apéry candidate range ends at {maximum}, exceeding the global invariant bound {MAX_GLOBAL_BETTI_ELEMENT}"
        )


def _require_global_catenary_bound(generators: tuple[int, ...]) -> None:
    _require_global_betti_bound(generators)
    for value in betti_data(generators)[2]:
        _require_materializable_factorizations(
            generators, value, MAX_GRAPH_FACTORIZATIONS
        )


def _require_global_delta_bound(generators: tuple[int, ...]) -> None:
    checked = delta_periodicity_bound(generators) + generators[-1] - 1
    if checked > MAX_GLOBAL_DELTA_CHECK:
        raise _validation_error(
            f"complete delta-set check requires elements through {checked}, exceeding the bound {MAX_GLOBAL_DELTA_CHECK}"
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
    generators: tuple[int, ...], value: int, family: tuple[tuple[int, ...], ...]
) -> None:
    if len(set(family)) != len(family):
        raise _validation_error("factorizations must be unique")
    for factorization in family:
        if len(factorization) != len(generators) or any(
            coordinate < 0 for coordinate in factorization
        ):
            raise _validation_error("factorization has invalid coordinates")
        if (
            sum(
                coordinate * generator
                for coordinate, generator in zip(factorization, generators, strict=True)
            )
            != value
        ):
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
        while (
            expanded := reached
            | {
                right if left in reached else left
                for left, right in edges
                if left in reached or right in reached
            }
        ) != reached:
            reached = expanded
        unseen.difference_update(reached)
        components.append(tuple(sorted(reached)))
    return edges, tuple(components)


__all__ = [
    "MAX_ELEMENT",
    "MAX_GENERATOR",
    "MAX_GENERATORS",
    "MAX_GLOBAL_BETTI_ELEMENT",
    "MAX_GLOBAL_DELTA_CHECK",
    "MAX_GRAPH_FACTORIZATIONS",
    "MAX_MATERIALIZED_FACTORIZATIONS",
    "NumericalSemigroupRequest",
]
