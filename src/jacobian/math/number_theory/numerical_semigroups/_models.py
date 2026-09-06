"""Shared canonical numerical-semigroup values and contract helpers."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.numerical_semigroups._algorithms import (
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


def _run_admission[T](
    operation: str,
    location: tuple[str | int, ...],
    action: Callable[[], T],
) -> T:
    """Run owner admission and expose one typed native-operation failure."""

    try:
        return action()
    except ValueError as error:
        code = getattr(error, "type", None)
        if not isinstance(code, str):
            code = f"numerical_semigroup.{operation}_admission"
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(error),
        ) from error


class NumericalSemigroupRequest(StrictModel):
    """Canonical positive-generator presentation shared by semigroup owners."""

    generators: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )


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


def _require_positive_bounded_generators(generators: tuple[int, ...]) -> None:
    values = generators
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


def _require_minimal_generators(generators: tuple[int, ...]) -> tuple[int, ...]:
    _require_positive_bounded_generators(generators)
    return minimal_generating_system(tuple(sorted(set(generators))))


def _require_canonical_generator_axis(generators: tuple[int, ...]) -> tuple[int, ...]:
    """Validate the retained generator axis without recomputing minimality."""

    _require_positive_bounded_generators(generators)
    values = generators
    if values != tuple(sorted(set(values))):
        raise _validation_error(
            "minimal_generators must be strictly increasing and duplicate-free"
        )
    return values


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
