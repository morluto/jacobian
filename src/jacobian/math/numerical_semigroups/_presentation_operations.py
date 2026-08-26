"""Presentation operations for numerical semigroups."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    factorization_predecessors,
    reconstruct_factorization,
)
from jacobian.math.numerical_semigroups._models import (
    _betti_component_index,
    _edges_span,
    _require_minimal_generators,
)
from jacobian.math.numerical_semigroups._presentation_models import (
    MinimalPresentationRelation,
    MinimalPresentationRequest,
    MinimalPresentationResult,
    PresentationBinomial,
    PresentationBinomialsRequest,
    PresentationBinomialsResult,
)


def _minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    """Normalize a presentation to its canonical generator axis."""

    return _require_minimal_generators(generators)


def compute_minimal_presentation(
    request: MinimalPresentationRequest,
) -> MinimalPresentationResult:
    """Build minimal spanning relations for every Betti fiber."""

    generators = _minimal_generators(request.generators)
    _, _, disconnected = betti_data(generators)
    predecessors = factorization_predecessors(generators, max(disconnected, default=0))
    relations: list[MinimalPresentationRelation] = []
    for betti_value, components in disconnected.items():
        representatives: list[tuple[int, ...]] = []
        for component in components:
            generator_index = component[0]
            residual = reconstruct_factorization(
                generators, predecessors, betti_value - generators[generator_index]
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
        minimal_generators=tuple(
            format_canonical_integer(value) for value in generators
        ),
        betti_elements=tuple(format_canonical_integer(value) for value in disconnected),
        relations=tuple(relations),
    )


def verify_minimal_presentation_result(result: MinimalPresentationResult) -> bool:
    """Replay the bounded complete Betti-component presentation check."""

    generators = tuple(
        parse_canonical_integer(value) for value in result.minimal_generators
    )
    _, _, disconnected = betti_data(generators)
    if tuple(
        parse_canonical_integer(value) for value in result.betti_elements
    ) != tuple(disconnected):
        return False
    relation_components: dict[int, list[tuple[int, int]]] = {
        betti: [] for betti in disconnected
    }
    for relation in result.relations:
        first_degree = sum(
            coordinate * generator
            for coordinate, generator in zip(relation.first, generators, strict=True)
        )
        if first_degree not in relation_components:
            return False
        components = disconnected[first_degree]
        left = _betti_component_index(relation.first, components)
        right = _betti_component_index(relation.second, components)
        if left == right:
            return False
        relation_components[first_degree].append((left, right))
    expected = {
        betti: len(components) - 1 for betti, components in disconnected.items()
    }
    if {betti: len(edges) for betti, edges in relation_components.items()} != expected:
        return False
    return all(
        _edges_span(len(disconnected[betti]), edges)
        for betti, edges in relation_components.items()
    )


def compute_presentation_binomials(
    request: PresentationBinomialsRequest,
) -> PresentationBinomialsResult:
    """Project homogeneous presentation relations to sparse binomials."""

    generators = _minimal_generators(request.generators)
    return PresentationBinomialsResult(
        minimal_generators=tuple(
            format_canonical_integer(value) for value in generators
        ),
        binomials=tuple(
            PresentationBinomial(
                left_exponents=tuple(relation.first),
                right_exponents=tuple(relation.second),
            )
            for relation in request.relations
        ),
    )


__all__ = [
    "compute_minimal_presentation",
    "compute_presentation_binomials",
    "verify_minimal_presentation_result",
]
