"""Presentation operations for numerical semigroups."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    factorization_predecessors,
    reconstruct_factorization,
)
from jacobian.math.numerical_semigroups._models import _require_minimal_generators
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
    return MinimalPresentationResult(
        minimal_generators=tuple(
            format_canonical_integer(value) for value in generators
        ),
        betti_elements=tuple(format_canonical_integer(value) for value in disconnected),
        relations=tuple(relations),
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
]
