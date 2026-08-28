"""Exact bounded native constructions for finite categories."""

from jacobian.math.finite_categories._models import CategoryProfileResult
from jacobian.math.finite_categories._product import product
from jacobian.math.finite_categories.values import (
    CategoryIdentifier,
    FiniteCategory,
    MorphismSpec,
)


def _category_profile_data(
    category: FiniteCategory,
) -> tuple[
    tuple[tuple[CategoryIdentifier, CategoryIdentifier, int], ...],
    tuple[tuple[CategoryIdentifier, int], ...],
]:
    """Compute canonical hom-set and endomorphism profile data."""

    hom_counts: dict[tuple[CategoryIdentifier, CategoryIdentifier], int] = {}
    for morphism in category.morphisms:
        key = (morphism.source, morphism.target)
        hom_counts[key] = hom_counts.get(key, 0) + 1

    hom_sets = tuple(
        (source, target, hom_counts[(source, target)])
        for source in category.objects
        for target in category.objects
        if (source, target) in hom_counts
    )

    endomorphism_counts: dict[CategoryIdentifier, int] = {}
    for morphism in category.morphisms:
        if morphism.source == morphism.target:
            endomorphism_counts[morphism.source] = (
                endomorphism_counts.get(morphism.source, 0) + 1
            )
    endomorphisms = tuple(
        (obj, endomorphism_counts[obj])
        for obj in category.objects
        if obj in endomorphism_counts
    )
    return hom_sets, endomorphisms


def category_profile(category: FiniteCategory) -> CategoryProfileResult:
    """Compute hom-set and endomorphism counts of a finite category."""

    hom_sets, endomorphisms = _category_profile_data(category)
    return CategoryProfileResult._from_kernel(category, hom_sets, endomorphisms)


def opposite_category(category: FiniteCategory) -> FiniteCategory:
    """Return the finite category with all arrows and compositions reversed."""

    opposite_morphisms = tuple(
        MorphismSpec(
            morphism_id=morphism.morphism_id,
            source=morphism.target,
            target=morphism.source,
        )
        for morphism in category.morphisms
    )
    opposite_composition = tuple(
        (right, left, result) for left, right, result in category.composition
    )
    return FiniteCategory(
        objects=category.objects,
        morphisms=opposite_morphisms,
        identities=category.identities,
        composition=opposite_composition,
    )


__all__ = ["category_profile", "opposite_category", "product"]
