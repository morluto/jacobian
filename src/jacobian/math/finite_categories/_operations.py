"""Exact finite category operations."""

from jacobian.math.finite_categories._models import (
    CategoryProductRequest,
    CategoryProfileResult,
)
from jacobian.math.finite_categories._product import product
from jacobian.math.finite_categories.values import (
    CategoryIdentifier,
    FiniteCategory,
    FiniteCategoryProduct,
    MorphismSpec,
)


def _category_profile_data(
    category: FiniteCategory,
) -> tuple[
    tuple[tuple[CategoryIdentifier, CategoryIdentifier, int], ...],
    tuple[tuple[CategoryIdentifier, int], ...],
]:
    """Compute the canonical derived profile data for a bounded category."""

    objects = category.objects
    morphisms = category.morphisms

    # Build hom-sets: Hom(a,b) = count of morphisms from a to b.
    hom_counts: dict[tuple[CategoryIdentifier, CategoryIdentifier], int] = {}
    for m in morphisms:
        key = (m.source, m.target)
        hom_counts[key] = hom_counts.get(key, 0) + 1

    hom_list: list[tuple[CategoryIdentifier, CategoryIdentifier, int]] = []
    for a in objects:
        for b in objects:
            count = hom_counts.get((a, b), 0)
            if count > 0:
                hom_list.append((a, b, count))

    # Endomorphisms: morphisms where source == target.
    endo_counts: dict[CategoryIdentifier, int] = {}
    for m in morphisms:
        if m.source == m.target:
            endo_counts[m.source] = endo_counts.get(m.source, 0) + 1
    endo_list = [
        (obj, endo_counts.get(obj, 0)) for obj in objects if obj in endo_counts
    ]

    return tuple(hom_list), tuple(endo_list)


def compute_category_profile(request: FiniteCategory) -> CategoryProfileResult:
    """Compute the profile of a finite category."""

    hom_sets, endomorphisms = _category_profile_data(request)
    return CategoryProfileResult._from_kernel(request, hom_sets, endomorphisms)


def verify_category_profile_claim(
    category: FiniteCategory, result: CategoryProfileResult
) -> bool:
    """Check a supplied profile claim against its explicit source category."""

    hom_sets, endomorphisms = _category_profile_data(category)
    return (
        result.objects == category.objects
        and result.num_objects == len(category.objects)
        and result.num_morphisms == len(category.morphisms)
        and result.hom_sets == hom_sets
        and result.endomorphisms == endomorphisms
        and result.identity_morphisms == category.identities
    )


def compute_opposite_category(request: FiniteCategory) -> FiniteCategory:
    """Compute the opposite category.

    Morphism directions are reversed and composition order is reversed: a
    composition ``(g, f, r)`` in the source category becomes ``(f, g, r)``
    in the opposite, so that ``f^op . g^op = r^op``.
    """
    opposite_morphisms = tuple(
        MorphismSpec(
            morphism_id=m.morphism_id,
            source=m.target,
            target=m.source,
        )
        for m in request.morphisms
    )
    opposite_composition = tuple(
        (f, g, result) for (g, f, result) in request.composition
    )
    return FiniteCategory(
        objects=request.objects,
        morphisms=opposite_morphisms,
        identities=request.identities,
        composition=opposite_composition,
    )


def compute_category_product(
    request: CategoryProductRequest,
) -> FiniteCategoryProduct:
    """Compute the exact Cartesian product of two finite categories."""

    return product(request.left, request.right)


__all__ = [
    "compute_category_product",
    "compute_category_profile",
    "compute_opposite_category",
]
