"""Exact finite category operations."""

from jacobian.math.finite_categories._models import (
    CategoryProfileResult,
    FiniteCategoryRequest,
    MorphismSpec,
    OppositeCategoryResult,
)


def compute_category_profile(request: FiniteCategoryRequest) -> CategoryProfileResult:
    """Compute the profile of a finite category."""
    objects = request.objects
    morphisms = request.morphisms

    # Build hom-sets: Hom(a,b) = count of morphisms from a to b.
    hom_counts: dict[tuple[str, str], int] = {}
    for m in morphisms:
        key = (m.source, m.target)
        hom_counts[key] = hom_counts.get(key, 0) + 1

    hom_list: list[tuple[str, str, int]] = []
    for a in objects:
        for b in objects:
            count = hom_counts.get((a, b), 0)
            if count > 0:
                hom_list.append((a, b, count))

    # Endomorphisms: morphisms where source == target.
    endo_counts: dict[str, int] = {}
    for m in morphisms:
        if m.source == m.target:
            endo_counts[m.source] = endo_counts.get(m.source, 0) + 1
    endo_list = [
        (obj, endo_counts.get(obj, 0)) for obj in objects if obj in endo_counts
    ]

    # Designated identity morphisms (one per object, from the value).
    identities = tuple(request.identities)

    return CategoryProfileResult(
        objects=objects,
        num_objects=len(objects),
        num_morphisms=len(morphisms),
        hom_sets=tuple(hom_list),
        endomorphisms=tuple(endo_list),
        identity_morphisms=identities,
    )


def compute_opposite_category(request: FiniteCategoryRequest) -> OppositeCategoryResult:
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
    return OppositeCategoryResult(
        objects=request.objects,
        morphisms=opposite_morphisms,
        identities=request.identities,
        composition=opposite_composition,
    )
