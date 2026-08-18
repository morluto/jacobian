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

    # Build hom-sets: Hom(a,b) = count of morphisms from a to b
    hom_counts: dict[tuple[str, str], int] = {}
    for m in morphisms:
        key = (m.source, m.target)
        hom_counts[key] = hom_counts.get(key, 0) + 1

    hom_list = []
    for a in objects:
        for b in objects:
            count = hom_counts.get((a, b), 0)
            if count > 0:
                hom_list.append((f"{a}->{b}", count))

    # Endomorphisms: morphisms where source == target
    endo_counts: dict[str, int] = {}
    for m in morphisms:
        if m.source == m.target:
            endo_counts[m.source] = endo_counts.get(m.source, 0) + 1
    endo_list = [
        (obj, endo_counts.get(obj, 0)) for obj in objects if obj in endo_counts
    ]

    # Identity morphisms (one per object, if present)
    identities = []
    for obj in objects:
        for m in morphisms:
            if m.source == obj and m.target == obj:
                identities.append((obj, m.morphism_id))
                break

    return CategoryProfileResult(
        objects=objects,
        num_objects=len(objects),
        num_morphisms=len(morphisms),
        hom_sets=tuple(hom_list),
        endomorphisms=tuple(endo_list),
        identity_morphisms=tuple(identities),
    )


def compute_opposite_category(request: FiniteCategoryRequest) -> OppositeCategoryResult:
    """Compute the opposite category with reversed morphisms."""
    opposite_morphisms = tuple(
        MorphismSpec(
            morphism_id=m.morphism_id,
            source=m.target,
            target=m.source,
        )
        for m in request.morphisms
    )
    return OppositeCategoryResult(
        objects=request.objects,
        morphisms=opposite_morphisms,
    )
