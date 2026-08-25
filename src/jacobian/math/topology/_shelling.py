"""Pure replay kernel for simplicial-complex shelling checks."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.topology._models import Simplex


def _restriction_failure(
    facet: set[str],
    previous_facets: list[set[str]],
) -> str | None:
    """Return why the new-faces interval of one facet is not a shelling."""

    faces_not_in_previous: set[tuple[str, ...]] = set()
    for size in range(1, len(facet) + 1):
        for subset in combinations(sorted(facet), size):
            subset_set = set(subset)
            if not any(subset_set.issubset(previous) for previous in previous_facets):
                faces_not_in_previous.add(subset)
    if not faces_not_in_previous:
        return "has no new faces"
    face_sets = {frozenset(face) for face in faces_not_in_previous}
    minimal_faces = [
        face
        for face in face_sets
        if not any(other < face for other in face_sets if other != face)
    ]
    if len(minimal_faces) != 1:
        return "restriction is not an interval"
    return None


def evaluate_shelling(
    facets: tuple[Simplex, ...],
    facet_order: tuple[int, ...],
) -> tuple[bool, int | None, str | None]:
    """Recompute the shelling decision for a submitted facet order."""

    dimension = len(facets[0])
    if not all(len(facet) == dimension for facet in facets):
        return False, 0, "complex is not pure"

    for position, index in enumerate(facet_order):
        facet = set(facets[index])
        if position == 0:
            continue
        previous_facets = [
            set(facets[facet_order[offset]]) for offset in range(position)
        ]
        if any(facet & previous == facet for previous in previous_facets):
            return False, position, f"facet {index} is contained in earlier facet"
        failure = _restriction_failure(facet, previous_facets)
        if failure is not None:
            return False, position, f"facet {index} {failure}"

    return True, None, None


__all__ = ["evaluate_shelling"]
