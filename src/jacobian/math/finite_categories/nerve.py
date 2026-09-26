"""Bounded exact nerve prefixes of finite categories."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_categories._models import FiniteCategoryNerve
from jacobian.math.finite_categories.values import (
    CategoryIdentifier,
    FiniteCategory,
    MorphismSpec,
    _check_category_laws,
    _identifier_character_count,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_SIMPLICIAL_SET_DEGREE,
    MAX_TOTAL_SIMPLICES,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

NERVE_IDENTITY_WORK_BOUND = 100_000
NERVE_TRANSPORT_BYTE_BOUND = 10 * 1024 * 1024


def _category_tables(
    category: FiniteCategory,
) -> tuple[
    dict[CategoryIdentifier, MorphismSpec],
    dict[CategoryIdentifier, CategoryIdentifier],
    dict[tuple[CategoryIdentifier, CategoryIdentifier], CategoryIdentifier],
    dict[CategoryIdentifier, list[CategoryIdentifier]],
]:
    morphisms = category.morphisms
    by_id = {morphism.morphism_id: morphism for morphism in morphisms}
    identities = dict(category.identities)
    composition = {(g, f): result for g, f, result in category.composition}
    outgoing: dict[CategoryIdentifier, list[CategoryIdentifier]] = {
        obj: [] for obj in category.objects
    }
    for morphism in morphisms:
        outgoing[morphism.source].append(morphism.morphism_id)
    return by_id, identities, composition, outgoing


def _admit(category: FiniteCategory, degree: int) -> None:
    if not isinstance(category, FiniteCategory):
        raise OperationDomainValidationError(
            location=("category",),
            code="finite_category.nerve_category_type",
            message="category must be a FiniteCategory",
        )
    if not isinstance(degree, int) or isinstance(degree, bool):
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="finite_category.nerve_degree_type",
            message="max_degree must be an integer",
        )
    if not 0 <= degree <= MAX_SIMPLICIAL_SET_DEGREE:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="finite_category.nerve_degree_out_of_bounds",
            message=f"max_degree must lie in 0..{MAX_SIMPLICIAL_SET_DEGREE}",
        )
    if not category.objects:
        raise OperationResourceAdmissionError(
            location=("category",),
            code="finite_category.nerve_empty_category",
            message="the nonempty simplicial-set carrier cannot represent an empty nerve",
        )
    try:
        _check_category_laws(category)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("category",), code=exc.type, message=exc.message()
        ) from exc
    by_id, _, _, outgoing = _category_tables(category)
    endpoint_counts = dict.fromkeys(category.objects, 1)
    sizes = [len(category.objects)]
    if sizes[0] > MAX_SIMPLICES_PER_DEGREE:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="finite_category.nerve_degree_size_budget",
            message=(
                f"nerve degree 0 has {sizes[0]} simplices, exceeding "
                f"the {MAX_SIMPLICES_PER_DEGREE}-simplex bound"
            ),
        )
    for _ in range(degree):
        next_counts = dict.fromkeys(category.objects, 0)
        for obj, count in endpoint_counts.items():
            if count:
                for arrow in outgoing[obj]:
                    target = by_id[arrow].target
                    next_counts[target] += count
        endpoint_counts = next_counts
        count = sum(endpoint_counts.values())
        if count > MAX_SIMPLICES_PER_DEGREE:
            raise OperationResourceAdmissionError(
                location=("max_degree",),
                code="finite_category.nerve_degree_size_budget",
                message=(
                    f"nerve degree {len(sizes)} has {count} simplices, exceeding "
                    f"the {MAX_SIMPLICES_PER_DEGREE}-simplex bound"
                ),
            )
        sizes.append(count)
    if sum(sizes) > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="finite_category.nerve_simplex_budget",
            message="the nerve prefix exceeds the finite simplicial-set simplex bound",
        )
    identity_work = sum(sizes[n] * ((n + 1) * n // 2) for n in range(2, degree + 1))
    identity_work += sum(
        sizes[n] * ((n + 1) * (n + 2) // 2) for n in range(max(0, degree - 1))
    )
    identity_work += sum(sizes[n] * ((n + 1) * (n + 2)) for n in range(degree))
    # Bound the echoed category and retained simplex paths together before
    # constructing levels. JSON string escaping can expand a character to six
    # bytes (\\uXXXX); include structural JSON overhead conservatively.
    # CategoryIdentifier may be a recursively nested pair; ``len`` only
    # counts its outer arity and would under-admit the echoed category.
    identifier_chars = sum(
        _identifier_character_count(value) for value in category.objects
    )
    identifier_chars += sum(
        _identifier_character_count(identifier)
        for morphism in category.morphisms
        for identifier in (morphism.morphism_id, morphism.source, morphism.target)
    )
    identifier_chars += sum(
        _identifier_character_count(identifier)
        for row in category.identities
        for identifier in row
    )
    identifier_chars += sum(
        _identifier_character_count(identifier)
        for row in category.composition
        for identifier in row
    )
    identifier_occurrences = sum(sizes) * (2 * degree + 3)
    identifiers = [*category.objects]
    identifiers.extend(
        identifier
        for morphism in category.morphisms
        for identifier in (morphism.morphism_id, morphism.source, morphism.target)
    )
    identifiers.extend(identifier for row in category.identities for identifier in row)
    identifiers.extend(identifier for row in category.composition for identifier in row)
    max_identifier_chars = max(map(_identifier_character_count, identifiers), default=0)
    identifier_chars += identifier_occurrences * max_identifier_chars
    if (
        identifier_chars * 6
        + 1024 * (len(category.objects) + len(category.morphisms) + sum(sizes))
        > NERVE_TRANSPORT_BYTE_BOUND
    ):
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="finite_category.nerve_identifier_budget",
            message="nerve transport exceeds the canonical output byte bound",
        )
    if identity_work > NERVE_IDENTITY_WORK_BOUND:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="finite_category.nerve_identity_work_budget",
            message="simplicial identity replay exceeds the nerve work bound",
        )


def nerve_prefix(category: FiniteCategory, max_degree: int) -> FiniteCategoryNerve:
    """Construct the nerve prefix and check its simplicial identities once."""
    _admit(category, max_degree)
    by_id, identities, composition, outgoing = _category_tables(category)

    levels: list[
        tuple[
            tuple[tuple[CategoryIdentifier, ...], tuple[CategoryIdentifier, ...]],
            ...,
        ]
    ] = [tuple(((), (obj,)) for obj in category.objects)]
    for _degree in range(max_degree):
        prior = levels[-1]
        by_endpoint: dict[
            CategoryIdentifier,
            list[tuple[tuple[CategoryIdentifier, ...], tuple[CategoryIdentifier, ...]]],
        ] = {obj: [] for obj in category.objects}
        for path, vertices in prior:
            by_endpoint[vertices[-1]].append((path, vertices))
        paths = tuple(
            ((*path, arrow), (*vertices, by_id[arrow].target))
            for obj in category.objects
            for path, vertices in by_endpoint[obj]
            for arrow in outgoing[obj]
        )
        levels.append(paths)

    labels = tuple(
        tuple(f"{degree}:{index}" for index in range(len(level)))
        for degree, level in enumerate(levels)
    )
    indices = [{simplex: i for i, simplex in enumerate(level)} for level in levels]
    face_maps = []
    for degree in range(1, max_degree + 1):
        rows = []
        for face in range(degree + 1):
            images = []
            for path, vertices in levels[degree]:
                if face == 0:
                    image = (path[1:], vertices[1:])
                elif face == degree:
                    image = (path[:-1], vertices[:-1])
                else:
                    image = (
                        (
                            *path[: face - 1],
                            composition[(path[face], path[face - 1])],
                            *path[face + 1 :],
                        ),
                        (*vertices[:face], *vertices[face + 1 :]),
                    )
                images.append(indices[degree - 1][image])
            rows.append(tuple(images))
        face_maps.append(tuple(rows))
    degeneracy_maps = []
    for degree in range(max_degree):
        rows = []
        for position in range(degree + 1):
            images = []
            for path, vertices in levels[degree]:
                vertex = vertices[position]
                image = (
                    (*path[:position], identities[vertex], *path[position:]),
                    (*vertices[: position + 1], vertex, *vertices[position + 1 :]),
                )
                images.append(indices[degree + 1][image])
            rows.append(tuple(images))
        degeneracy_maps.append(tuple(rows))

    checked = from_tables(max_degree, labels, tuple(face_maps), tuple(degeneracy_maps))
    if checked.simplicial_set is None:
        raise RuntimeError(
            "finite category nerve construction violated a simplicial identity"
        )
    simplex_morphisms = tuple(tuple(path for path, _ in level) for level in levels)
    simplex_objects = tuple(
        tuple(vertices for _, vertices in level) for level in levels
    )
    return FiniteCategoryNerve._from_kernel(
        category,
        checked.simplicial_set,
        simplex_morphisms,
        simplex_objects,
    )


__all__ = ["nerve_prefix"]
