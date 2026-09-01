"""Owner-local product admission, construction, and claim verification."""

from __future__ import annotations

from jacobian.canonical import strict_json_object_size
from jacobian.math.finite_categories.values import (
    MAX_CATEGORY_COMPOSABLE_PAIRS,
    MAX_CATEGORY_COMPOSABLE_TRIPLES,
    MAX_CATEGORY_IDENTIFIER_CHARACTERS,
    MAX_CATEGORY_IDENTIFIER_DEPTH,
    MAX_CATEGORY_IDENTIFIER_LEAVES,
    MAX_CATEGORY_MORPHISMS,
    MAX_CATEGORY_OBJECTS,
    CategoryIdentifier,
    FiniteCategory,
    FiniteCategoryProduct,
    MorphismSpec,
    ProductMorphismProjection,
    ProductObjectProjection,
    _category_counts,
    _identifier_character_count,
    _identifier_shape,
)

MAX_CATEGORY_PRODUCT_EXECUTION_STEPS = 1_000_000


class CategoryProductAdmissionError(ValueError):
    """Native admission failure for category products."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _product_admission_error(
    reason: str, message: str
) -> CategoryProductAdmissionError:
    return CategoryProductAdmissionError(reason, message)


def _source_identifier_sizes(
    left: FiniteCategory, right: FiniteCategory
) -> dict[CategoryIdentifier, int]:
    identifiers = (
        set(left.objects)
        | {morphism.morphism_id for morphism in left.morphisms}
        | set(right.objects)
        | {morphism.morphism_id for morphism in right.morphisms}
    )
    return {
        identifier: _identifier_character_count(identifier)
        for identifier in identifiers
    }


def _array_size(count: int, item_bytes: int) -> int:
    return 2 + max(count - 1, 0) + item_bytes


def _object_overhead(*field_names: str) -> int:
    return strict_json_object_size(tuple((field_name, 0) for field_name in field_names))


def _cross_pair_bytes(left_sizes: tuple[int, ...], right_sizes: tuple[int, ...]) -> int:
    count = len(left_sizes) * len(right_sizes)
    return (
        len(right_sizes) * sum(left_sizes)
        + len(left_sizes) * sum(right_sizes)
        + 3 * count
    )


def _require_pair_identifier_budget(
    left: tuple[CategoryIdentifier, ...],
    right: tuple[CategoryIdentifier, ...],
    identifier_sizes: dict[CategoryIdentifier, int],
    *,
    label: str,
) -> None:
    if not left or not right:
        return
    left_shapes = tuple(_identifier_shape(identifier) for identifier in left)
    right_shapes = tuple(_identifier_shape(identifier) for identifier in right)
    depth = 1 + max(
        max(shape[0] for shape in left_shapes),
        max(shape[0] for shape in right_shapes),
    )
    leaves = max(shape[1] for shape in left_shapes) + max(
        shape[1] for shape in right_shapes
    )
    wire_size = (
        3
        + max(identifier_sizes[item] for item in left)
        + max(identifier_sizes[item] for item in right)
    )
    if depth > MAX_CATEGORY_IDENTIFIER_DEPTH:
        raise _product_admission_error(
            "product_identifier_depth_budget",
            f"{label} exceed the structural identifier-depth budget",
        )
    if leaves > MAX_CATEGORY_IDENTIFIER_LEAVES:
        raise _product_admission_error(
            "product_identifier_leaf_budget",
            f"{label} exceed the structural identifier-leaf budget",
        )
    if wire_size > MAX_CATEGORY_IDENTIFIER_CHARACTERS:
        raise _product_admission_error(
            "product_identifier_character_budget",
            f"{label} exceed the structural identifier character budget",
        )


def _product_category_wire_size(
    left: FiniteCategory,
    right: FiniteCategory,
    identifier_sizes: dict[CategoryIdentifier, int],
) -> int:
    left_object_sizes = tuple(identifier_sizes[item] for item in left.objects)
    right_object_sizes = tuple(identifier_sizes[item] for item in right.objects)
    object_count = len(left.objects) * len(right.objects)
    object_pair_bytes = _cross_pair_bytes(left_object_sizes, right_object_sizes)

    morphism_count = len(left.morphisms) * len(right.morphisms)
    left_morphism_fields = (
        tuple(identifier_sizes[item.morphism_id] for item in left.morphisms),
        tuple(identifier_sizes[item.source] for item in left.morphisms),
        tuple(identifier_sizes[item.target] for item in left.morphisms),
    )
    right_morphism_fields = (
        tuple(identifier_sizes[item.morphism_id] for item in right.morphisms),
        tuple(identifier_sizes[item.source] for item in right.morphisms),
        tuple(identifier_sizes[item.target] for item in right.morphisms),
    )
    morphism_item_bytes = morphism_count * _object_overhead(
        "morphism_id", "source", "target"
    ) + sum(
        _cross_pair_bytes(left_sizes, right_sizes)
        for left_sizes, right_sizes in zip(
            left_morphism_fields, right_morphism_fields, strict=True
        )
    )

    identity_pair_bytes = _cross_pair_bytes(
        tuple(identifier_sizes[morphism_id] for _, morphism_id in left.identities),
        tuple(identifier_sizes[morphism_id] for _, morphism_id in right.identities),
    )
    identity_item_bytes = 3 * object_count + object_pair_bytes + identity_pair_bytes

    composition_count = len(left.composition) * len(right.composition)
    composition_item_bytes = 4 * composition_count + sum(
        _cross_pair_bytes(
            tuple(identifier_sizes[row[index]] for row in left.composition),
            tuple(identifier_sizes[row[index]] for row in right.composition),
        )
        for index in range(3)
    )
    return strict_json_object_size(
        (
            ("objects", _array_size(object_count, object_pair_bytes)),
            ("morphisms", _array_size(morphism_count, morphism_item_bytes)),
            ("identities", _array_size(object_count, identity_item_bytes)),
            (
                "composition",
                _array_size(composition_count, composition_item_bytes),
            ),
        )
    )


def _admit_product(left: FiniteCategory, right: FiniteCategory) -> None:
    """Admit one product from its structural counts and execution work."""

    object_count = len(left.objects) * len(right.objects)
    morphism_count = len(left.morphisms) * len(right.morphisms)
    composable_pair_count = len(left.composition) * len(right.composition)
    _, left_triples = _category_counts(left.objects, left.morphisms)
    _, right_triples = _category_counts(right.objects, right.morphisms)
    composable_triple_count = left_triples * right_triples

    if object_count > MAX_CATEGORY_OBJECTS:
        raise _product_admission_error(
            "product_object_count_budget",
            "product category exceeds the bounded structural object count of "
            f"{MAX_CATEGORY_OBJECTS}",
        )
    if morphism_count > MAX_CATEGORY_MORPHISMS:
        raise _product_admission_error(
            "product_morphism_count_budget",
            "product category exceeds the bounded structural morphism count of "
            f"{MAX_CATEGORY_MORPHISMS}",
        )
    if composable_pair_count > MAX_CATEGORY_COMPOSABLE_PAIRS:
        raise _product_admission_error(
            "product_composable_pair_budget",
            "product category exceeds the bounded composable-pair count of "
            f"{MAX_CATEGORY_COMPOSABLE_PAIRS}",
        )
    if composable_triple_count > MAX_CATEGORY_COMPOSABLE_TRIPLES:
        raise _product_admission_error(
            "product_composable_triple_budget",
            "product category exceeds the bounded composable-triple count of "
            f"{MAX_CATEGORY_COMPOSABLE_TRIPLES}",
        )

    identifier_sizes = _source_identifier_sizes(left, right)
    _require_pair_identifier_budget(
        left.objects,
        right.objects,
        identifier_sizes,
        label="product object identifiers",
    )
    _require_pair_identifier_budget(
        tuple(morphism.morphism_id for morphism in left.morphisms),
        tuple(morphism.morphism_id for morphism in right.morphisms),
        identifier_sizes,
        label="product morphism identifiers",
    )

    # Product rows are materialized once and then scanned once by the canonical
    # FiniteCategory constructor to establish the category laws.
    execution_steps = 2 * (
        5 * object_count
        + 7 * morphism_count
        + 4 * composable_pair_count
        + composable_triple_count
    )
    if execution_steps > MAX_CATEGORY_PRODUCT_EXECUTION_STEPS:
        raise _product_admission_error(
            "product_execution_work_budget",
            "product construction exceeds the bounded execution work budget of "
            f"{MAX_CATEGORY_PRODUCT_EXECUTION_STEPS} steps",
        )

def _product_data(
    left: FiniteCategory, right: FiniteCategory
) -> tuple[
    FiniteCategory,
    tuple[ProductObjectProjection, ...],
    tuple[ProductMorphismProjection, ...],
]:
    _admit_product(left, right)
    objects = tuple(
        (left_object, right_object)
        for left_object in left.objects
        for right_object in right.objects
    )
    morphisms = tuple(
        MorphismSpec(
            morphism_id=(left_morphism.morphism_id, right_morphism.morphism_id),
            source=(left_morphism.source, right_morphism.source),
            target=(left_morphism.target, right_morphism.target),
        )
        for left_morphism in left.morphisms
        for right_morphism in right.morphisms
    )
    identities = tuple(
        (
            (left_object, right_object),
            (left_identity, right_identity),
        )
        for left_object, left_identity in left.identities
        for right_object, right_identity in right.identities
    )
    composition = tuple(
        (
            (left_g, right_g),
            (left_f, right_f),
            (left_result, right_result),
        )
        for left_g, left_f, left_result in left.composition
        for right_g, right_f, right_result in right.composition
    )
    product_category = FiniteCategory(
        objects=objects,
        morphisms=morphisms,
        identities=identities,
        composition=composition,
    )
    object_projections = tuple(
        ProductObjectProjection(
            product=(left_object, right_object),
            left=left_object,
            right=right_object,
        )
        for left_object in left.objects
        for right_object in right.objects
    )
    morphism_projections = tuple(
        ProductMorphismProjection(
            product=(left_morphism.morphism_id, right_morphism.morphism_id),
            left=left_morphism.morphism_id,
            right=right_morphism.morphism_id,
        )
        for left_morphism in left.morphisms
        for right_morphism in right.morphisms
    )
    return product_category, object_projections, morphism_projections


def product(left: FiniteCategory, right: FiniteCategory) -> FiniteCategoryProduct:
    """Return the exact Cartesian product of two finite categories."""

    product_category, object_projections, morphism_projections = _product_data(
        left, right
    )
    return FiniteCategoryProduct._from_kernel(
        left=left,
        right=right,
        product=product_category,
        object_projections=object_projections,
        morphism_projections=morphism_projections,
    )


__all__ = ["product"]
