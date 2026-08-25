"""Exact bounded native constructions for finite categories."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_core import PydanticCustomError

from jacobian.canonical import strict_json_object_size
from jacobian.math.finite_categories.values import (
    MAX_CATEGORY_COMPOSABLE_PAIRS,
    MAX_CATEGORY_COMPOSABLE_TRIPLES,
    MAX_CATEGORY_IDENTIFIER_BYTES,
    MAX_CATEGORY_IDENTIFIER_DEPTH,
    MAX_CATEGORY_IDENTIFIER_LEAVES,
    MAX_CATEGORY_MORPHISMS,
    MAX_CATEGORY_OBJECTS,
    MAX_CATEGORY_VALUE_BYTES,
    CategoryIdentifier,
    FiniteCategory,
    FiniteCategoryProduct,
    MorphismSpec,
    ProductMorphismProjection,
    ProductObjectProjection,
    _category_counts,
    _category_wire_size,
    _identifier_shape,
    _identifier_wire_size,
)

MAX_CATEGORY_PRODUCT_RESULT_BYTES = 8 * 1024 * 1024
MAX_CATEGORY_PRODUCT_REPLAY_STEPS = 1_000_000


def _product_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable request-admission error for category products."""

    return PydanticCustomError(f"finite_category.{reason}", message)


class CategoryProductAdmissionError(ValueError):
    """Native admission failure for category products."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _product_admission_error(
    reason: str, message: str
) -> CategoryProductAdmissionError:
    return CategoryProductAdmissionError(reason, message)


@dataclass(frozen=True, slots=True)
class _CategoryProductPlan:
    object_count: int
    morphism_count: int
    composable_pair_count: int
    composable_triple_count: int
    replay_steps: int
    serialized_result_bytes: int


def _source_identifier_sizes(
    left: FiniteCategory, right: FiniteCategory
) -> dict[CategoryIdentifier, int]:
    identifiers = (
        set(left.objects)
        | {morphism.morphism_id for morphism in left.morphisms}
        | set(right.objects)
        | {morphism.morphism_id for morphism in right.morphisms}
    )
    return {identifier: _identifier_wire_size(identifier) for identifier in identifiers}


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
    if wire_size > MAX_CATEGORY_IDENTIFIER_BYTES:
        raise _product_admission_error(
            "product_identifier_wire_budget",
            f"{label} exceed the structural identifier wire budget",
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


def _product_result_wire_size(
    left: FiniteCategory,
    right: FiniteCategory,
    product_category_size: int,
    identifier_sizes: dict[CategoryIdentifier, int],
) -> int:
    left_object_sizes = tuple(identifier_sizes[item] for item in left.objects)
    right_object_sizes = tuple(identifier_sizes[item] for item in right.objects)
    object_count = len(left.objects) * len(right.objects)
    object_projection_item_bytes = (
        object_count * _object_overhead("product", "left", "right")
        + _cross_pair_bytes(left_object_sizes, right_object_sizes)
        + len(right.objects) * sum(left_object_sizes)
        + len(left.objects) * sum(right_object_sizes)
    )
    left_morphism_sizes = tuple(
        identifier_sizes[item.morphism_id] for item in left.morphisms
    )
    right_morphism_sizes = tuple(
        identifier_sizes[item.morphism_id] for item in right.morphisms
    )
    morphism_count = len(left.morphisms) * len(right.morphisms)
    morphism_projection_item_bytes = (
        morphism_count * _object_overhead("product", "left", "right")
        + _cross_pair_bytes(left_morphism_sizes, right_morphism_sizes)
        + len(right.morphisms) * sum(left_morphism_sizes)
        + len(left.morphisms) * sum(right_morphism_sizes)
    )
    return strict_json_object_size(
        (
            ("left", _category_wire_size(left)),
            ("right", _category_wire_size(right)),
            ("product", product_category_size),
            (
                "object_projections",
                _array_size(object_count, object_projection_item_bytes),
            ),
            (
                "morphism_projections",
                _array_size(morphism_count, morphism_projection_item_bytes),
            ),
        )
    )


def _product_plan(left: FiniteCategory, right: FiniteCategory) -> _CategoryProductPlan:
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

    replay_steps = 2 * (
        5 * object_count
        + 7 * morphism_count
        + 4 * composable_pair_count
        + composable_triple_count
    )
    if replay_steps > MAX_CATEGORY_PRODUCT_REPLAY_STEPS:
        raise _product_admission_error(
            "product_replay_work_budget",
            "product construction exceeds the bounded construction-and-replay work "
            f"budget of {MAX_CATEGORY_PRODUCT_REPLAY_STEPS} steps",
        )

    product_category_size = _product_category_wire_size(left, right, identifier_sizes)
    if product_category_size > MAX_CATEGORY_VALUE_BYTES:
        raise _product_admission_error(
            "product_wire_size_budget",
            "product category exceeds the bounded canonical category wire size of "
            f"{MAX_CATEGORY_VALUE_BYTES} bytes",
        )
    serialized_result_bytes = _product_result_wire_size(
        left, right, product_category_size, identifier_sizes
    )
    if serialized_result_bytes > MAX_CATEGORY_PRODUCT_RESULT_BYTES:
        raise ValueError(
            "product construction exceeds the bounded canonical serialized result "
            f"size of {MAX_CATEGORY_PRODUCT_RESULT_BYTES} bytes"
        )
    return _CategoryProductPlan(
        object_count=object_count,
        morphism_count=morphism_count,
        composable_pair_count=composable_pair_count,
        composable_triple_count=composable_triple_count,
        replay_steps=replay_steps,
        serialized_result_bytes=serialized_result_bytes,
    )


def _product_data(
    left: FiniteCategory, right: FiniteCategory
) -> tuple[
    FiniteCategory,
    tuple[ProductObjectProjection, ...],
    tuple[ProductMorphismProjection, ...],
]:
    _product_plan(left, right)
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
    return FiniteCategoryProduct(
        left=left,
        right=right,
        product=product_category,
        object_projections=object_projections,
        morphism_projections=morphism_projections,
    )


__all__ = ["product"]
