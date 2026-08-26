"""Canonical values for exact bounded finite categories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, strict_json_object_size
from jacobian.math._labels import OpaqueLabel

MAX_CATEGORY_OBJECTS = 1_024
MAX_CATEGORY_MORPHISMS = 4_096
MAX_CATEGORY_COMPOSABLE_PAIRS = 50_000
MAX_CATEGORY_COMPOSABLE_TRIPLES = 250_000
MAX_CATEGORY_VALUE_BYTES = 4 * 1024 * 1024
MAX_CATEGORY_IDENTIFIER_DEPTH = 8
MAX_CATEGORY_IDENTIFIER_LEAVES = 256
MAX_CATEGORY_IDENTIFIER_BYTES = 4_096


type CategoryIdentifier = OpaqueLabel | tuple[CategoryIdentifier, CategoryIdentifier]


def _category_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-category values."""

    return PydanticCustomError(f"finite_category.{reason}", message)


def _identifier_sort_key(identifier: CategoryIdentifier) -> tuple[object, ...]:
    if isinstance(identifier, str):
        return (0, identifier)
    return (
        1,
        _identifier_sort_key(identifier[0]),
        _identifier_sort_key(identifier[1]),
    )


def _identifier_wire_size(identifier: CategoryIdentifier) -> int:
    if isinstance(identifier, str):
        return len(encode_strict_json(identifier))
    return (
        3 + _identifier_wire_size(identifier[0]) + _identifier_wire_size(identifier[1])
    )


def _identifier_shape(identifier: CategoryIdentifier) -> tuple[int, int]:
    """Return ``(structural depth, atomic-leaf count)`` for one identifier."""

    if isinstance(identifier, str):
        return 0, 1
    left_depth, left_leaves = _identifier_shape(identifier[0])
    right_depth, right_leaves = _identifier_shape(identifier[1])
    return 1 + max(left_depth, right_depth), left_leaves + right_leaves


def _require_identifier_budget(identifier: CategoryIdentifier) -> None:
    depth, leaves = _identifier_shape(identifier)
    if depth > MAX_CATEGORY_IDENTIFIER_DEPTH:
        raise _category_error(
            "identifier_depth_budget",
            "category identifier exceeds the bounded structural-depth budget",
        )
    if leaves > MAX_CATEGORY_IDENTIFIER_LEAVES:
        raise _category_error(
            "identifier_leaf_budget",
            "category identifier exceeds the bounded atomic-leaf budget",
        )
    if _identifier_wire_size(identifier) > MAX_CATEGORY_IDENTIFIER_BYTES:
        raise _category_error(
            "identifier_wire_size_budget",
            "category identifier exceeds the bounded wire-size budget",
        )


def _json_array_size(item_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


class MorphismSpec(StrictModel):
    """One morphism with a structural identifier and typed endpoints."""

    morphism_id: CategoryIdentifier
    source: CategoryIdentifier
    target: CategoryIdentifier


def _category_counts(
    objects: tuple[CategoryIdentifier, ...],
    morphisms: tuple[MorphismSpec, ...],
) -> tuple[int, int]:
    by_source = dict.fromkeys(objects, 0)
    by_target = dict.fromkeys(objects, 0)
    for morphism in morphisms:
        if morphism.source in by_source:
            by_source[morphism.source] += 1
        if morphism.target in by_target:
            by_target[morphism.target] += 1
    composable_pairs = sum(
        by_target[object_id] * by_source[object_id] for object_id in objects
    )
    composable_triples = sum(
        by_target.get(morphism.source, 0) * by_source.get(morphism.target, 0)
        for morphism in morphisms
    )
    return composable_pairs, composable_triples


def _morphism_index(
    objects: tuple[CategoryIdentifier, ...], morphisms: tuple[MorphismSpec, ...]
) -> dict[CategoryIdentifier, MorphismSpec]:
    object_set = set(objects)
    if len(object_set) != len(objects):
        raise _category_error(
            "duplicate_object_identifier", "object identifiers must be distinct"
        )
    by_id = {morphism.morphism_id: morphism for morphism in morphisms}
    if len(by_id) != len(morphisms):
        raise _category_error(
            "duplicate_morphism_identifier", "morphism identifiers must be distinct"
        )
    for morphism in morphisms:
        if morphism.source not in object_set or morphism.target not in object_set:
            raise _category_error(
                "undeclared_morphism_endpoint",
                "every morphism source/target must be a declared object",
            )
    return by_id


def _identity_map(
    objects: tuple[CategoryIdentifier, ...],
    by_id: dict[CategoryIdentifier, MorphismSpec],
    identities: tuple[tuple[CategoryIdentifier, CategoryIdentifier], ...],
) -> dict[CategoryIdentifier, CategoryIdentifier]:
    object_set = set(objects)
    identity_map: dict[CategoryIdentifier, CategoryIdentifier] = {}
    for object_id, morphism_id in identities:
        if object_id not in object_set:
            raise _category_error(
                "undeclared_identity_object",
                "identity objects must be declared objects",
            )
        if object_id in identity_map:
            raise _category_error(
                "duplicate_identity", "each object has exactly one identity"
            )
        morphism = by_id.get(morphism_id)
        if morphism is None:
            raise _category_error(
                "undeclared_identity_morphism",
                "an identity must name a declared morphism",
            )
        if morphism.source != object_id or morphism.target != object_id:
            raise _category_error(
                "non_endomorphism_identity",
                "an identity must be an endomorphism of its object",
            )
        identity_map[object_id] = morphism_id
    if set(identity_map) != object_set:
        raise _category_error(
            "missing_identity", "every object must have exactly one identity"
        )
    return identity_map


def _composition_table(
    morphisms: tuple[MorphismSpec, ...],
    by_id: dict[CategoryIdentifier, MorphismSpec],
    composition: tuple[
        tuple[CategoryIdentifier, CategoryIdentifier, CategoryIdentifier], ...
    ],
) -> dict[tuple[CategoryIdentifier, CategoryIdentifier], CategoryIdentifier]:
    table: dict[tuple[CategoryIdentifier, CategoryIdentifier], CategoryIdentifier] = {}
    for g, f, result in composition:
        if g not in by_id or f not in by_id or result not in by_id:
            raise _category_error(
                "undeclared_composition_morphism",
                "composition must name declared morphisms",
            )
        f_spec = by_id[f]
        g_spec = by_id[g]
        result_spec = by_id[result]
        if f_spec.target != g_spec.source:
            raise _category_error(
                "noncomposable_pair", "composition requires target(f) == source(g)"
            )
        if result_spec.source != f_spec.source or result_spec.target != g_spec.target:
            raise _category_error(
                "composition_result_endpoint",
                "composition result must have source(f) and target(g)",
            )
        if (g, f) in table:
            raise _category_error(
                "duplicate_composition_pair",
                "composition must be total and single-valued",
            )
        table[(g, f)] = result

    by_source: dict[CategoryIdentifier, list[CategoryIdentifier]] = {}
    by_target: dict[CategoryIdentifier, list[CategoryIdentifier]] = {}
    for morphism in morphisms:
        by_source.setdefault(morphism.source, []).append(morphism.morphism_id)
        by_target.setdefault(morphism.target, []).append(morphism.morphism_id)
    composable = {
        (g, f)
        for object_id in set(by_source) | set(by_target)
        for f in by_target.get(object_id, ())
        for g in by_source.get(object_id, ())
    }
    if set(table) != composable:
        raise _category_error(
            "incomplete_composition_table",
            "composition table domain must be exactly the composable pairs",
        )
    return table


def _check_unit_laws(
    morphisms: tuple[MorphismSpec, ...],
    identities: dict[CategoryIdentifier, CategoryIdentifier],
    composition: dict[
        tuple[CategoryIdentifier, CategoryIdentifier], CategoryIdentifier
    ],
) -> None:
    for morphism in morphisms:
        identity_target = identities[morphism.target]
        identity_source = identities[morphism.source]
        if composition[(identity_target, morphism.morphism_id)] != morphism.morphism_id:
            raise _category_error("left_identity_law", "left identity law violated")
        if composition[(morphism.morphism_id, identity_source)] != morphism.morphism_id:
            raise _category_error("right_identity_law", "right identity law violated")


def _check_associativity(
    morphisms: tuple[MorphismSpec, ...],
    composition: dict[
        tuple[CategoryIdentifier, CategoryIdentifier], CategoryIdentifier
    ],
) -> None:
    by_source: dict[CategoryIdentifier, list[CategoryIdentifier]] = {}
    by_target: dict[CategoryIdentifier, list[CategoryIdentifier]] = {}
    for morphism in morphisms:
        by_source.setdefault(morphism.source, []).append(morphism.morphism_id)
        by_target.setdefault(morphism.target, []).append(morphism.morphism_id)
    for g_spec in morphisms:
        g = g_spec.morphism_id
        for f in by_target.get(g_spec.source, ()):
            gf = composition[(g, f)]
            for h in by_source.get(g_spec.target, ()):
                hg = composition[(h, g)]
                if composition[(hg, f)] != composition[(h, gf)]:
                    raise _category_error("associativity", "associativity violated")


def _morphism_wire_size(
    morphism: MorphismSpec,
    identifier_sizes: Mapping[CategoryIdentifier, int],
) -> int:
    return strict_json_object_size(
        (
            ("morphism_id", identifier_sizes[morphism.morphism_id]),
            ("source", identifier_sizes[morphism.source]),
            ("target", identifier_sizes[morphism.target]),
        )
    )


def _category_wire_size(category: FiniteCategory) -> int:
    identifiers = set(category.objects) | {
        morphism.morphism_id for morphism in category.morphisms
    }
    identifier_sizes = {
        identifier: _identifier_wire_size(identifier) for identifier in identifiers
    }
    return strict_json_object_size(
        (
            (
                "objects",
                _json_array_size(
                    tuple(identifier_sizes[item] for item in category.objects)
                ),
            ),
            (
                "morphisms",
                _json_array_size(
                    tuple(
                        _morphism_wire_size(item, identifier_sizes)
                        for item in category.morphisms
                    )
                ),
            ),
            (
                "identities",
                _json_array_size(
                    tuple(
                        _json_array_size(
                            (
                                identifier_sizes[object_id],
                                identifier_sizes[morphism_id],
                            )
                        )
                        for object_id, morphism_id in category.identities
                    )
                ),
            ),
            (
                "composition",
                _json_array_size(
                    tuple(
                        _json_array_size(tuple(identifier_sizes[item] for item in row))
                        for row in category.composition
                    )
                ),
            ),
        )
    )


class FiniteCategory(StrictModel):
    """A canonical finite category presented by complete extensional tables.

    Atomic identifiers are opaque labels. A product construction represents
    identifiers by structural nested pairs, so products compose without a
    caller inventing or decoding concatenated names. Composition rows are
    ``(g, f, result)`` and mean ``g ∘ f = result``.
    """

    objects: tuple[CategoryIdentifier, ...] = Field(max_length=MAX_CATEGORY_OBJECTS)
    morphisms: tuple[MorphismSpec, ...] = Field(max_length=MAX_CATEGORY_MORPHISMS)
    identities: tuple[tuple[CategoryIdentifier, CategoryIdentifier], ...] = Field(
        max_length=MAX_CATEGORY_OBJECTS
    )
    composition: tuple[
        tuple[CategoryIdentifier, CategoryIdentifier, CategoryIdentifier], ...
    ] = Field(max_length=MAX_CATEGORY_COMPOSABLE_PAIRS)

    @model_validator(mode="after")
    def require_canonical_bounded_category(self) -> Self:
        identifiers = (
            tuple(self.objects)
            + tuple(
                item
                for morphism in self.morphisms
                for item in (morphism.morphism_id, morphism.source, morphism.target)
            )
            + tuple(item for row in self.identities for item in row)
            + tuple(item for row in self.composition for item in row)
        )
        for identifier in set(identifiers):
            _require_identifier_budget(identifier)

        objects = tuple(sorted(self.objects, key=_identifier_sort_key))
        morphisms = tuple(
            sorted(
                self.morphisms,
                key=lambda item: _identifier_sort_key(item.morphism_id),
            )
        )
        identities = tuple(
            sorted(self.identities, key=lambda row: _identifier_sort_key(row[0]))
        )
        composition = tuple(
            sorted(
                self.composition,
                key=lambda row: (
                    _identifier_sort_key(row[0]),
                    _identifier_sort_key(row[1]),
                    _identifier_sort_key(row[2]),
                ),
            )
        )
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "morphisms", morphisms)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "composition", composition)

        by_id = _morphism_index(objects, morphisms)
        composable_pairs, composable_triples = _category_counts(objects, morphisms)
        if composable_pairs > MAX_CATEGORY_COMPOSABLE_PAIRS:
            raise _category_error(
                "composable_pair_budget",
                "category exceeds the bounded composable-pair work budget",
            )
        if composable_triples > MAX_CATEGORY_COMPOSABLE_TRIPLES:
            raise _category_error(
                "composable_triple_budget",
                "category exceeds the bounded composable-triple work budget",
            )
        identity_map = _identity_map(objects, by_id, identities)
        composition_table = _composition_table(morphisms, by_id, composition)
        _check_unit_laws(morphisms, identity_map, composition_table)
        _check_associativity(morphisms, composition_table)
        if _category_wire_size(self) > MAX_CATEGORY_VALUE_BYTES:
            raise _category_error(
                "wire_size_budget",
                "category exceeds the bounded canonical wire-size budget",
            )
        return self


class ProductObjectProjection(StrictModel):
    """The two component projections of one product-category object."""

    product: CategoryIdentifier
    left: CategoryIdentifier
    right: CategoryIdentifier


class ProductMorphismProjection(StrictModel):
    """The two component projections of one product-category morphism."""

    product: CategoryIdentifier
    left: CategoryIdentifier
    right: CategoryIdentifier


class FiniteCategoryProduct(StrictModel):
    """A bounded product-category claim with its pair projections.

    Deserialization checks only the structural envelope.  The product law is
    available through the owner-local explicit verifier for independently
    supplied claims; kernel output is created by ``_from_kernel``.
    """

    left: FiniteCategory
    right: FiniteCategory
    product: FiniteCategory
    object_projections: tuple[ProductObjectProjection, ...] = Field(
        max_length=MAX_CATEGORY_OBJECTS
    )
    morphism_projections: tuple[ProductMorphismProjection, ...] = Field(
        max_length=MAX_CATEGORY_MORPHISMS
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: FiniteCategory,
        right: FiniteCategory,
        product: FiniteCategory,
        object_projections: tuple[ProductObjectProjection, ...],
        morphism_projections: tuple[ProductMorphismProjection, ...],
    ) -> Self:
        """Build a product result from the trusted owner-local kernel."""

        return cls(
            left=left,
            right=right,
            product=product,
            object_projections=object_projections,
            morphism_projections=morphism_projections,
        )


__all__ = [
    "MAX_CATEGORY_COMPOSABLE_PAIRS",
    "MAX_CATEGORY_COMPOSABLE_TRIPLES",
    "MAX_CATEGORY_IDENTIFIER_BYTES",
    "MAX_CATEGORY_IDENTIFIER_DEPTH",
    "MAX_CATEGORY_IDENTIFIER_LEAVES",
    "MAX_CATEGORY_MORPHISMS",
    "MAX_CATEGORY_OBJECTS",
    "MAX_CATEGORY_VALUE_BYTES",
    "CategoryIdentifier",
    "FiniteCategory",
    "FiniteCategoryProduct",
    "MorphismSpec",
    "ProductMorphismProjection",
    "ProductObjectProjection",
]
