"""Tests for finite category operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.finite_categories import (
    FiniteCategory,
    FiniteCategoryProduct,
    product,
)
from jacobian.math.finite_categories._models import (
    CategoryProductRequest,
)
from jacobian.math.finite_categories._operations import (
    compute_category_product,
    compute_category_profile,
    compute_opposite_category,
)
from jacobian.math.finite_categories.operations import _product_plan

CATEGORY = {
    "objects": ["A", "B"],
    "morphisms": [
        {"morphism_id": "id_A", "source": "A", "target": "A"},
        {"morphism_id": "id_B", "source": "B", "target": "B"},
        {"morphism_id": "f", "source": "A", "target": "B"},
    ],
    "identities": [["A", "id_A"], ["B", "id_B"]],
    "composition": [
        ["id_A", "id_A", "id_A"],
        ["f", "id_A", "f"],
        ["id_B", "id_B", "id_B"],
        ["id_B", "f", "f"],
    ],
}

TERMINAL_CATEGORY = {
    "objects": ["T"],
    "morphisms": [
        {"morphism_id": "id_T", "source": "T", "target": "T"},
    ],
    "identities": [["T", "id_T"]],
    "composition": [["id_T", "id_T", "id_T"]],
}


def _discrete_category(size: int, prefix: str) -> FiniteCategory:
    objects = tuple(f"{prefix}{index}" for index in range(size))
    identities = tuple((object_id, f"id_{object_id}") for object_id in objects)
    return FiniteCategory(
        objects=objects,
        morphisms=tuple(
            {
                "morphism_id": identity,
                "source": object_id,
                "target": object_id,
            }
            for object_id, identity in identities
        ),
        identities=identities,
        composition=tuple((identity, identity, identity) for _, identity in identities),
    )


def _cyclic_group_category(order: int) -> FiniteCategory:
    morphism_ids = tuple(f"g{index}" for index in range(order))
    return FiniteCategory(
        objects=("*",),
        morphisms=tuple(
            {"morphism_id": item, "source": "*", "target": "*"} for item in morphism_ids
        ),
        identities=(("*", "g0"),),
        composition=tuple(
            (f"g{left}", f"g{right}", f"g{(left + right) % order}")
            for left in range(order)
            for right in range(order)
        ),
    )


def _parallel_arrow_category(count: int, label_width: int) -> FiniteCategory:
    def label(prefix: str, index: int) -> str:
        head = f"{prefix}{index}_"
        return head + "x" * (label_width - len(head))

    source = label("A", 0)
    target = label("B", 0)
    source_identity = label("ia", 0)
    target_identity = label("ib", 0)
    arrows = tuple(label("f", index) for index in range(count))
    return FiniteCategory(
        objects=(source, target),
        morphisms=(
            {
                "morphism_id": source_identity,
                "source": source,
                "target": source,
            },
            {
                "morphism_id": target_identity,
                "source": target,
                "target": target,
            },
            *(
                {"morphism_id": arrow, "source": source, "target": target}
                for arrow in arrows
            ),
        ),
        identities=((source, source_identity), (target, target_identity)),
        composition=(
            (source_identity, source_identity, source_identity),
            (target_identity, target_identity, target_identity),
            *((arrow, source_identity, arrow) for arrow in arrows),
            *((target_identity, arrow, arrow) for arrow in arrows),
        ),
    )


class TestProfile:
    def test_counts(self) -> None:
        result = compute_category_profile(FiniteCategory(**CATEGORY))
        assert result.num_objects == 2
        assert result.num_morphisms == 3

    def test_hom_sets_are_structural(self) -> None:
        result = compute_category_profile(FiniteCategory(**CATEGORY))
        assert set(result.hom_sets) == {("A", "A", 1), ("A", "B", 1), ("B", "B", 1)}

    def test_endomorphisms(self) -> None:
        result = compute_category_profile(FiniteCategory(**CATEGORY))
        endo = dict(result.endomorphisms)
        assert endo.get("A") == 1
        assert endo.get("B") == 1

    def test_identity_morphisms(self) -> None:
        result = compute_category_profile(FiniteCategory(**CATEGORY))
        ids = dict(result.identity_morphisms)
        assert ids.get("A") == "id_A"
        assert ids.get("B") == "id_B"


class TestOpposite:
    def test_reverses_morphisms(self) -> None:
        result = compute_opposite_category(FiniteCategory(**CATEGORY))
        morph_map = {m.morphism_id: m for m in result.morphisms}
        assert morph_map["f"].source == "B"
        assert morph_map["f"].target == "A"
        assert morph_map["id_A"].source == "A"
        assert morph_map["id_A"].target == "A"

    def test_reverses_composition(self) -> None:
        result = compute_opposite_category(FiniteCategory(**CATEGORY))
        comp = {(g, f): r for (g, f, r) in result.composition}
        # f∘f is not composable, but id_B∘f = f in the source becomes
        # f∘id_B = f in the opposite.
        assert comp[("f", "id_B")] == "f"
        assert comp[("id_A", "f")] == "f"

    def test_opposite_is_a_valid_category(self) -> None:
        result = compute_opposite_category(FiniteCategory(**CATEGORY))
        # The opposite value must itself satisfy the category laws.
        assert set(result.objects) == set(CATEGORY["objects"])
        assert len(result.morphisms) == 3


class TestValidation:
    def test_empty_category_is_a_valid_degenerate_value(self) -> None:
        category = FiniteCategory(
            objects=(), morphisms=(), identities=(), composition=()
        )
        assert category.objects == ()

    def test_extensional_row_order_is_canonicalized(self) -> None:
        shuffled = FiniteCategory(
            objects=tuple(reversed(CATEGORY["objects"])),
            morphisms=tuple(reversed(CATEGORY["morphisms"])),
            identities=tuple(reversed(CATEGORY["identities"])),
            composition=tuple(reversed(CATEGORY["composition"])),
        )
        assert shuffled == FiniteCategory(**CATEGORY)

    def test_duplicate_objects(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A", "A"], morphisms=[], identities=[], composition=[]
            )

        assert (
            error.value.errors()[0]["type"]
            == "finite_category.duplicate_object_identifier"
        )

    def test_invalid_morphism_target(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A"],
                morphisms=[{"morphism_id": "f", "source": "A", "target": "B"}],
                identities=[],
                composition=[],
            )

        assert (
            error.value.errors()[0]["type"]
            == "finite_category.undeclared_morphism_endpoint"
        )

    def test_missing_identity_rejected(self) -> None:
        # No designated identity for object A.
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A"],
                morphisms=[{"morphism_id": "id_A", "source": "A", "target": "A"}],
                identities=[],
                composition=[["id_A", "id_A", "id_A"]],
            )

        assert error.value.errors()[0]["type"] == "finite_category.missing_identity"

    def test_non_endomorphism_identity_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A", "B"],
                morphisms=[
                    {"morphism_id": "id_A", "source": "A", "target": "A"},
                    {"morphism_id": "id_B", "source": "B", "target": "B"},
                    {"morphism_id": "f", "source": "A", "target": "B"},
                ],
                identities=[["A", "f"], ["B", "id_B"]],
                composition=[
                    ["id_A", "id_A", "id_A"],
                    ["f", "id_A", "f"],
                    ["id_B", "id_B", "id_B"],
                    ["id_B", "f", "f"],
                ],
            )

        assert (
            error.value.errors()[0]["type"]
            == "finite_category.non_endomorphism_identity"
        )

    def test_incomplete_composition_rejected(self) -> None:
        # Missing the (f, id_A) composition entry.
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A", "B"],
                morphisms=[
                    {"morphism_id": "id_A", "source": "A", "target": "A"},
                    {"morphism_id": "id_B", "source": "B", "target": "B"},
                    {"morphism_id": "f", "source": "A", "target": "B"},
                ],
                identities=[["A", "id_A"], ["B", "id_B"]],
                composition=[
                    ["id_A", "id_A", "id_A"],
                    ["id_B", "id_B", "id_B"],
                    ["id_B", "f", "f"],
                ],
            )

        assert (
            error.value.errors()[0]["type"]
            == "finite_category.incomplete_composition_table"
        )

    def test_identity_law_violation_rejected(self) -> None:
        # id_B∘g is declared to be f (both A→B), breaking the left identity
        # law id_B∘g = g.
        with pytest.raises(ValidationError) as error:
            FiniteCategory(
                objects=["A", "B"],
                morphisms=[
                    {"morphism_id": "id_A", "source": "A", "target": "A"},
                    {"morphism_id": "id_B", "source": "B", "target": "B"},
                    {"morphism_id": "f", "source": "A", "target": "B"},
                    {"morphism_id": "g", "source": "A", "target": "B"},
                ],
                identities=[["A", "id_A"], ["B", "id_B"]],
                composition=[
                    ["id_A", "id_A", "id_A"],
                    ["f", "id_A", "f"],
                    ["g", "id_A", "g"],
                    ["id_B", "id_B", "id_B"],
                    ["id_B", "f", "f"],
                    ["id_B", "g", "f"],
                ],
            )

        assert error.value.errors()[0]["type"] == "finite_category.left_identity_law"


class TestProduct:
    def test_wire_preflight_matches_canonical_serialization(self) -> None:
        left = FiniteCategory(
            objects=('A"\\é',),
            morphisms=(
                {
                    "morphism_id": 'id_A"\\é',
                    "source": 'A"\\é',
                    "target": 'A"\\é',
                },
            ),
            identities=(('A"\\é', 'id_A"\\é'),),
            composition=((('id_A"\\é'), ('id_A"\\é'), ('id_A"\\é')),),
        )
        right = FiniteCategory(**TERMINAL_CATEGORY)

        result = product(left, right)

        assert _product_plan(left, right).serialized_result_bytes == len(
            encode_strict_json(result.model_dump(mode="json"))
        )

    def test_constructs_structural_pairs_componentwise(self) -> None:
        left = FiniteCategory(**CATEGORY)
        right = FiniteCategory(**TERMINAL_CATEGORY)

        result = compute_category_product(
            CategoryProductRequest(left=left, right=right)
        )

        assert result.product.objects == (("A", "T"), ("B", "T"))
        morphisms = {
            morphism.morphism_id: morphism for morphism in result.product.morphisms
        }
        assert morphisms[("f", "id_T")].source == ("A", "T")
        assert morphisms[("f", "id_T")].target == ("B", "T")
        assert tuple(
            (row.product, row.left, row.right) for row in result.object_projections
        ) == (
            (("A", "T"), "A", "T"),
            (("B", "T"), "B", "T"),
        )

    def test_composition_is_componentwise(self) -> None:
        result = product(FiniteCategory(**CATEGORY), FiniteCategory(**CATEGORY))
        composition = {(g, f): value for g, f, value in result.product.composition}

        assert composition[(("id_B", "id_B"), ("f", "f"))] == ("f", "f")
        assert composition[(("f", "f"), ("id_A", "id_A"))] == ("f", "f")

    def test_product_value_serializes_into_a_later_product_unchanged(self) -> None:
        first = product(FiniteCategory(**CATEGORY), FiniteCategory(**TERMINAL_CATEGORY))
        serialized = first.product.model_dump(mode="json")
        restored = FiniteCategory.model_validate(serialized)

        second = product(restored, FiniteCategory(**TERMINAL_CATEGORY))

        assert second.left == first.product
        assert second.product.objects[0] == (("A", "T"), "T")

    def test_empty_factor_retains_both_sources(self) -> None:
        empty = FiniteCategory(objects=(), morphisms=(), identities=(), composition=())
        terminal = FiniteCategory(**TERMINAL_CATEGORY)

        result = product(empty, terminal)

        assert result.left == empty
        assert result.right == terminal
        assert result.product == empty
        assert result.object_projections == ()
        assert result.morphism_projections == ()

    def test_result_rejects_a_different_valid_category(self) -> None:
        expected = product(
            FiniteCategory(**TERMINAL_CATEGORY),
            FiniteCategory(**TERMINAL_CATEGORY),
        )
        different_terminal = FiniteCategory(
            objects=("other",),
            morphisms=(
                {"morphism_id": "id_other", "source": "other", "target": "other"},
            ),
            identities=(("other", "id_other"),),
            composition=(("id_other", "id_other", "id_other"),),
        )

        with pytest.raises(ValidationError) as error:
            FiniteCategoryProduct(
                left=expected.left,
                right=expected.right,
                product=different_terminal,
                object_projections=expected.object_projections,
                morphism_projections=expected.morphism_projections,
            )

        assert error.value.errors()[0]["type"] == "finite_category.incorrect_product"

    def test_result_rejects_forged_pair_projection(self) -> None:
        result = product(
            FiniteCategory(**TERMINAL_CATEGORY),
            FiniteCategory(**TERMINAL_CATEGORY),
        )
        payload = result.model_dump(mode="json")
        payload["object_projections"][0]["left"] = "not_T"

        with pytest.raises(ValidationError) as error:
            FiniteCategoryProduct.model_validate(payload)

        assert (
            error.value.errors()[0]["type"]
            == "finite_category.incorrect_object_projections"
        )

    def test_object_product_count_is_accepted_at_and_rejected_above_bound(
        self,
    ) -> None:
        accepted = product(_discrete_category(32, "L"), _discrete_category(32, "R"))
        assert len(accepted.product.objects) == 1_024

        with pytest.raises(ValidationError) as error:
            CategoryProductRequest(
                left=_discrete_category(33, "L"),
                right=_discrete_category(32, "R"),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_category.product_object_count_budget"
        )

    def test_composable_triple_count_is_accepted_and_rejected_at_boundary(
        self,
    ) -> None:
        accepted = product(_cyclic_group_category(6), _cyclic_group_category(10))
        assert len(accepted.product.morphisms) == 60

        with pytest.raises(ValidationError) as error:
            CategoryProductRequest(
                left=_cyclic_group_category(7),
                right=_cyclic_group_category(9),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_category.product_composable_triple_budget"
        )

    def test_wire_preflight_is_result_sensitive_to_identifier_size(self) -> None:
        short = _parallel_arrow_category(50, 4)
        CategoryProductRequest(left=short, right=short)

        long = _parallel_arrow_category(50, 64)
        with pytest.raises(ValidationError) as error:
            CategoryProductRequest(left=long, right=long)
        assert (
            error.value.errors()[0]["type"]
            == "finite_category.product_wire_size_budget"
        )

    def test_identifier_nesting_is_bounded_before_another_product(self) -> None:
        terminal = FiniteCategory(**TERMINAL_CATEGORY)
        category = terminal
        for _ in range(8):
            category = product(category, terminal).product

        with pytest.raises(ValueError, match="identifier-depth budget"):
            product(category, terminal)
