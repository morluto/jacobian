"""Tests for finite category operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_categories._models import FiniteCategoryRequest
from jacobian.math.finite_categories._operations import (
    compute_category_profile,
    compute_opposite_category,
)

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


class TestProfile:
    def test_counts(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        assert result.num_objects == 2
        assert result.num_morphisms == 3

    def test_hom_sets_are_structural(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        assert set(result.hom_sets) == {("A", "A", 1), ("A", "B", 1), ("B", "B", 1)}

    def test_endomorphisms(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        endo = dict(result.endomorphisms)
        assert endo.get("A") == 1
        assert endo.get("B") == 1

    def test_identity_morphisms(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        ids = dict(result.identity_morphisms)
        assert ids.get("A") == "id_A"
        assert ids.get("B") == "id_B"


class TestOpposite:
    def test_reverses_morphisms(self) -> None:
        result = compute_opposite_category(FiniteCategoryRequest(**CATEGORY))
        morph_map = {m.morphism_id: m for m in result.morphisms}
        assert morph_map["f"].source == "B"
        assert morph_map["f"].target == "A"
        assert morph_map["id_A"].source == "A"
        assert morph_map["id_A"].target == "A"

    def test_reverses_composition(self) -> None:
        result = compute_opposite_category(FiniteCategoryRequest(**CATEGORY))
        comp = {(g, f): r for (g, f, r) in result.composition}
        # f∘f is not composable, but id_B∘f = f in the source becomes
        # f∘id_B = f in the opposite.
        assert comp[("f", "id_B")] == "f"
        assert comp[("id_A", "f")] == "f"

    def test_opposite_is_a_valid_category(self) -> None:
        result = compute_opposite_category(FiniteCategoryRequest(**CATEGORY))
        # The opposite value must itself satisfy the category laws.
        assert set(result.objects) == set(CATEGORY["objects"])
        assert len(result.morphisms) == 3


class TestValidation:
    def test_duplicate_objects(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            FiniteCategoryRequest(
                objects=["A", "A"], morphisms=[], identities=[], composition=[]
            )

    def test_invalid_morphism_target(self) -> None:
        with pytest.raises(ValidationError, match="declared object"):
            FiniteCategoryRequest(
                objects=["A"],
                morphisms=[{"morphism_id": "f", "source": "A", "target": "B"}],
                identities=[],
                composition=[],
            )

    def test_missing_identity_rejected(self) -> None:
        # No designated identity for object A.
        with pytest.raises(ValidationError, match="exactly one identity"):
            FiniteCategoryRequest(
                objects=["A"],
                morphisms=[{"morphism_id": "id_A", "source": "A", "target": "A"}],
                identities=[],
                composition=[["id_A", "id_A", "id_A"]],
            )

    def test_non_endomorphism_identity_rejected(self) -> None:
        with pytest.raises(ValidationError, match="endomorphism"):
            FiniteCategoryRequest(
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

    def test_incomplete_composition_rejected(self) -> None:
        # Missing the (f, id_A) composition entry.
        with pytest.raises(ValidationError, match="composable pairs"):
            FiniteCategoryRequest(
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

    def test_identity_law_violation_rejected(self) -> None:
        # id_B∘g is declared to be f (both A→B), breaking the left identity
        # law id_B∘g = g.
        with pytest.raises(ValidationError, match="left identity law"):
            FiniteCategoryRequest(
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
