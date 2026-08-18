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
}


class TestProfile:
    def test_counts(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        assert result.num_objects == 2
        assert result.num_morphisms == 3

    def test_hom_sets(self) -> None:
        result = compute_category_profile(FiniteCategoryRequest(**CATEGORY))
        hom = dict(result.hom_sets)
        assert hom.get("A->A") == 1
        assert hom.get("A->B") == 1
        assert hom.get("B->B") == 1

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


class TestValidation:
    def test_duplicate_objects(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            FiniteCategoryRequest(objects=["A", "A"], morphisms=[])

    def test_invalid_morphism_target(self) -> None:
        with pytest.raises(ValidationError, match="declared object"):
            FiniteCategoryRequest(
                objects=["A"],
                morphisms=[{"morphism_id": "f", "source": "A", "target": "B"}],
            )
