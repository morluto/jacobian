"""Finite category operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_categories import operations as native
from jacobian.math.finite_categories._models import (
    CategoryProductRequest,
    CategoryProfileResult,
)
from jacobian.math.finite_categories.values import FiniteCategory, FiniteCategoryProduct


def compute_category_profile(request: FiniteCategory) -> CategoryProfileResult:
    return native.category_profile(request)


def compute_opposite_category(request: FiniteCategory) -> FiniteCategory:
    return native.opposite_category(request)


def compute_category_product(request: CategoryProductRequest) -> FiniteCategoryProduct:
    return native.product(request.left, request.right)


_CATEGORY = {
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

_TERMINAL_CATEGORY = {
    "objects": ["T"],
    "morphisms": [
        {"morphism_id": "id_T", "source": "T", "target": "T"},
    ],
    "identities": [["T", "id_T"]],
    "composition": [["id_T", "id_T", "id_T"]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="finite_category.profile.compute",
        title="Compute the profile of a finite category",
        description="Compute hom-set cardinalities, endomorphism counts, and the "
        "designated identity morphism for each object of a finite category "
        "presented extensionally with identities and a total composition "
        "table.",
        request_type=FiniteCategory,
        result_type=CategoryProfileResult,
        run=compute_category_profile,
        tags=("algebra", "category", "exact"),
        examples=(
            OperationExample(
                name="two_object_category",
                description="Profile a 2-object, 3-morphism category; every morphism "
                "source/target must be a declared object and the category "
                "laws must hold.",
                input={
                    "objects": _CATEGORY["objects"],
                    "morphisms": _CATEGORY["morphisms"],
                    "identities": _CATEGORY["identities"],
                    "composition": _CATEGORY["composition"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_category.opposite.compute",
        title="Compute the opposite category",
        description="Compute the opposite category with all morphism directions reversed "
        "(source and target swapped) and composition order reversed.",
        request_type=FiniteCategory,
        result_type=FiniteCategory,
        run=compute_opposite_category,
        tags=("algebra", "category", "exact"),
        examples=(
            OperationExample(
                name="opposite_of_two_object",
                description="Compute the opposite of a 2-object category; every morphism "
                "source/target must be a declared object and the category "
                "laws must hold.",
                input={
                    "objects": _CATEGORY["objects"],
                    "morphisms": _CATEGORY["morphisms"],
                    "identities": _CATEGORY["identities"],
                    "composition": _CATEGORY["composition"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_category.product.compute",
        title="Compute the product of two finite categories",
        description="Construct the exact Cartesian product category with structural pair "
        "identifiers, componentwise identities and composition, and explicit "
        "left/right projections for every product object and morphism. Object, "
        "morphism, composable-pair, composable-triple, execution-work, identifier, "
        "and canonical-result sizes are all preflight-bounded.",
        request_type=CategoryProductRequest,
        result_type=FiniteCategoryProduct,
        run=compute_category_product,
        tags=("algebra", "category", "product", "exact"),
        examples=(
            OperationExample(
                name="two_object_by_terminal",
                description="Construct a two-object category times the terminal category; "
                "product identifiers are nested JSON pairs, not joined labels.",
                input={
                    "left": _CATEGORY,
                    "right": _TERMINAL_CATEGORY,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
