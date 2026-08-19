"""Finite category operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_categories._models import (
    CategoryProfileResult,
    FiniteCategoryRequest,
    OppositeCategoryResult,
)
from jacobian.math.finite_categories._operations import (
    compute_category_profile,
    compute_opposite_category,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "finite_category.profile.compute",
        "Compute the profile of a finite category",
        "Compute hom-set cardinalities, endomorphism counts, and the "
        "designated identity morphism for each object of a finite category "
        "presented extensionally with identities and a total composition "
        "table.",
        FiniteCategoryRequest,
        CategoryProfileResult,
        compute_category_profile,
        "algebra",
        "category",
        "exact",
        examples=(
            example(
                "two_object_category",
                "Profile a 2-object, 3-morphism category; every morphism "
                "source/target must be a declared object and the category "
                "laws must hold.",
                {
                    "objects": _CATEGORY["objects"],
                    "morphisms": _CATEGORY["morphisms"],
                    "identities": _CATEGORY["identities"],
                    "composition": _CATEGORY["composition"],
                },
            ),
        ),
    ),
    _op(
        "finite_category.opposite.compute",
        "Compute the opposite category",
        "Compute the opposite category with all morphism directions reversed "
        "(source and target swapped) and composition order reversed.",
        FiniteCategoryRequest,
        OppositeCategoryResult,
        compute_opposite_category,
        "algebra",
        "category",
        "exact",
        examples=(
            example(
                "opposite_of_two_object",
                "Compute the opposite of a 2-object category; every morphism "
                "source/target must be a declared object and the category "
                "laws must hold.",
                {
                    "objects": _CATEGORY["objects"],
                    "morphisms": _CATEGORY["morphisms"],
                    "identities": _CATEGORY["identities"],
                    "composition": _CATEGORY["composition"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
