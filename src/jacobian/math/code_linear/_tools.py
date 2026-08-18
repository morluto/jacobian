"""Code linear operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.code_linear._models import (
    DualCodeResult,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    MacWilliamsResult,
    PunctureRequest,
    PunctureResult,
)
from jacobian.math.code_linear._operations import (
    compute_dual_code,
    compute_macwilliams_transform,
    compute_puncture,
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "code.linear.dual.compute",
        "Compute the dual code of a linear code",
        "Compute the exact dual code C^perp as a generator matrix, returning dual dimension and length.",
        GeneratorMatrixRequest,
        DualCodeResult,
        compute_dual_code,
        "code",
        "dual",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Dual of the binary repetition code of length two.",
                {"field_order": 2, "generator_matrix": [[1, 1]]},
            ),
        ),
    ),
    _op(
        "code.linear.macwilliams_transform.compute",
        "MacWilliams transform of a weight distribution",
        "Apply the q-ary MacWilliams identity to compute the dual code weight distribution.",
        MacWilliamsRequest,
        MacWilliamsResult,
        compute_macwilliams_transform,
        "code",
        "macwilliams",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "MacWilliams transform of the binary length-2 repetition code.",
                {
                    "field_order": 2,
                    "code_cardinality": 2,
                    "length": 2,
                    "weights": [1, 0, 1],
                },
            ),
        ),
    ),
    _op(
        "code.linear.puncture.compute",
        "Puncture a linear code at one coordinate",
        "Delete one coordinate from the generator matrix and return the punctured code.",
        PunctureRequest,
        PunctureResult,
        compute_puncture,
        "code",
        "puncture",
        "exact",
        examples=(
            example(
                "binary_repetition",
                "Puncture the binary length-2 repetition code at coordinate 0.",
                {"field_order": 2, "generator_matrix": [[1, 1]], "coordinate": 0},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
