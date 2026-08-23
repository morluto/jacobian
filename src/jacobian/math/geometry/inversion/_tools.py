"""Circle inversion operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)
from jacobian.math.geometry.inversion._operations import compute_circle_inversion


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...],
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "geometry.inversion.circle_inversion.compute",
        "Compute circle inversion of a rational planar point",
        "Given a rational planar center, a positive rational inversion power "
        "(squared radius), and a rational point p ≠ c, return the exact "
        "rational inverted point q = c + (s / ||p-c||²) * (p-c).",
        CircleInversionRequest,
        CircleInversionResult,
        compute_circle_inversion,
        "geometry",
        "inversion",
        "exact",
        examples=(
            example(
                "unit_inversion_at_origin",
                "Unit circle inversion around the origin: B=(4,0) -> (1/4,0); "
                "the point must differ from the inversion center.",
                {
                    "center": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "power": {"num": "1", "den": "1"},
                    "point": {
                        "x": {"num": "4", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
