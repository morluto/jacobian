"""Rational-map composition operation declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.rational_functions.composition._models import (
    RationalFunctionMapComposition,
    RationalMapCompositionRequest,
)
from jacobian.math.polynomials.rational_functions.composition.operations import (
    compose_maps,
)


def _compute(
    request: RationalMapCompositionRequest,
) -> RationalFunctionMapComposition:
    return compose_maps(request.outer, request.inner)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="rational_function_map.compose.compute",
        title="Compose two rational coordinate maps",
        description=(
            "Compose two canonical rational coordinate maps when "
            "inner.target_coordinates equals outer.source_variables. The "
            "result retains both maps, the inner source and outer target axes, and "
            "the complete pre-cancellation construction locus."
        ),
        request_type=RationalMapCompositionRequest,
        result_type=RationalFunctionMapComposition,
        run=_compute,
        tags=("rational-function", "map", "composition", "exact"),
        examples=(
            OperationExample(
                name="two_variable_chart_transition",
                description=(
                    "Compose a rational inner chart with an outer coordinate map "
                    "whose source_variables equal the inner target_coordinates; "
                    "all denominator conditions remain explicit."
                ),
                input={
                    "outer": {
                        "source_variables": ["y1", "y2"],
                        "target_coordinates": ["z1", "z2"],
                        "components": [
                            {
                                "variables": ["y1", "y2"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1, 0],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 1],
                                        },
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 0],
                                        }
                                    ]
                                },
                            },
                            {
                                "variables": ["y1", "y2"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1, 0],
                                        }
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 1],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0, 0],
                                        },
                                    ]
                                },
                            },
                        ],
                    },
                    "inner": {
                        "source_variables": ["x"],
                        "target_coordinates": ["y1", "y2"],
                        "components": [
                            {
                                "variables": ["x"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                            {
                                "variables": ["x"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        },
                                        {
                                            "coefficient": {"num": "-2", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
