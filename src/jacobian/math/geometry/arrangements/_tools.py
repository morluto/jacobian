"""Hyperplane arrangement operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.arrangements._models import (
    ChamberCountRequest,
    ChamberCountResult,
    CharacteristicPolynomialRequest,
    CharacteristicPolynomialResult,
    HyperplaneArrangementRequest,
    HyperplaneArrangementResult,
)
from jacobian.math.geometry.arrangements.operations import (
    arrangement,
    chamber_count,
    characteristic_polynomial,
)


def _run_arrangement(
    request: HyperplaneArrangementRequest,
) -> HyperplaneArrangementResult:
    return arrangement(request.ambient_dimension, request.hyperplanes)


def _run_characteristic_polynomial(
    request: CharacteristicPolynomialRequest,
) -> CharacteristicPolynomialResult:
    return characteristic_polynomial(
        request.ambient_dimension, request.hyperplane_count
    )


def _run_chamber_count(request: ChamberCountRequest) -> ChamberCountResult:
    return chamber_count(request.ambient_dimension, request.hyperplane_count)


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
        "arrangement.construct",
        "Construct a hyperplane arrangement",
        "Construct a hyperplane arrangement and check if it is central.",
        HyperplaneArrangementRequest,
        HyperplaneArrangementResult,
        _run_arrangement,
        "hyperplane",
        "arrangement",
        "exact",
        examples=(
            example(
                "central_2d",
                "Two central hyperplanes in R^2.",
                {
                    "ambient_dimension": 2,
                    "hyperplanes": [
                        {
                            "coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "constant": {"num": "0", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "constant": {"num": "0", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
    _op(
        "arrangement.characteristic_polynomial.compute",
        "Compute the characteristic polynomial of a generic arrangement",
        "Compute the characteristic polynomial chi(t) of a generic central "
        "hyperplane arrangement using the Zaslavsky formula.",
        CharacteristicPolynomialRequest,
        CharacteristicPolynomialResult,
        _run_characteristic_polynomial,
        "hyperplane",
        "characteristic-polynomial",
        "exact",
        examples=(
            example(
                "generic_2_2",
                "Characteristic polynomial of 2 hyperplanes in R^2.",
                {"ambient_dimension": 2, "hyperplane_count": 2},
            ),
        ),
    ),
    _op(
        "arrangement.chamber_count.compute",
        "Count chambers of a generic central arrangement",
        "Count the number of chambers (regions) of a generic central "
        "hyperplane arrangement using the central formula 2 * sum C(m-1, k).",
        ChamberCountRequest,
        ChamberCountResult,
        _run_chamber_count,
        "hyperplane",
        "chamber-count",
        "exact",
        examples=(
            example(
                "generic_2_2",
                "Chamber count of 2 hyperplanes in R^2.",
                {"ambient_dimension": 2, "hyperplane_count": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
