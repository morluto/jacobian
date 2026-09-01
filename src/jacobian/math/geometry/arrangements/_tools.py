"""Hyperplane arrangement operation declarations."""

from typing import Any

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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="arrangement.construct",
        title="Construct a hyperplane arrangement",
        description="Construct a hyperplane arrangement and check if it is central.",
        request_type=HyperplaneArrangementRequest,
        result_type=HyperplaneArrangementResult,
        run=_run_arrangement,
        tags=("hyperplane", "arrangement", "exact"),
        examples=(
            OperationExample(
                name="central_2d",
                description="Two central hyperplanes in R^2.",
                input={
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
    MathTool(
        operation_id="arrangement.characteristic_polynomial.compute",
        title="Compute the characteristic polynomial of a generic arrangement",
        description="Compute the characteristic polynomial chi(t) of a generic central "
        "hyperplane arrangement using the Zaslavsky formula.",
        request_type=CharacteristicPolynomialRequest,
        result_type=CharacteristicPolynomialResult,
        run=_run_characteristic_polynomial,
        tags=("hyperplane", "characteristic-polynomial", "exact"),
        examples=(
            OperationExample(
                name="generic_2_2",
                description="Characteristic polynomial of 2 hyperplanes in R^2.",
                input={"ambient_dimension": 2, "hyperplane_count": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="arrangement.chamber_count.compute",
        title="Count chambers of a generic central arrangement",
        description="Count the number of chambers (regions) of a generic central "
        "hyperplane arrangement using the central formula 2 * sum C(m-1, k).",
        request_type=ChamberCountRequest,
        result_type=ChamberCountResult,
        run=_run_chamber_count,
        tags=("hyperplane", "chamber-count", "exact"),
        examples=(
            OperationExample(
                name="generic_2_2",
                description="Chamber count of 2 hyperplanes in R^2.",
                input={"ambient_dimension": 2, "hyperplane_count": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
