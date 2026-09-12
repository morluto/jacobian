"""Certified finite quadratic-surd operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.diophantine_approximation import _surd_kernel as native
from jacobian.math.number_theory.diophantine_approximation._surd_models import (
    NearestIntegerDistanceRequest,
    NearestIntegerDistanceValue,
    RangeProfileRequest,
    RangeProfileResult,
    RecordMinimaRequest,
    RecordMinimaResult,
    ScaledFloorRequest,
    ScaledFloorValue,
    SimultaneousProductRequest,
    SimultaneousProductResult,
)

SURD_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_theory.quadratic_surd.scaled_floor.compute",
        title="Compute the exact floor and ceiling of n*sqrt(d)",
        description=(
            "Return floor(n*sqrt(d)) = isqrt(d*n^2) and its ceiling for a "
            "positive multiplier and a nonsquare integer radicand, together "
            "with the exact endpoint squares that replay the bracket."
        ),
        request_type=ScaledFloorRequest,
        result_type=ScaledFloorValue,
        run=native.scaled_floor,
        tags=("number-theory", "quadratic-surd", "floor", "exact"),
        examples=(
            OperationExample(
                name="three_sqrt_two",
                description=(
                    "For the nonsquare radicand 2, floor(3*sqrt(2)) = 4 and ceiling 5."
                ),
                input={"multiplier": "3", "radicand": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.quadratic_surd.nearest_integer_distance.compute",
        title="Certify the distance from n*sqrt(d) to the nearest integer",
        description=(
            "Return the exact floor, ceiling, nearest integer, and branch of "
            "n*sqrt(d) together with a certified rational enclosure of the "
            "distance ||n*sqrt(d)|| at the requested binary scale. A precision "
            "that cannot separate the two branches is rejected rather than "
            "guessed."
        ),
        request_type=NearestIntegerDistanceRequest,
        result_type=NearestIntegerDistanceValue,
        run=native.nearest_integer_distance,
        tags=("number-theory", "quadratic-surd", "certified", "enclosure"),
        examples=(
            OperationExample(
                name="sqrt_two_distance",
                description=(
                    "For nonsquare radicand 2, ||sqrt(2)|| is enclosed near "
                    "0.414 at 32-bit precision."
                ),
                input={"multiplier": "1", "radicand": 2, "scale_bits": 32},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.simultaneous_approximation.product_enclosure.compute",
        title="Certify n times a product of quadratic-surd distances",
        description=(
            "Return every per-factor nearest-integer row and a certified "
            "rational enclosure of n*prod_i||n*sqrt(d_i)|| over an ordered "
            "radicand axis."
        ),
        request_type=SimultaneousProductRequest,
        result_type=SimultaneousProductResult,
        run=native.simultaneous_product,
        tags=("number-theory", "simultaneous-approximation", "certified"),
        examples=(
            OperationExample(
                name="sqrt_two_sqrt_three",
                description=(
                    "Product factor for distinct nonsquare radicands 2 and 3 at n = 1."
                ),
                input={"multiplier": "1", "radicands": [2, 3], "scale_bits": 32},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.simultaneous_approximation.range_profile.compute",
        title="Return a complete certified simultaneous-approximation range",
        description=(
            "Return one certified row for every integer 1 <= n <= limit, with "
            "floors, nearest integers, per-factor enclosure intervals, and the "
            "product interval. Every row of the declared range is present; the "
            "operation never omits a difficult row silently."
        ),
        request_type=RangeProfileRequest,
        result_type=RangeProfileResult,
        run=native.range_profile,
        tags=("number-theory", "simultaneous-approximation", "range", "certified"),
        examples=(
            OperationExample(
                name="sqrt_two_sqrt_three_to_ten",
                description=(
                    "Complete certified range for distinct nonsquare radicands "
                    "2 and 3 up to 10."
                ),
                input={"radicands": [2, 3], "limit": 10, "scale_bits": 32},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.simultaneous_approximation.record_minima.compute",
        title="Extract strict certified record minima from a finite range",
        description=(
            "Return the strict record-minimum sequence and the finite argmin of "
            "n*prod_i||n*sqrt(d_i)|| over a complete certified range, or an "
            "explicit UNRESOLVED row with both enclosures when the declared "
            "precision cannot strictly separate a comparison."
        ),
        request_type=RecordMinimaRequest,
        result_type=RecordMinimaResult,
        run=native.record_minima,
        tags=("number-theory", "simultaneous-approximation", "records", "certified"),
        examples=(
            OperationExample(
                name="records_sqrt_two_sqrt_three",
                description=(
                    "Strict records for distinct nonsquare radicands 2 and 3 "
                    "with n <= 50."
                ),
                input={"radicands": [2, 3], "limit": 50, "scale_bits": 64},
            ),
        ),
    ),
)
