"""Complete exact incidence materialization for labelled projective lines."""

from __future__ import annotations

from fractions import Fraction
from math import comb, gcd
from typing import cast

from jacobian.contracts.projective_geometry import (
    NormalizedProjectiveLine,
    PrimitiveProjectiveTriple,
    ProjectiveArrangementFlat,
    ProjectiveLineArrangementRequest,
    ProjectiveLineArrangementResult,
    ProjectiveMultiplicityCount,
)
from jacobian.domains._examples import example
from jacobian.operations import (
    MaterializedOperation,
    MaterializedOperationFactory,
    OperationFailure,
)


def _primitive(values: tuple[Fraction, Fraction, Fraction]) -> tuple[int, int, int]:
    denominator_lcm = 1
    for fraction_value in values:
        denominator_lcm = (
            denominator_lcm
            * fraction_value.denominator
            // gcd(denominator_lcm, fraction_value.denominator)
        )
    integers = tuple(
        value.numerator * (denominator_lcm // value.denominator) for value in values
    )
    divisor = 0
    for integer in integers:
        divisor = gcd(divisor, abs(integer))
    if divisor == 0:
        raise ValueError("projective homogeneous coordinates must be nonzero")
    primitive = tuple(value // divisor for value in integers)
    if next(value for value in primitive if value) < 0:
        primitive = tuple(-value for value in primitive)
    return cast(tuple[int, int, int], primitive)


def _cross(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> tuple[int, int, int]:
    return _primitive(
        (
            Fraction(left[1] * right[2] - left[2] * right[1]),
            Fraction(left[2] * right[0] - left[0] * right[2]),
            Fraction(left[0] * right[1] - left[1] * right[0]),
        )
    )


def _wire_triple(values: tuple[int, int, int]) -> PrimitiveProjectiveTriple:
    return PrimitiveProjectiveTriple(
        coordinates=(str(values[0]), str(values[1]), str(values[2]))
    )


def materialize_projective_line_flats(
    request: ProjectiveLineArrangementRequest,
) -> ProjectiveLineArrangementResult:
    normalized = tuple(
        sorted(
            (
                line.label,
                _primitive(
                    cast(
                        tuple[Fraction, Fraction, Fraction],
                        tuple(
                            coefficient.as_fraction()
                            for coefficient in line.coefficients
                        ),
                    )
                ),
            )
            for line in request.lines
        )
    )
    points = {
        _cross(normalized[left][1], normalized[right][1])
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    }
    flats: list[ProjectiveArrangementFlat] = []
    for point in sorted(points):
        incident = tuple(
            label
            for label, coefficients in normalized
            if sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(
                    coefficients,
                    point,
                    strict=True,
                )
            )
            == 0
        )
        multiplicity = len(incident)
        if multiplicity < 2:
            raise RuntimeError("pair intersection lost both incident lines")
        flats.append(
            ProjectiveArrangementFlat(
                point=_wire_triple(point),
                incident_labels=incident,
                multiplicity=multiplicity,
                pair_count=comb(multiplicity, 2),
            )
        )
    histogram: dict[int, int] = {}
    for flat in flats:
        histogram[flat.multiplicity] = histogram.get(flat.multiplicity, 0) + 1
    return ProjectiveLineArrangementResult(
        line_count=len(normalized),
        normalized_lines=tuple(
            NormalizedProjectiveLine(
                label=label,
                coefficients=_wire_triple(coefficients),
            )
            for label, coefficients in normalized
        ),
        flats=tuple(flats),
        non_double_flats=tuple(
            sorted(flat.incident_labels for flat in flats if flat.multiplicity > 2)
        ),
        multiplicity_histogram=tuple(
            ProjectiveMultiplicityCount(
                multiplicity=multiplicity,
                flat_count=count,
            )
            for multiplicity, count in sorted(histogram.items())
        ),
        pair_count_total=comb(len(normalized), 2),
    )


_FACTORY = MaterializedOperationFactory(
    OperationFailure(
        code="PROJECTIVE_ARRANGEMENT_NOT_APPLICABLE",
        stage="projective_arrangement_computation",
        hint=(
            "Supply distinct labelled nonzero rational homogeneous line coefficients."
        ),
        exceptions=(ArithmeticError, RuntimeError, TypeError, ValueError),
    )
)

PROJECTIVE_LINE_ARRANGEMENT_CAPABILITY: MaterializedOperation[
    ProjectiveLineArrangementRequest,
    ProjectiveLineArrangementResult,
    ProjectiveLineArrangementResult,
] = _FACTORY(
    "geometry.projective_line_arrangement.flats.materialize",
    "Materialize projective line-arrangement flats",
    (
        "Normalize labelled rational projective lines and exactly materialize "
        "every rank-two flat, full incidence set, multiplicity, non-double flat, "
        "and line-pair accounting identity."
    ),
    ProjectiveLineArrangementRequest,
    ProjectiveLineArrangementResult,
    materialize_projective_line_flats,
    "geometry",
    "projective",
    "line-arrangement",
    "incidence",
    "flats",
    "exact",
    relation_id="geometry.projective_line_arrangement.flats.relation",
    version="3",
    invocation_examples=(
        example(
            "two_coordinate_lines",
            "Materialize flats for two coordinate lines.",
            {
                "lines": [
                    {
                        "label": "x",
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "label": "y",
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                ]
            },
        ),
    ),
)


__all__ = ["PROJECTIVE_LINE_ARRANGEMENT_CAPABILITY"]
