"""Complete exact incidence materialization for labelled projective lines."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import cast

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.projective_geometry import (
    NormalizedProjectiveLine,
    PrimitiveProjectiveTriple,
    ProjectiveArrangementFlat,
    ProjectiveLineArrangementRequest,
    ProjectiveLineArrangementResult,
    ProjectiveMultiplicityCount,
)
from jacobian.domains._examples import example
from jacobian.math.arithmetic import primitive_integer_vector
from jacobian.math_tools import MathTool


def _primitive(values: tuple[Fraction, Fraction, Fraction]) -> tuple[int, int, int]:
    try:
        primitive = primitive_integer_vector(values)
    except ValueError as exc:
        raise ValueError("projective homogeneous coordinates must be nonzero") from exc
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
        coordinates=(
            format_canonical_integer(values[0]),
            format_canonical_integer(values[1]),
            format_canonical_integer(values[2]),
        )
    )


def compute_projective_line_flats(
    request: ProjectiveLineArrangementRequest,
) -> ProjectiveLineArrangementResult:
    """Compute the complete exact flat lattice for one labelled arrangement."""

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


PROJECTIVE_LINE_ARRANGEMENT_OPERATION: MathTool[
    ProjectiveLineArrangementRequest,
    ProjectiveLineArrangementResult,
] = MathTool(
    operation_id="geometry.projective_line_arrangement.flats.compute",
    version="5",
    title="Compute projective line-arrangement flats",
    description=(
        "Normalize labelled rational projective lines and exactly compute "
        "every rank-two flat, full incidence set, multiplicity, non-double flat, "
        "and line-pair accounting identity."
    ),
    request_type=ProjectiveLineArrangementRequest,
    result_type=ProjectiveLineArrangementResult,
    run=compute_projective_line_flats,
    tags=(
        "geometry",
        "projective",
        "line-arrangement",
        "incidence",
        "flats",
        "exact",
    ),
    examples=(
        example(
            "two_coordinate_lines",
            "Compute flats for two coordinate lines.",
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


__all__ = [
    "PROJECTIVE_LINE_ARRANGEMENT_OPERATION",
    "compute_projective_line_flats",
]
