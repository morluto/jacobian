"""Exact contract tests for rational polygon visibility kernels."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.geometry.polygon_kernel._models import (
    MAX_KERNEL_COORDINATE_DIGITS,
    MAX_KERNEL_SOURCE_VERTICES,
    KernelPolygon,
    PolygonKernelRequest,
    PolygonKernelResult,
)
from jacobian.math.geometry.polygon_kernel._operations import (
    compute_visibility_kernel,
)


def _point(x: int | Fraction, y: int | Fraction) -> dict[str, object]:
    x_value, y_value = Fraction(x), Fraction(y)
    return {
        "x": {"num": str(x_value.numerator), "den": str(x_value.denominator)},
        "y": {"num": str(y_value.numerator), "den": str(y_value.denominator)},
    }


def _request(
    points: Sequence[tuple[int | Fraction, int | Fraction]],
) -> PolygonKernelRequest:
    return PolygonKernelRequest(
        polygon=KernelPolygon.model_validate(
            {"points": [_point(x, y) for x, y in points]}
        )
    )


def _kernel_points(
    result: PolygonKernelResult,
) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple(
        (row.point.x.as_fraction(), row.point.y.as_fraction())
        for row in result.kernel_boundary
    )


PUBLISHED_PENTAGON = [
    (0, 4620),
    (0, -4620),
    (23100, -385),
    (22176, 0),
    (23100, 385),
]


def _cross(
    origin: tuple[Fraction, Fraction],
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
) -> Fraction:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _canonical_hull(
    points: list[tuple[Fraction, Fraction]],
) -> tuple[tuple[Fraction, Fraction], ...]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return tuple(ordered)
    lower: list[tuple[Fraction, Fraction]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _half_plane_value(
    coefficients: tuple[Fraction, Fraction, Fraction],
    point: tuple[Fraction, Fraction],
) -> Fraction:
    a, b, c = coefficients
    return a * point[0] + b * point[1] + c


def _clip_kernel_oracle(
    source: Sequence[tuple[int, int]],
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Independent exact sequential half-plane clipping oracle for tests."""

    points = [(Fraction(x), Fraction(y)) for x, y in source]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    clipped = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        coefficients = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

        output: list[tuple[Fraction, Fraction]] = []
        for previous, current in zip(clipped[-1:] + clipped[:-1], clipped, strict=True):
            previous_value = _half_plane_value(coefficients, previous)
            current_value = _half_plane_value(coefficients, current)
            if current_value >= 0:
                if previous_value < 0:
                    parameter = previous_value / (previous_value - current_value)
                    output.append(
                        (
                            previous[0] + parameter * (current[0] - previous[0]),
                            previous[1] + parameter * (current[1] - previous[1]),
                        )
                    )
                output.append(current)
            elif previous_value >= 0:
                parameter = previous_value / (previous_value - current_value)
                output.append(
                    (
                        previous[0] + parameter * (current[0] - previous[0]),
                        previous[1] + parameter * (current[1] - previous[1]),
                    )
                )
        clipped = output
        if not clipped:
            break
    return _canonical_hull(clipped)


def test_published_pentagon_reconstructs_kernel_and_exact_area_profile() -> None:
    # Nakano, arXiv:2606.05052v2, Section 4 and Appendix B.
    result = compute_visibility_kernel(_request(PUBLISHED_PENTAGON))

    # For the downward first edge, the CCW interior is to its left: x >= 0.
    # Keeping the edge-derived scale makes the orientation trap explicit.
    first = result.half_planes[0]
    assert (
        first.a.as_fraction(),
        first.b.as_fraction(),
        first.c.as_fraction(),
    ) == (Fraction(9240), Fraction(0), Fraction(0))
    assert result.interior_half_plane_convention == "a*x+b*y+c>=0"

    assert tuple(row.cross.as_fraction() for row in result.vertex_turns) == (
        Fraction(213444000),
        Fraction(213444000),
        Fraction(12806640),
        Fraction(-711480),
        Fraction(12806640),
    )
    assert result.reflex_vertex_indices == (3,)
    assert result.kernel_dimension == "POLYGON"
    assert _kernel_points(result) == (
        (Fraction(0), Fraction(-4620)),
        (Fraction(19800), Fraction(-990)),
        (Fraction(22176), Fraction(0)),
        (Fraction(19800), Fraction(990)),
        (Fraction(0), Fraction(4620)),
    )
    assert result.polygon_area.as_fraction() == 115259760
    assert result.kernel_area.as_fraction() == 113430240
    assert result.convex_hull_area.as_fraction() == 115615500
    assert result.kernel_to_polygon_area_ratio.as_fraction() == Fraction(62, 63)
    assert result.polygon_to_hull_area_ratio.as_fraction() == Fraction(324, 325)


@pytest.mark.parametrize(
    "points",
    [
        PUBLISHED_PENTAGON,
        [(0, 0), (6, 0), (6, 5), (4, 3), (2, 5), (0, 5)],
        [(0, 0), (4, 0), (4, 4), (0, 4)],
    ],
)
def test_pairwise_kernel_matches_independent_sequential_clipping_oracle(
    points: list[tuple[int, int]],
) -> None:
    result = compute_visibility_kernel(_request(points))
    assert _kernel_points(result) == _clip_kernel_oracle(points)


@pytest.mark.parametrize(
    ("dimension", "points", "expected_boundary"),
    [
        (
            "EMPTY",
            [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)],
            (),
        ),
        (
            "POINT",
            [
                (-5, -5),
                (-4, -5),
                (-3, -4),
                (-2, -5),
                (5, -5),
                (5, 5),
                (-1, 5),
                (-2, -3),
                (-4, 5),
                (-5, 5),
            ],
            ((Fraction(-2), Fraction(-3)),),
        ),
        (
            "SEGMENT",
            [
                (-5, -5),
                (-4, -5),
                (-3, -3),
                (-2, -5),
                (5, -5),
                (5, 5),
                (1, 5),
                (-2, -1),
                (-4, 5),
                (-5, 5),
            ],
            (
                (Fraction(-3), Fraction(-3)),
                (Fraction(-2), Fraction(-1)),
            ),
        ),
        (
            "POLYGON",
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            (
                (Fraction(0), Fraction(0)),
                (Fraction(4), Fraction(0)),
                (Fraction(4), Fraction(4)),
                (Fraction(0), Fraction(4)),
            ),
        ),
    ],
)
def test_all_kernel_dimensions_are_distinct_and_complete(
    dimension: str,
    points: list[tuple[int, int]],
    expected_boundary: tuple[tuple[Fraction, Fraction], ...],
) -> None:
    result = compute_visibility_kernel(_request(points))
    assert result.kernel_dimension == dimension
    assert _kernel_points(result) == expected_boundary
    assert result.kernel_area.as_fraction() == (
        Fraction(16) if dimension == "POLYGON" else Fraction(0)
    )


def test_clockwise_orientation_trap_is_rejected_before_kernel_work() -> None:
    clockwise = [PUBLISHED_PENTAGON[0], *reversed(PUBLISHED_PENTAGON[1:])]
    with pytest.raises(ValidationError, match="counterclockwise cyclic order"):
        _request(clockwise)


def test_non_simple_ring_is_rejected() -> None:
    with pytest.raises(ValidationError, match="simple polygon"):
        _request([(0, 0), (2, 2), (0, 2), (2, 0)])


def test_cyclic_rotation_preserves_kernel_and_scalar_measures() -> None:
    first = compute_visibility_kernel(_request(PUBLISHED_PENTAGON))
    rotated_points = PUBLISHED_PENTAGON[2:] + PUBLISHED_PENTAGON[:2]
    rotated = compute_visibility_kernel(_request(rotated_points))
    assert _kernel_points(rotated) == _kernel_points(first)
    assert rotated.convex_hull == first.convex_hull
    assert rotated.polygon_area == first.polygon_area
    assert rotated.kernel_area == first.kernel_area
    assert rotated.kernel_to_polygon_area_ratio == first.kernel_to_polygon_area_ratio


def test_fractional_polygon_round_trips_through_source_bound_validation() -> None:
    result = compute_visibility_kernel(
        _request(
            [
                (Fraction(1, 3), Fraction(1, 5)),
                (Fraction(7, 3), Fraction(1, 5)),
                (Fraction(7, 3), Fraction(11, 5)),
                (Fraction(1, 3), Fraction(11, 5)),
            ]
        )
    )
    replayed = PolygonKernelResult.model_validate_json(result.model_dump_json())
    assert replayed == result
    assert replayed.polygon_area.as_fraction() == 4


@pytest.mark.parametrize(
    "mutation",
    ["source", "half_plane", "kernel", "area", "dimension"],
)
def test_result_validation_rejects_independent_mutations(mutation: str) -> None:
    result = compute_visibility_kernel(_request(PUBLISHED_PENTAGON))
    payload = deepcopy(result.model_dump(mode="json"))
    if mutation == "source":
        payload["polygon"]["points"][3]["x"] = {"num": "22175", "den": "1"}
    elif mutation == "half_plane":
        payload["half_planes"][0]["a"] = {"num": "9239", "den": "1"}
    elif mutation == "kernel":
        payload["kernel_boundary"][1]["point"]["x"] = {
            "num": "19799",
            "den": "1",
        }
    elif mutation == "area":
        payload["kernel_area"] = {"num": "113430241", "den": "1"}
    else:
        payload["kernel_dimension"] = "SEGMENT"
    with pytest.raises(ValidationError, match="does not match the retained source"):
        PolygonKernelResult.model_validate(payload)


def _parabola_polygon(scale: int = 1) -> list[tuple[int, int]]:
    return [
        (x * scale, x * x * scale)
        for x in range(
            -(MAX_KERNEL_SOURCE_VERTICES // 2),
            MAX_KERNEL_SOURCE_VERTICES // 2,
        )
    ]


def test_accepts_vertex_and_coordinate_boundaries() -> None:
    polygon = _request(_parabola_polygon())
    assert len(polygon.polygon.points) == MAX_KERNEL_SOURCE_VERTICES
    result = compute_visibility_kernel(polygon)
    assert result.kernel_dimension == "POLYGON"
    assert len(result.kernel_boundary) == MAX_KERNEL_SOURCE_VERTICES

    magnitude = 10 ** (MAX_KERNEL_COORDINATE_DIGITS - 1)
    coordinate_boundary = _request([(0, 0), (magnitude, 0), (0, 1)])
    assert coordinate_boundary.polygon.points[1].x.num == str(magnitude)


def test_rejects_immediately_above_structural_boundaries() -> None:
    too_many = [*_parabola_polygon(), (MAX_KERNEL_SOURCE_VERTICES, 0)]
    with pytest.raises(ValidationError, match="at most 64 items"):
        _request(too_many)

    magnitude = 10**MAX_KERNEL_COORDINATE_DIGITS
    with pytest.raises(ValidationError, match="64-digit visibility-kernel bound"):
        _request([(0, 0), (magnitude, 0), (0, 1)])


def test_rejects_derived_work_before_pairwise_expansion() -> None:
    with pytest.raises(ValidationError, match="feasibility work"):
        _request(_parabola_polygon(10**30))


def test_rejects_derived_result_size_before_pairwise_expansion() -> None:
    with pytest.raises(ValidationError, match="character bound"):
        _request(_parabola_polygon(10**45))
