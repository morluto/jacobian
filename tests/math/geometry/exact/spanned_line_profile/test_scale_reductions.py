"""Tractable pair and axis-line profiles precede private line-key growth."""

from itertools import combinations, permutations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.spanned_line_profile.operations import (
    compute_spanned_line_profile,
    verify_spanned_line_profile,
)


def _configuration(rows: tuple[tuple[str, tuple[int, ...]], ...]) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=label,
                coordinates=tuple(CanonicalRational(num=value, den=1) for value in row),
            )
            for label, row in rows
        )
    )


@pytest.mark.parametrize("axis_parallel", [False, True])
def test_large_two_point_profile_preserves_source_order_and_labels(
    axis_parallel: bool,
) -> None:
    translation = 10**10000
    points = (
        ("zeta", (translation, 0)),
        ("alpha", (translation if axis_parallel else -translation, 1)),
    )
    for rows in permutations(points):
        source = _configuration(rows)
        result = compute_spanned_line_profile(source)
        assert result.line_count == 1
        assert result.lines[0].source_pairs == ((0, 1),)
        assert result.lines[0].point_count == 2
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored.configuration == source
        assert verify_spanned_line_profile(restored)


@pytest.mark.parametrize("variable_axis", [0, 1, 2])
def test_translated_axis_line_profile_retains_all_pairs(variable_axis: int) -> None:
    translation = 10**10000
    points = tuple(
        (
            label,
            tuple(value if axis == variable_axis else translation for axis in range(3)),
        )
        for label, value in (("q", 3), ("a", -2), ("z", 0))
    )
    for rows in permutations(points):
        source = _configuration(rows)
        result = compute_spanned_line_profile(source)
        assert result.line_count == 1
        assert result.lines[0].source_pairs == tuple(combinations(range(3), 2))
        assert result.lines[0].point_count == 3
        assert verify_spanned_line_profile(
            type(result).model_validate_json(result.model_dump_json())
        )


@pytest.mark.parametrize("constant_axis", [0, 1, 2])
@pytest.mark.parametrize("collinear", [False, True])
def test_common_translated_axes_do_not_charge_line_key_growth(
    constant_axis: int, collinear: bool
) -> None:
    translation = 10**10000
    plane_points = ((0, 0), (1, 1), (2, 2) if collinear else (0, 2))
    points = []
    for label, coordinates in zip(("z", "q", "a"), plane_points, strict=True):
        row = list(coordinates)
        row.insert(constant_axis, translation)
        points.append((label, tuple(row)))
    for rows in permutations(points):
        source = _configuration(rows)
        result = compute_spanned_line_profile(source)
        assert result.configuration == source
        assert result.line_count == (1 if collinear else 3)
        assert sorted(
            pair for line in result.lines for pair in line.source_pairs
        ) == list(combinations(range(3), 2))
        assert all(line.point_count == (3 if collinear else 2) for line in result.lines)
