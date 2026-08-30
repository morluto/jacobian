from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.spanned_line_profile.operations import (
    compute_spanned_line_profile,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _config(points: Sequence[tuple[str, Sequence[int]]]) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=_label,
                coordinates=tuple(_cr(x) for x in coords),
            )
            for _label, coords in points
        )
    )


def test_three_collinear() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0]), ("c", [2, 0])])
    result = compute_spanned_line_profile(config)
    assert result.line_count == 1  # All on one line


def test_collinear_pairs_with_opposite_directions_share_one_line() -> None:
    config = _config([("a", [1, 0]), ("b", [0, 0]), ("c", [2, 0])])

    result = compute_spanned_line_profile(config)

    assert result.line_count == 1
    assert result.lines[0].source_pairs == ((0, 1), (0, 2), (1, 2))


def test_triangle() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0]), ("c", [0, 1])])
    result = compute_spanned_line_profile(config)
    assert result.line_count == 3  # Three pairs, three lines


def test_square() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0]), ("c", [1, 1]), ("d", [0, 1])])
    result = compute_spanned_line_profile(config)
    # 4 sides + 2 diagonals = 6 lines
    assert result.line_count == 6


def test_result_preserves_source() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0])])
    result = compute_spanned_line_profile(config)
    assert result.configuration == config


def test_coincident_points_are_rejected() -> None:
    config = _config([("a", [0, 0]), ("b", [0, 0])])

    with pytest.raises(PydanticCustomError):
        compute_spanned_line_profile(config)


def test_derived_line_key_growth_is_rejected_before_pair_enumeration() -> None:
    denominator = "9" * 3_000
    wide = CanonicalRational(num="1", den=denominator)
    points = (
        LabelledRationalPoint(label="a", coordinates=(wide,) * 20),
        LabelledRationalPoint(
            label="b", coordinates=(CanonicalRational(num="0", den="1"),) * 20
        ),
    )
    with pytest.raises(ValueError, match="line keys exceed"):
        compute_spanned_line_profile(PointConfiguration(points=points))
