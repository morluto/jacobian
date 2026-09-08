"""Area profiles preserve translation invariance across coordinate-height limits."""

import json
from fractions import Fraction
from itertools import permutations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.triangle_area_profile._models import (
    TriangleAreaProfileResult,
)
from jacobian.math.geometry.exact.triangle_area_profile._tools import TOOLS


def _run(points: tuple[tuple[Fraction, Fraction], ...]) -> TriangleAreaProfileResult:
    source = PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=label,
                coordinates=tuple(CanonicalRational.from_fraction(x) for x in row),
            )
            for label, row in zip(("z", "a", "q", "b"), points, strict=False)
        )
    )
    tool = TOOLS[0]
    result = tool.run(
        tool.request_type.model_validate_json(
            json.dumps({"configuration": source.model_dump(mode="json")})
        )
    )
    restored = TriangleAreaProfileResult.model_validate_json(result.model_dump_json())
    assert restored.configuration == source
    return restored


def test_rational_square_keeps_all_areas_after_large_translation() -> None:
    offset = 10**10000
    points = (
        (Fraction(offset), Fraction(offset)),
        (Fraction(offset) + Fraction(1, 2), Fraction(offset)),
        (Fraction(offset), Fraction(offset) + Fraction(1, 3)),
        (Fraction(offset) + Fraction(1, 2), Fraction(offset) + Fraction(1, 3)),
    )
    for rows in permutations(points):
        result = _run(rows)
        assert len(result.entries) == 4
        assert all(
            entry.area.as_fraction() == Fraction(1, 12) for entry in result.entries
        )
        assert len(result.area_classes) == 1
        assert len(result.area_classes[0][1]) == 4


def test_truly_oversized_area_remains_refused() -> None:
    wide = Fraction(10**20000)
    with pytest.raises(OperationResourceAdmissionError, match="area"):
        _run(((Fraction(0), Fraction(0)), (wide, Fraction(0)), (Fraction(0), wide)))
