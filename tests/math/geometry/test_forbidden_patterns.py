"""Tests for the forbidden-pattern screening operation."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _point(label: str, x: int, y: int) -> ForbiddenLabelledPoint:
    return ForbiddenLabelledPoint(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational(num=str(x), den="1"),
            y=CanonicalRational(num=str(y), den="1"),
        ),
    )


def _configuration(*coordinates: tuple[int, int]) -> ForbiddenConfiguration:
    return ForbiddenConfiguration(
        points=tuple(
            _point(f"p{index}", x, y) for index, (x, y) in enumerate(coordinates)
        )
    )


class TestCheckedCountBinding:
    """Reported exhaustive-work counts replay against the configuration."""

    def test_pattern_free_counts_match_the_lexicographic_scan(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (0, 1), (2, 2))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1
        ForbiddenPatternsResult.model_validate(result.model_dump())

    def test_forged_zero_counts_rejected(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (0, 1), (2, 2))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        forged = result.model_copy(
            update={"checked_triples": 0, "checked_quadruples": 0}
        )
        with pytest.raises(ValidationError, match="checked_triples"):
            ForbiddenPatternsResult.model_validate(forged.model_dump())

    def test_inflated_counts_rejected(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (0, 1), (2, 2))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        forged = result.model_copy(update={"checked_quadruples": 7})
        with pytest.raises(ValidationError, match="checked_quadruples"):
            ForbiddenPatternsResult.model_validate(forged.model_dump())

    def test_witness_stops_the_triple_prefix_at_one(self) -> None:
        # The lexicographically first triple is already collinear.
        configuration = _configuration((0, 0), (1, 0), (2, 0), (5, 5))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        assert result.has_collinear_triple
        assert result.checked_triples == 1

    def test_larger_configuration_counts_every_prefix(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (0, 2), (3, 1), (5, 7), (4, 11))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        assert result.checked_triples == 20  # C(6, 3): no collinear triple
        assert result.checked_quadruples == 15  # C(6, 4): no concyclic quad
        ForbiddenPatternsResult.model_validate(result.model_dump())
