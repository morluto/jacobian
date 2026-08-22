"""Tests for forbidden-pattern screening witness validation."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    CollinearTriple,
    ConcyclicQuadruple,
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _labelled(index: int, coord: tuple[int, int]) -> ForbiddenLabelledPoint:
    x, y = coord
    return ForbiddenLabelledPoint(
        label=f"p{index}",
        point=RationalPoint2D.model_validate(
            {
                "x": {"num": str(x), "den": "1"},
                "y": {"num": str(y), "den": "1"},
            }
        ),
    )


def _configuration(*coords: tuple[int, int]) -> ForbiddenConfiguration:
    return ForbiddenConfiguration(
        points=tuple(_labelled(i, c) for i, c in enumerate(coords))
    )


class TestForbiddenPatternWitnesses:
    def test_collinear_witness_replays(self):
        config = _configuration((0, 0), (1, 0), (2, 0), (0, 1))
        result = forbidden_patterns(ForbiddenPatternsRequest(configuration=config))
        assert result.has_collinear_triple
        assert result.collinear_triple is not None
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)

    def test_noncollinear_triple_witness_rejected(self):
        # The reported triple (0,1,3) is not collinear even though some
        # collinear triple exists in the configuration.
        config = _configuration((0, 0), (1, 0), (2, 0), (0, 1))
        with pytest.raises(ValidationError, match="replay as collinear"):
            ForbiddenPatternsResult(
                configuration=config,
                point_count=4,
                has_collinear_triple=True,
                has_concyclic_quadruple=False,
                collinear_triple=CollinearTriple(first=0, second=1, third=3),
                checked_triples=4,
                checked_quadruples=1,
            )

    def test_concyclic_witness_replays(self):
        # A non-square rectangle's four vertices are concyclic.
        config = _configuration((0, 0), (2, 0), (2, 1), (0, 1))
        result = forbidden_patterns(ForbiddenPatternsRequest(configuration=config))
        assert result.has_concyclic_quadruple
        quadruple = result.concyclic_quadruple
        assert quadruple is not None
        ForbiddenPatternsResult.model_validate(result.model_dump())

    def test_nonconcyclic_quadruple_witness_rejected(self):
        # Four points on the unit circle make the predicate true, but the
        # reported witness swaps in an off-circle fifth point.
        config = _configuration((1, 0), (-1, 0), (0, 1), (0, -1), (5, 5))
        with pytest.raises(ValidationError, match="replay as concyclic"):
            ForbiddenPatternsResult(
                configuration=config,
                point_count=5,
                has_collinear_triple=False,
                has_concyclic_quadruple=True,
                concyclic_quadruple=ConcyclicQuadruple(
                    first=0, second=1, third=2, fourth=4
                ),
                checked_triples=10,
                checked_quadruples=5,
            )
