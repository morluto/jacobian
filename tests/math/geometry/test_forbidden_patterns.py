"""Tests for forbidden-pattern screening and witness binding."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    CollinearTriple,
    ConcyclicQuadruple,
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _configuration(*points: tuple[int, int]) -> ForbiddenConfiguration:
    return ForbiddenConfiguration(
        points=tuple(
            ForbiddenLabelledPoint(
                label=str(index),
                point={
                    "x": {"num": str(x), "den": "1"},
                    "y": {"num": str(y), "den": "1"},
                },
            )
            for index, (x, y) in enumerate(points)
        )
    )


def _request(*points: tuple[int, int]) -> ForbiddenPatternsRequest:
    return ForbiddenPatternsRequest(configuration=_configuration(*points))


_COLLINEAR_CONFIG = ((0, 0), (1, 0), (2, 0), (0, 1))
_RECTANGLE_CONFIG = ((0, 0), (2, 0), (0, 1), (2, 1))
_GENERIC_CONFIG = ((0, 0), (1, 0), (0, 1), (2, 3))


class TestForbiddenPatternsKnownAnswer:
    def test_collinear_triple_is_the_canonical_first_witness(self):
        result = forbidden_patterns(_request(*_COLLINEAR_CONFIG))
        assert result.has_collinear_triple is True
        assert result.collinear_triple == CollinearTriple(first=0, second=1, third=2)
        assert result.checked_triples == 1
        assert result.has_concyclic_quadruple is False
        assert result.checked_quadruples == 1

    def test_rectangle_is_concyclic_with_clean_collinear_sweep(self):
        result = forbidden_patterns(_request(*_RECTANGLE_CONFIG))
        assert result.has_collinear_triple is False
        assert result.collinear_triple is None
        assert result.checked_triples == 4
        assert result.has_concyclic_quadruple is True
        assert result.concyclic_quadruple == ConcyclicQuadruple(
            first=0, second=1, third=2, fourth=3
        )
        assert result.checked_quadruples == 1

    def test_generic_configuration_has_neither_pattern(self):
        result = forbidden_patterns(_request(*_GENERIC_CONFIG))
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1


class TestWitnessBinding:
    def test_result_replays_from_retained_configuration(self):
        for points in (_COLLINEAR_CONFIG, _RECTANGLE_CONFIG, _GENERIC_CONFIG):
            result = forbidden_patterns(_request(*points))
            replayed = ForbiddenPatternsResult.model_validate(result.model_dump())
            assert replayed == result

    def test_forged_collinear_witness_rejected(self):
        payload = forbidden_patterns(_request(*_COLLINEAR_CONFIG)).model_dump()
        payload["collinear_triple"] = {"first": 0, "second": 1, "third": 3}
        with pytest.raises(ValidationError, match="canonical first"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_wrong_checked_prefix_rejected(self):
        payload = forbidden_patterns(_request(*_COLLINEAR_CONFIG)).model_dump()
        payload["checked_triples"] = 2
        with pytest.raises(ValidationError, match="witness position"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_stray_witness_on_clean_sweep_rejected(self):
        payload = forbidden_patterns(_request(*_GENERIC_CONFIG)).model_dump()
        payload["collinear_triple"] = {"first": 0, "second": 1, "third": 3}
        with pytest.raises(ValidationError, match="no collinear witness"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_incomplete_clean_sweep_count_rejected(self):
        payload = forbidden_patterns(_request(*_GENERIC_CONFIG)).model_dump()
        payload["checked_quadruples"] = 0
        with pytest.raises(ValidationError, match="C\\(point_count, 4\\)"):
            ForbiddenPatternsResult.model_validate(payload)
