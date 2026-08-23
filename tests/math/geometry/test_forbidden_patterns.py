"""Tests for forbidden-pattern screening over labelled rational points."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _point(
    label: str, x: tuple[str, str], y: tuple[str, str]
) -> ForbiddenLabelledPoint:
    return ForbiddenLabelledPoint(
        label=label,
        point={
            "x": {"num": x[0], "den": x[1]},
            "y": {"num": y[0], "den": y[1]},
        },
    )


class TestForbiddenPatternsReplay:
    def test_mixed_denominator_collinear_triple_detected(self):
        """Collinearity of (0,1), (1/2,1), (1,1) must survive the replay's
        integer clearing, which scales each point by its own denominator."""
        pts = (
            _point("a", ("0", "1"), ("1", "1")),
            _point("b", ("1", "2"), ("1", "1")),
            _point("c", ("1", "1"), ("1", "1")),
        )
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(points=pts)
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple
        assert result.collinear_triple == type(result.collinear_triple)(
            first=0, second=1, third=2
        )
        assert result.checked_triples == 1

    def test_mixed_denominator_negative_configuration_roundtrips(self):
        pts = (
            _point("a", ("0", "1"), ("0", "1")),
            _point("b", ("1", "3"), ("1", "1")),
            _point("c", ("1", "1"), ("1", "5")),
            _point("d", ("2", "7"), ("7", "2")),
        )
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(points=pts)
        )
        result = forbidden_patterns(request)
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1
        revalidated = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert revalidated == result

    def test_mixed_denominator_concyclic_quadruple_detected(self):
        """(1,0), (0,1), (-3/5,4/5), (3/5,4/5) lie on the unit circle and
        clear to differing per-point homogeneous scales."""
        pts = (
            _point("a", ("1", "1"), ("0", "1")),
            _point("b", ("0", "1"), ("1", "1")),
            _point("c", ("-3", "5"), ("4", "5")),
            _point("d", ("3", "5"), ("4", "5")),
        )
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(points=pts)
        )
        result = forbidden_patterns(request)
        assert not result.has_collinear_triple
        assert result.has_concyclic_quadruple
        assert result.concyclic_quadruple.first == 0
        assert result.concyclic_quadruple.fourth == 3

    def test_collinear_degeneracy_excluded_from_concyclicity(self):
        """A vanishing determinant from a collinear triple is not a circle."""
        pts = (
            _point("a", ("0", "1"), ("0", "1")),
            _point("b", ("1", "2"), ("0", "1")),
            _point("c", ("1", "1"), ("0", "1")),
            _point("d", ("0", "1"), ("1", "1")),
        )
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(points=pts)
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple
        assert result.collinear_triple.first == 0
        assert not result.has_concyclic_quadruple
        assert result.concyclic_quadruple is None

    def test_forged_witness_rejected(self):
        pts = (
            _point("a", ("0", "1"), ("0", "1")),
            _point("b", ("1", "3"), ("1", "1")),
            _point("c", ("1", "1"), ("1", "5")),
        )
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(points=pts)
        )
        genuine = forbidden_patterns(request)
        forged = genuine.model_dump()
        forged["has_collinear_triple"] = True
        with pytest.raises(ValidationError):
            ForbiddenPatternsResult.model_validate(forged)
