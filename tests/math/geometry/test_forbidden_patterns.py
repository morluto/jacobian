"""Contract tests for bounded forbidden-pattern configuration screening."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import (
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import forbidden_patterns


def _labelled(index: int, x: int, y: int) -> ForbiddenLabelledPoint:
    return ForbiddenLabelledPoint(
        label=f"P{index}",
        point=RationalPoint2D(
            x={"num": str(x), "den": "1"},
            y={"num": str(y), "den": "1"},
        ),
    )


def _configuration(*points: tuple[int, int]) -> ForbiddenConfiguration:
    return ForbiddenConfiguration(
        points=tuple(_labelled(i, x, y) for i, (x, y) in enumerate(points))
    )


class TestForbiddenConfigurationBounds:
    def test_rejects_more_than_32_points(self) -> None:
        points = [(i, i * i % 13 + 17 * (i % 5)) for i in range(33)]
        with pytest.raises(ValidationError):
            _configuration(*points)

    def test_admits_32_points(self) -> None:
        points = [(i, i * i % 13 + 17 * (i % 5)) for i in range(32)]
        assert len(_configuration(*points).points) == 32

    def test_rejects_coordinates_beyond_64_digits(self) -> None:
        huge = {"num": "1" + "0" * 64, "den": "1"}
        with pytest.raises(ValidationError, match="64-digit"):
            ForbiddenConfiguration(
                points=(
                    ForbiddenLabelledPoint(
                        label="A",
                        point=RationalPoint2D(x=huge, y={"num": "0", "den": "1"}),
                    ),
                    ForbiddenLabelledPoint(
                        label="B",
                        point=RationalPoint2D(
                            x={"num": "1", "den": "1"}, y={"num": "0", "den": "1"}
                        ),
                    ),
                )
            )


class TestForbiddenPatternsScreening:
    def test_general_position_has_no_forbidden_pattern(self) -> None:
        request = ForbiddenPatternsRequest(
            configuration=_configuration((0, 0), (1, 0), (0, 1), (2, 3))
        )
        result = forbidden_patterns(request)
        assert result.point_count == 4
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        assert result.collinear_triple is None
        assert result.concyclic_quadruple is None
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1

    def test_square_reports_concyclic_quadruple(self) -> None:
        request = ForbiddenPatternsRequest(
            configuration=_configuration((0, 0), (1, 0), (1, 1), (0, 1))
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is True
        assert (
            result.concyclic_quadruple.first,
            result.concyclic_quadruple.second,
            result.concyclic_quadruple.third,
            result.concyclic_quadruple.fourth,
        ) == (0, 1, 2, 3)

    def test_quadruple_with_collinear_triple_is_not_concyclic(self) -> None:
        """No finite circle contains three collinear points.

        (0,0), (1,0), (2,0), (0,1): the 4x4 concyclicity determinant vanishes
        because the first three points are collinear, but the quadruple is
        degenerate and must not carry a concyclic claim.
        """
        request = ForbiddenPatternsRequest(
            configuration=_configuration((0, 0), (1, 0), (2, 0), (0, 1))
        )
        result = forbidden_patterns(request)
        assert result.has_concyclic_quadruple is False
        assert result.concyclic_quadruple is None

    def test_fully_collinear_quadruple_is_not_concyclic(self) -> None:
        request = ForbiddenPatternsRequest(
            configuration=_configuration((0, 0), (1, 0), (2, 0), (3, 0))
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        assert result.has_concyclic_quadruple is False

    def test_collinear_triple_witness(self) -> None:
        request = ForbiddenPatternsRequest(
            configuration=_configuration((0, 0), (1, 1), (2, 2), (0, 1), (1, 0))
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        witness = result.collinear_triple
        assert (witness.first, witness.second, witness.third) == (0, 1, 2)

    def test_retains_configuration_in_result(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (0, 1), (2, 3))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        assert result.configuration == configuration


class TestSourceBoundResultValidation:
    def test_result_replays_its_own_configuration(self) -> None:
        configuration = _configuration((0, 0), (1, 0), (1, 1), (0, 1))
        result = forbidden_patterns(
            ForbiddenPatternsRequest(configuration=configuration)
        )
        revalidated = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert revalidated.has_concyclic_quadruple is True
        assert revalidated.concyclic_quadruple == result.concyclic_quadruple

    def test_forged_negative_conclusion_is_rejected(self) -> None:
        """A concyclic square cannot validate as a generic configuration."""
        payload = forbidden_patterns(
            ForbiddenPatternsRequest(
                configuration=_configuration((0, 0), (1, 0), (0, 1), (2, 3))
            )
        ).model_dump()
        payload["configuration"] = _configuration(
            (0, 0), (1, 0), (1, 1), (0, 1)
        ).model_dump()
        with pytest.raises(ValidationError, match="exact screening"):
            ForbiddenPatternsResult.model_validate(payload)

    def test_mismatched_point_count_is_rejected(self) -> None:
        payload = forbidden_patterns(
            ForbiddenPatternsRequest(
                configuration=_configuration((0, 0), (1, 0), (0, 1), (2, 3))
            )
        ).model_dump()
        payload["point_count"] = 3
        with pytest.raises(ValidationError, match="point count"):
            ForbiddenPatternsResult.model_validate(payload)
