"""Tests for incidence structure operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.incidence_structures._models import IncidenceMatrixRequest
from jacobian.math.incidence_structures._operations import (
    compute_degree_profile,
    compute_incidence_matrix,
)

STRUCTURE = {
    "points": ["p1", "p2", "p3"],
    "block_ids": ["b1", "b2"],
    "blocks": [["p1", "p2"], ["p2", "p3"]],
}


class TestIncidenceMatrix:
    def test_matrix(self) -> None:
        result = compute_incidence_matrix(IncidenceMatrixRequest(incidence=STRUCTURE))
        assert result.matrix == ((1, 0), (1, 1), (0, 1))
        assert result.points == ("p1", "p2", "p3")
        assert result.block_ids == ("b1", "b2")

    def test_duplicate_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            IncidenceMatrixRequest(
                incidence={
                    "points": ["p1", "p1"],
                    "block_ids": ["b1"],
                    "blocks": [["p1"]],
                }
            )

    def test_invalid_block_member(self) -> None:
        with pytest.raises(ValidationError, match="declared point"):
            IncidenceMatrixRequest(
                incidence={
                    "points": ["p1"],
                    "block_ids": ["b1"],
                    "blocks": [["p2"]],
                }
            )


class TestDegreeProfile:
    def test_degrees(self) -> None:
        result = compute_degree_profile(IncidenceMatrixRequest(incidence=STRUCTURE))
        assert result.point_degrees == (("p1", 1), ("p2", 2), ("p3", 1))
        assert result.block_degrees == (("b1", 2), ("b2", 2))
        assert result.total_incidences == 4

    def test_single_point(self) -> None:
        result = compute_degree_profile(
            IncidenceMatrixRequest(
                incidence={
                    "points": ["p1"],
                    "block_ids": ["b1"],
                    "blocks": [["p1"]],
                }
            )
        )
        assert result.total_incidences == 1
