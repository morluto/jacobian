from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.finite_metric_spaces._models import (
    MAX_DISTANCE,
    BallRequest,
    FiniteMetricSpace,
    GromovHyperbolicityRequest,
    MetricProfileRequest,
)
from jacobian.math.finite_metric_spaces._operations import (
    compute_ball,
    compute_gromov_hyperbolicity,
    compute_metric_profile,
)


def _ms(distances: list[list[int]]) -> FiniteMetricSpace:
    return FiniteMetricSpace(
        point_count=len(distances), distances=tuple(tuple(r) for r in distances)
    )


def test_profile_path_graph() -> None:
    """Path graph 0-1-2: diameter=2, radius=1, center=1."""
    ms = _ms([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    result = compute_metric_profile(MetricProfileRequest(metric_space=ms))
    assert result.diameter == 2
    assert result.radius == 1
    assert result.centers == (1,)
    assert result.periphery == (0, 2)
    assert result.method == "DIRECT_DISTANCE_MATRIX_SCAN"


def test_profile_complete_graph() -> None:
    """Complete graph K3: all eccentricities = 1, diameter = radius = 1."""
    ms = _ms([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    result = compute_metric_profile(MetricProfileRequest(metric_space=ms))
    assert result.diameter == 1
    assert result.radius == 1
    assert set(result.centers) == {0, 1, 2}


def test_profile_star_graph_center_has_min_eccentricity() -> None:
    ms = _ms([[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]])
    result = compute_metric_profile(MetricProfileRequest(metric_space=ms))
    assert result.centers == (0,)
    assert result.radius == 1
    assert result.diameter == 2


def test_ball_radius_0() -> None:
    """Ball of radius 0 contains only the center."""
    ms = _ms([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    result = compute_ball(BallRequest(metric_space=ms, center=1, radius=0))
    assert result.points == (1,)


def test_ball_radius_1_path() -> None:
    """Ball of radius 1 at point 1 in a path: {0, 1, 2}."""
    ms = _ms([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    result = compute_ball(BallRequest(metric_space=ms, center=1, radius=1))
    assert set(result.points) == {0, 1, 2}


def test_ball_radius_1_at_endpoint() -> None:
    """Ball of radius 1 at point 0 in a path: {0, 1}."""
    ms = _ms([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    result = compute_ball(BallRequest(metric_space=ms, center=0, radius=1))
    assert set(result.points) == {0, 1}


def test_gromov_hyperbolicity_path_graph() -> None:
    """Path graph 0-1-2-3: Gromov hyperbolicity is 0 (tree)."""
    ms = _ms([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    result = compute_gromov_hyperbolicity(GromovHyperbolicityRequest(metric_space=ms))
    assert result.hyperbolicity.as_fraction() == Fraction(0, 1)


def test_gromov_hyperbolicity_cycle_c4() -> None:
    """C4 (cycle on 4 points): Gromov hyperbolicity is 1 (integer)."""
    ms = _ms([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    result = compute_gromov_hyperbolicity(GromovHyperbolicityRequest(metric_space=ms))
    assert result.hyperbolicity.as_fraction() == Fraction(1, 1)


def test_gromov_hyperbolicity_cycle_c5_half_integer() -> None:
    """C5 (cycle on 5 points): Gromov hyperbolicity is 1/2 (half-integer)."""
    ms = _ms(
        [
            [0, 1, 2, 2, 1],
            [1, 0, 1, 2, 2],
            [2, 1, 0, 1, 2],
            [2, 2, 1, 0, 1],
            [1, 2, 2, 1, 0],
        ]
    )
    result = compute_gromov_hyperbolicity(GromovHyperbolicityRequest(metric_space=ms))
    assert result.hyperbolicity.as_fraction() == Fraction(1, 2)


def test_contract_rejects_nonsymmetric() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteMetricSpace(
            point_count=2,
            distances=((0, 1), (2, 0)),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_metric_space.distance_matrix_asymmetric"
    )


def test_contract_rejects_nonzero_diagonal() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteMetricSpace(
            point_count=2,
            distances=((1, 1), (1, 0)),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_metric_space.distance_diagonal_nonzero"
    )


def test_contract_rejects_triangle_inequality() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteMetricSpace(
            point_count=3,
            distances=((0, 1, 3), (1, 0, 1), (3, 1, 0)),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_metric_space.distance_triangle_inequality_violation"
    )


def test_contract_rejects_zero_distance() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteMetricSpace(
            point_count=3,
            distances=((0, 0, 1), (0, 0, 1), (1, 1, 0)),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_metric_space.distance_nonpositive_between_distinct_points"
    )


def test_contract_rejects_oversized_distance() -> None:
    with pytest.raises(ValidationError) as error:
        FiniteMetricSpace(
            point_count=2,
            distances=((0, MAX_DISTANCE + 1), (MAX_DISTANCE + 1, 0)),
        )
    assert error.value.errors()[0]["type"] == "less_than_equal"


def test_ball_rejects_center_out_of_range() -> None:
    ms = _ms([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    with pytest.raises(ValidationError) as error:
        BallRequest(metric_space=ms, center=3, radius=1)
    assert (
        error.value.errors()[0]["type"]
        == "finite_metric_space.ball_center_out_of_range"
    )
