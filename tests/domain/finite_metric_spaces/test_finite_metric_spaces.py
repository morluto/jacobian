from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.finite_metric_spaces import (
    BallRequest,
    FiniteMetricSpace,
    GromovHyperbolicityRequest,
    MetricProfileRequest,
)
from jacobian.domains.finite_metric_spaces.operations import (
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


def test_profile_complete_graph() -> None:
    """Complete graph K3: all eccentricities = 1, diameter = radius = 1."""
    ms = _ms([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    result = compute_metric_profile(MetricProfileRequest(metric_space=ms))
    assert result.diameter == 1
    assert result.radius == 1
    assert set(result.centers) == {0, 1, 2}


def test_profile_single_point_is_center() -> None:
    """Star graph: center has min eccentricity."""
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
    """Path graph 0-1-2-3: computed Gromov hyperbolicity is positive."""
    ms = _ms([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    result = compute_gromov_hyperbolicity(GromovHyperbolicityRequest(metric_space=ms))
    assert result.hyperbolicity >= 0


def test_gromov_hyperbolicity_cycle_c4() -> None:
    """C4 (cycle on 4 points): Gromov hyperbolicity is computed exactly."""
    ms = _ms([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    result = compute_gromov_hyperbolicity(GromovHyperbolicityRequest(metric_space=ms))
    assert result.hyperbolicity >= 0


def test_contract_rejects_nonsymmetric() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        FiniteMetricSpace(
            point_count=2,
            distances=((0, 1), (2, 0)),
        )


def test_contract_rejects_nonzero_diagonal() -> None:
    with pytest.raises(ValidationError, match="zero"):
        FiniteMetricSpace(
            point_count=2,
            distances=((1, 1), (1, 0)),
        )
