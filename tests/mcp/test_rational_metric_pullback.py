"""MCP declaration smoke test for rational metric pullback."""

from jacobian.math.geometry.differential.pullback._tools import TOOLS


def test_pullback_operation_is_published() -> None:
    assert (
        TOOLS[0].operation_id
        == "differential_geometry.rational_metric.pullback.compute"
    )
