"""Exact public API contract for ``jacobian.math.geometry.euclidean``."""

from jacobian.math.geometry import euclidean


def test_exact_public_api_symbols() -> None:
    expected = (
        "Triangle",
        "angles_equal",
        "squared_segment_ratio",
        "triangles_similar",
    )

    assert tuple(euclidean.__all__) == expected
    assert len(euclidean.__all__) == len(set(euclidean.__all__))
    assert all(hasattr(euclidean, name) for name in euclidean.__all__)
