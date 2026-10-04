"""Exact public API contract for ``jacobian.math.geometry.euclidean``."""

from jacobian.math.geometry import euclidean


def test_exact_public_api_symbols() -> None:
    expected = (
        "Triangle",
        "angles_equal",
        "squared_segment_ratio",
        "triangles_similar",
        "verify_angle_equality",
        "verify_triangle_similarity",
    )

    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(euclidean.__all__)
    # The subset assertion above cannot see a newly added name, so the
    # private-name rule is checked over the complete __all__.
    assert all(not name.startswith("_") for name in euclidean.__all__)
    assert len(euclidean.__all__) == len(set(euclidean.__all__))
    assert all(hasattr(euclidean, name) for name in euclidean.__all__)
