"""Tests for the exact initial h-difference (g-) vector operation."""

from jacobian.math.topology._structural import (
    FVectorRequest,
    compute_f_vector,
    compute_g_vector,
)


def _request(facets: list[list[str]], vertices: list[str]) -> FVectorRequest:
    return FVectorRequest.model_validate(
        {"complex": {"vertices": vertices, "facets": facets}}
    )


def test_four_cycle_g_vector_matches_hand_computation() -> None:
    # f=(1,4,4), so h=(1,2,1) and g=(1,1).
    request = _request(
        [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
        ["a", "b", "c", "d"],
    )
    faces = compute_f_vector(request)
    result = compute_g_vector(request)
    assert faces.f_vector == result.f_vector == (1, 4, 4)
    assert faces.h_vector == result.h_vector == (1, 2, 1)
    assert result.g_vector == (1, 1)


def test_g_vector_stops_at_midpoint_for_two_dimensional_simplex() -> None:
    request = _request([["a", "b", "c"]], ["a", "b", "c"])
    result = compute_g_vector(request)
    assert result.f_vector == (1, 3, 3, 1)
    assert result.h_vector == (1, 0, 0, 0)
    assert result.g_vector == (1, -1)


def test_g_vector_does_not_imply_sphere_or_nonnegative_entries() -> None:
    request = _request([["a", "b", "c"]], ["a", "b", "c"])
    assert compute_g_vector(request).g_vector == (1, -1)
