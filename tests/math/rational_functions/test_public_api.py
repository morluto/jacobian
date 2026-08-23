"""Public native API contract for rational functions."""


def test_public_api_is_exact() -> None:
    from jacobian.math import rational_functions

    assert rational_functions.__all__ == ["hermite_reduction"]
