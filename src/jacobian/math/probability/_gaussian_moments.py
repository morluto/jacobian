"""Exact moments of standard independent Gaussian coordinates."""


def gaussian_univariate_moment(exponent: int) -> int:
    """Return ``E[X**exponent]`` for a standard Gaussian ``X``."""

    if exponent % 2:
        return 0
    result = 1
    for factor in range(1, exponent, 2):
        result *= factor
    return result


__all__ = ["gaussian_univariate_moment"]
