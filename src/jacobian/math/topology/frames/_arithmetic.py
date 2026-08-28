"""Exact arithmetic shared by finite-frame execution and replay."""


def dot(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the exact standard-coordinate inner product."""

    return sum(a * b for a, b in zip(left, right, strict=True))


__all__ = ["dot"]
