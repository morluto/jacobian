"""Exact arithmetic shared by rational linear-program operations and replay."""

from fractions import Fraction


def rational_dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    """Return the exact inner product of equally sized rational vectors."""

    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


__all__ = ["rational_dot"]
