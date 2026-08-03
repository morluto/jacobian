"""Small polynomial payload builders shared across semantic test lanes."""

from __future__ import annotations


def univariate_term(coefficient: int, exponent: int) -> dict[str, object]:
    return {
        "coefficient": {"num": str(coefficient), "den": "1"},
        "exponents": [exponent],
    }
