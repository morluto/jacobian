"""Independent finite-assignment checks for sparse character pullback."""

from fractions import Fraction

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def _q(n: int, d: int = 1) -> dict[str, str]:
    return {"num": str(n), "den": str(d)}


@pytest.mark.parametrize("target_dimension", [2, 512])
def test_pullback_combines_phases_and_colliding_characters(
    target_dimension: int,
) -> None:
    terms = [((), 3, 1), ((0,), 1, 2), ((0, 1), 1, 1), ((0, 2), 2, 1), ((2, 3), 1, 1)]
    polynomial = {
        "variable_count": 4,
        "terms": [{"character": list(c), "coefficient": _q(n, d)} for c, n, d in terms],
    }
    result = invoke_operation(
        "boolean.walsh_polynomial.affine_pullback.compute",
        {
            "polynomial": polynomial,
            "affine_map": {
                "target_dimension": target_dimension,
                "rows": [[0], [1], [0], [1]],
                "offset": [0, 0, 1, 0],
            },
        },
        Catalog.open(),
    ).output
    assert result["variable_count"] == target_dimension
    assert result["terms"] == [
        {"character": [], "coefficient": _q(1)},
        {"character": [0], "coefficient": _q(1, 2)},
    ]
    for y in range(4):
        x = [y & 1, y >> 1, (y & 1) ^ 1, y >> 1]
        source = sum(Fraction(n, d) * (-1) ** sum(x[i] for i in c) for c, n, d in terms)
        assert source == 1 + Fraction(1, 2) * (-1) ** (y & 1)


def test_pullback_to_point_and_zero_polynomial_retain_zero_axes() -> None:
    result = invoke_operation(
        "boolean.walsh_polynomial.affine_pullback.compute",
        {
            "polynomial": {
                "variable_count": 1,
                "terms": [
                    {"character": [], "coefficient": _q(1)},
                    {"character": [0], "coefficient": _q(1)},
                ],
            },
            "affine_map": {"target_dimension": 0, "rows": [[]], "offset": [1]},
        },
        Catalog.open(),
    ).output
    assert result == {
        "variable_count": 0,
        "terms": [],
        "convention": "BOOLEAN_CHARACTERS",
    }
