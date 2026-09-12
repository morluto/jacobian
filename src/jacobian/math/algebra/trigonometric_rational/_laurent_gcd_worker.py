"""Killable worker for Gaussian Laurent polynomial GCD cancellation."""

from __future__ import annotations

import json
import sys
from typing import Any


def _load_polynomial(ring: Any, payload: dict[str, list[Any]]) -> Any:
    from sympy import I, Rational

    mapping = {}
    for support, real_num, real_den, imag_num, imag_den in zip(
        payload["supports"],
        payload["real_numerators"],
        payload["real_denominators"],
        payload["imag_numerators"],
        payload["imag_denominators"],
        strict=True,
    ):
        mapping[tuple(int(value) for value in support)] = Rational(
            int(real_num), int(real_den)
        ) + I * Rational(int(imag_num), int(imag_den))
    return ring.from_dict(mapping)


def _dump_polynomial(polynomial: Any) -> dict[str, list[Any]]:
    supports: list[list[int]] = []
    real_numerators: list[str] = []
    real_denominators: list[str] = []
    imag_numerators: list[str] = []
    imag_denominators: list[str] = []
    for support, coefficient in polynomial.terms():
        real = getattr(coefficient, "x", None)
        imaginary = getattr(coefficient, "y", None)
        if real is None or imaginary is None:
            real, imaginary = coefficient.as_real_imag()
        supports.append([int(value) for value in support])
        real_numerators.append(str(int(real.p)))
        real_denominators.append(str(int(real.q)))
        imag_numerators.append(str(int(imaginary.p)))
        imag_denominators.append(str(int(imaginary.q)))
    return {
        "supports": supports,
        "real_numerators": real_numerators,
        "real_denominators": real_denominators,
        "imag_numerators": imag_numerators,
        "imag_denominators": imag_denominators,
    }


def main() -> int:
    from sympy import Symbol
    from sympy.polys.domains import QQ_I
    from sympy.polys.rings import ring

    payload = json.load(sys.stdin)
    axis = int(payload["axis"])
    symbols = tuple(Symbol(f"x{index}") for index in range(axis))
    polynomial_ring, *_ = ring(symbols, QQ_I)
    left = _load_polynomial(polynomial_ring, payload["left"])
    right = _load_polynomial(polynomial_ring, payload["right"])
    common = left.gcd(right)
    json.dump(
        {
            "left": _dump_polynomial(left.exquo(common)),
            "right": _dump_polynomial(right.exquo(common)),
        },
        sys.stdout,
        separators=(",", ":"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
