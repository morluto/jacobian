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


def _flint_ring_inputs(payload: dict[str, list[Any]], minimum_exponent: int) -> Any:
    """Represent a Laurent polynomial as ``QQ[z, I]`` with a shared shift.

    A Gaussian univariate polynomial is embedded in ``QQ[z, I]`` by treating
    ``i`` as a second indeterminate. Over this transcendental setting,
    multivariate GCD agrees with the GCD over ``QQ(i)[z]`` up to a unit, which
    the caller normalizes. ``minimum_exponent`` is the shift applied by the
    caller so both operands share one coordinate system.
    """

    from flint import fmpq, fmpq_mpoly_ctx

    ctx = fmpq_mpoly_ctx.get(["z", "I"])
    z, imaginary = ctx.gens()
    polynomial = 0 * z
    for index, support in enumerate(payload["supports"]):
        exponent = int(support[0]) - minimum_exponent
        real_num, real_den, imag_num, imag_den = (
            int(payload["real_numerators"][index]),
            int(payload["real_denominators"][index]),
            int(payload["imag_numerators"][index]),
            int(payload["imag_denominators"][index]),
        )
        polynomial += fmpq(real_num, real_den) * z**exponent
        if imag_num:
            polynomial += fmpq(imag_num, imag_den) * imaginary * z**exponent
    return polynomial


def _flint_cancel(
    left_payload: dict[str, list[Any]], right_payload: dict[str, list[Any]]
) -> dict[str, Any] | None:
    """Cancel the common factor through the maintained flint backend.

    The flint result is validated with exact division before it is trusted; a
    mismatch falls back to the sympy ring path.
    """

    supports = [
        int(support[0])
        for payload in (left_payload, right_payload)
        for support in payload["supports"]
    ]
    if not supports:
        return None
    minimum_exponent = min(supports)
    try:
        left = _flint_ring_inputs(left_payload, minimum_exponent)
        right = _flint_ring_inputs(right_payload, minimum_exponent)
        common = left.gcd(right)
        if common.is_zero():
            return None
        left_quotient, left_rem = divmod(left, common)
        right_quotient, right_rem = divmod(right, common)
        if not left_rem.is_zero() or not right_rem.is_zero():
            return None
        # Exact division alone admits a proper common factor. Confirm the
        # cofactors are coprime so the cancellation is the greatest one; flint's
        # multivariate GCD is not guaranteed to be greatest in every case.
        if not left_quotient.gcd(right_quotient).is_constant():
            return None
    except Exception:
        return None
    return {
        "left": _flint_to_payload(left_quotient, minimum_exponent),
        "right": _flint_to_payload(right_quotient, minimum_exponent),
    }


def _flint_to_payload(polynomial: Any, minimum_exponent: int) -> dict[str, list[Any]]:
    """Convert a bivariate ``QQ[z, I]`` polynomial back to the wire payload."""

    supports: list[list[int]] = []
    real_numerators: list[str] = []
    real_denominators: list[str] = []
    imag_numerators: list[str] = []
    imag_denominators: list[str] = []
    for exponents, coefficient in polynomial.terms():
        exponent = exponents[0] + minimum_exponent
        imaginary_degree = exponents[1]
        if imaginary_degree not in (0, 1):
            raise ValueError("unexpected imaginary monomial degree")
        real_numerators.append(str(coefficient.p))
        real_denominators.append(str(coefficient.q))
        imag_numerators.append(str(coefficient.p) if imaginary_degree else "0")
        imag_denominators.append(str(coefficient.q))
        supports.append([int(exponent)])
    return {
        "supports": supports,
        "real_numerators": real_numerators,
        "real_denominators": real_denominators,
        "imag_numerators": imag_numerators,
        "imag_denominators": imag_denominators,
    }


def _sympy_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import Symbol
    from sympy.polys.domains import QQ_I
    from sympy.polys.rings import ring

    axis = int(payload["axis"])
    symbols = tuple(Symbol(f"x{index}") for index in range(axis))
    polynomial_ring, *_ = ring(symbols, QQ_I)
    left = _load_polynomial(polynomial_ring, payload["left"])
    right = _load_polynomial(polynomial_ring, payload["right"])
    common = left.gcd(right)
    return {
        "left": _dump_polynomial(left.exquo(common)),
        "right": _dump_polynomial(right.exquo(common)),
    }


def main() -> int:
    payload = json.load(sys.stdin)
    if int(payload["axis"]) == 1:
        cancelled = _flint_cancel(payload["left"], payload["right"])
        if cancelled is not None:
            json.dump(cancelled, sys.stdout, separators=(",", ":"))
            return 0
    json.dump(_sympy_cancel(payload), sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
