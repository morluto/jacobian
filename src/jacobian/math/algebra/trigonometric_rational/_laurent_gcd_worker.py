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

    ``QQ[z, I]`` treats ``I`` as transcendental, so a flint GCD is only a
    candidate: it can miss a common factor that exists over ``QQ(i)``. The
    candidate is therefore validated by checking the cofactors are coprime in
    the true field, and a disagreement returns ``None`` so the caller uses the
    ``QQ_I`` ring path.
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
        left_result = _flint_to_payload(left_quotient, minimum_exponent)
        right_result = _flint_to_payload(right_quotient, minimum_exponent)
        if not _coprime_over_gaussian_rational(left_result, right_result):
            return None
    except Exception:
        return None
    return {"left": left_result, "right": right_result}


def _coprime_over_gaussian_rational(
    left_payload: dict[str, list[Any]], right_payload: dict[str, list[Any]]
) -> bool:
    """Whether two Laurent payloads are coprime over ``QQ(i)[z]``.

    A necessary condition is that the norms (real polynomials over ``QQ``) are
    coprime; the real flint GCD is fast even for sparse high-degree operands, so
    it clears the common no-shared-factor case immediately. Only when the norms
    share a factor is the exact ``QQ(i)`` GCD computed.
    """

    if _gaussian_norms_coprime(left_payload, right_payload):
        return True
    # The norms share a factor, so the true Gaussian GCD may be trivial. Reduce
    # modulo a prime where -1 is a square and compute the GCD over GF(p)[z]; a
    # unit there means the Gaussian operands are coprime with overwhelming
    # probability, avoiding the slow exact ring GCD.
    modular = _modular_gaussian_gcd_degree(left_payload, right_payload)
    if modular == 0:
        return True
    from sympy import I, Rational, Symbol
    from sympy.polys.domains import QQ_I
    from sympy.polys.rings import ring

    def _load(payload: dict[str, list[Any]], polynomial_ring: Any) -> Any:
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
        return polynomial_ring.from_dict(mapping)

    symbol = Symbol("z")
    polynomial_ring, *_ = ring((symbol,), QQ_I)
    left = _load(left_payload, polynomial_ring)
    right = _load(right_payload, polynomial_ring)
    if left.is_zero or right.is_zero:
        return False
    common = left.gcd(right)
    return bool(common.degree() == 0)


def _gaussian_norms_coprime(
    left_payload: dict[str, list[Any]], right_payload: dict[str, list[Any]]
) -> bool:
    """Fast necessary coprimality test through the real polynomial norms."""

    from flint import fmpq, fmpq_poly

    def _coeffs(
        payload: dict[str, list[Any]], numerator_key: str, denominator_key: str
    ) -> list[Any]:
        coefficients: dict[int, Any] = {}
        for support, num, den in zip(
            payload["supports"],
            payload[numerator_key],
            payload[denominator_key],
            strict=True,
        ):
            coefficients[int(support[0])] = fmpq(int(num), int(den))
        length = max(coefficients, default=0) + 1
        return [coefficients.get(index, fmpq(0)) for index in range(length)]

    def _norm(payload: dict[str, list[Any]]) -> Any:
        # N(a + i b) = a^2 + b^2 as a polynomial over QQ.
        real = fmpq_poly(_coeffs(payload, "real_numerators", "real_denominators"))
        imag = fmpq_poly(_coeffs(payload, "imag_numerators", "imag_denominators"))
        return real * real + imag * imag

    try:
        left_norm = _norm(left_payload)
        right_norm = _norm(right_payload)
    except Exception:
        return False
    if left_norm.is_zero() or right_norm.is_zero():
        return False
    return bool(left_norm.gcd(right_norm).degree() == 0)


def _modular_gaussian_gcd_degree(
    left_payload: dict[str, list[Any]], right_payload: dict[str, list[Any]]
) -> int | None:
    """GCD degree over ``GF(p)[z]`` with ``i`` mapped to a square root of -1.

    Returns 0 when the reduction proves coprimality, a positive degree when a
    shared factor reduces nontrivially, and ``None`` when the reduction is not
    usable and the exact ``QQ(i)`` path must run.
    """

    from flint import nmod_poly

    prime = 998244353
    imaginary_root = 911660635  # a square root of -1 modulo the prime

    def _reduce(payload: dict[str, list[Any]]) -> Any:
        coefficients: dict[int, int] = {}
        for support, real_num, real_den, imag_num, imag_den in zip(
            payload["supports"],
            payload["real_numerators"],
            payload["real_denominators"],
            payload["imag_numerators"],
            payload["imag_denominators"],
            strict=True,
        ):
            real = (int(real_num) * pow(int(real_den), -1, prime)) % prime
            imag = (int(imag_num) * pow(int(imag_den), -1, prime)) % prime
            coefficients[int(support[0])] = (real + imag * imaginary_root) % prime
        length = max(coefficients, default=0) + 1
        return nmod_poly([coefficients.get(index, 0) for index in range(length)], prime)

    try:
        left = _reduce(left_payload)
        right = _reduce(right_payload)
    except Exception:
        return None
    if left.is_zero() or right.is_zero():
        return None
    # The reduction must preserve both degrees: an unlucky prime that kills a
    # leading coefficient can erase a common factor and falsely report
    # coprimality, so fall back to the exact field in that case.
    left_degree = max(int(support[0]) for support in left_payload["supports"])
    right_degree = max(int(support[0]) for support in right_payload["supports"])
    if left.degree() != left_degree or right.degree() != right_degree:
        return None
    return int(left.gcd(right).degree())


def _flint_to_payload(polynomial: Any, minimum_exponent: int) -> dict[str, list[Any]]:
    """Convert a bivariate ``QQ[z, I]`` polynomial back to the wire payload.

    A term ``c * I * z**e`` (imaginary degree one) contributes ``c`` to the
    imaginary component and nothing to the real component. Terms that share a
    ``z`` exponent with opposite imaginary degrees aggregate into one row so no
    support is emitted twice.
    """

    from flint import fmpq

    real_by_exponent: dict[int, Any] = {}
    imag_by_exponent: dict[int, Any] = {}
    for exponents, coefficient in polynomial.terms():
        exponent = int(exponents[0])
        imaginary_degree = int(exponents[1])
        if imaginary_degree not in (0, 1):
            raise ValueError("unexpected imaginary monomial degree")
        target = imag_by_exponent if imaginary_degree else real_by_exponent
        target[exponent] = target.get(exponent, fmpq(0)) + coefficient
    supports: list[list[int]] = []
    real_numerators: list[str] = []
    real_denominators: list[str] = []
    imag_numerators: list[str] = []
    imag_denominators: list[str] = []
    for exponent in sorted(set(real_by_exponent) | set(imag_by_exponent)):
        real = real_by_exponent.get(exponent, fmpq(0))
        imaginary = imag_by_exponent.get(exponent, fmpq(0))
        if not real and not imaginary:
            continue
        supports.append([exponent + minimum_exponent])
        real_numerators.append(str(real.p))
        real_denominators.append(str(real.q))
        imag_numerators.append(str(imaginary.p))
        imag_denominators.append(str(imaginary.q))
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
