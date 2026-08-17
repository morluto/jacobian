"""Exact continued fraction and Pell equation kernels backed by SymPy."""

from __future__ import annotations

__all__ = ["continued_fraction", "convergents", "solve_pell"]


def _cf_coefficients(discriminant: int) -> tuple[list[int], list[int]]:
    """Return (preperiod, period) of the continued fraction of sqrt(D)."""
    from sympy import continued_fraction_periodic

    cf = continued_fraction_periodic(0, 1, discriminant)
    preperiod = [cf[0]] if not isinstance(cf[0], list) else cf[0]
    period = cf[1] if isinstance(cf[1], list) else [cf[1]]
    return (preperiod, period)


def continued_fraction(
    discriminant: int, term_count: int
) -> tuple[list[int], int, int]:
    """Return the continued fraction expansion of sqrt(D).

    Returns (coefficients, preperiod_length, period_length).
    """
    preperiod, period = _cf_coefficients(discriminant)
    full = preperiod + list(period) * ((term_count - len(preperiod)) // len(period) + 2)
    return (full[:term_count], len(preperiod), len(period))


def convergents(discriminant: int, count: int) -> list[tuple[int, int, int]]:
    """Return the first count convergents (index, p_n, q_n) of sqrt(D)."""
    preperiod, period = _cf_coefficients(discriminant)
    cf_coeffs = preperiod + list(period) * 10

    p_prev2, p_prev1 = 1, int(cf_coeffs[0])
    q_prev2, q_prev1 = 0, 1

    result = [(0, int(p_prev1), int(q_prev1))]
    for i in range(1, count):
        a = int(cf_coeffs[i])
        p_curr = a * p_prev1 + p_prev2
        q_curr = a * q_prev1 + q_prev2
        p_prev2, p_prev1 = p_prev1, p_curr
        q_prev2, q_prev1 = q_prev1, q_curr
        result.append((i, int(p_prev1), int(q_prev1)))

    return result


def solve_pell(discriminant: int, max_convergents: int = 10000) -> tuple[int, int]:
    """Solve x^2 - D*y^2 = 1 for the fundamental solution (x, y)."""
    preperiod, period = _cf_coefficients(discriminant)
    cf_coeffs = preperiod + list(period) * 1000

    p_prev2, p_prev1 = 1, int(cf_coeffs[0])
    q_prev2, q_prev1 = 0, 1

    if p_prev1**2 - discriminant * q_prev1**2 == 1:
        return (p_prev1, q_prev1)

    for _i in range(1, max_convergents):
        a = int(cf_coeffs[_i])
        p_curr = a * p_prev1 + p_prev2
        q_curr = a * q_prev1 + q_prev2
        p_prev2, p_prev1 = p_prev1, p_curr
        q_prev2, q_prev1 = q_prev1, q_curr

        if p_prev1**2 - discriminant * q_prev1**2 == 1:
            return (p_prev1, q_prev1)

    msg = f"No Pell solution found within {max_convergents} convergents"
    raise ValueError(msg)
