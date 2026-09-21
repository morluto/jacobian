"""Killable SymPy root isolation for one scaled radix prefix."""

from __future__ import annotations

import json
import sys
from fractions import Fraction

from jacobian._worker_protocol import encode_worker_result_frame


def _unique_floor_of_open_interval(lower: Fraction, upper: Fraction) -> int | None:
    floor_lower = lower.numerator // lower.denominator
    ceil_upper = -((-upper.numerator) // upper.denominator)
    greatest_integer_below_upper = ceil_upper - 1
    return int(floor_lower) if floor_lower == greatest_integer_below_upper else None


def _scaled_integer_part(
    coefficients: list[int],
    real_root_index: int,
    scale: int,
    isolation_bits: int,
) -> int:
    import sympy

    symbol = sympy.Symbol("x")
    source = sympy.Poly.from_list(
        [int(coefficient) for coefficient in coefficients],
        gens=symbol,
        domain=sympy.ZZ,
    )
    if source.is_irreducible is not True:
        raise ValueError(
            "real algebraic minimal polynomial must be irreducible over QQ"
        )
    scaled_coefficients = [
        coefficient * scale**position
        for position, coefficient in enumerate(coefficients)
    ]
    polynomial = sympy.Poly.from_list(
        [int(coefficient) for coefficient in scaled_coefficients],
        gens=symbol,
        domain=sympy.ZZ,
    )
    intervals = polynomial.intervals()
    if real_root_index >= len(intervals):
        raise ValueError("real_root_index must select an existing real root")
    lower, upper = intervals[real_root_index][0]
    max_refinements = max(4_096, (isolation_bits + 7) // 8)
    for _ in range(max_refinements):
        if (candidate := _unique_floor_of_open_interval(lower, upper)) is not None:
            return candidate
        lower, upper = polynomial.refine_root(lower, upper, steps=8)
    raise TimeoutError(
        "root isolation did not separate the scaled value from an integer"
    )


def main() -> int:
    payload = json.loads(sys.stdin.read())
    try:
        scaled_floor = _scaled_integer_part(
            [int(coefficient) for coefficient in payload["polynomial"]],
            int(payload["real_root_index"]),
            int(payload["scale"]),
            int(payload["isolation_bits"]),
        )
    except ValueError as exc:
        message = str(exc)
        code = "not_irreducible" if "irreducible" in message else "root_index"
        sys.stdout.buffer.write(
            encode_worker_result_frame(
                {"tag": "domain_error", "code": code, "message": message}
            )
        )
        return 0
    except TimeoutError as exc:
        sys.stdout.buffer.write(
            encode_worker_result_frame({"tag": "refinement", "message": str(exc)})
        )
        return 0
    sys.stdout.buffer.write(
        encode_worker_result_frame(
            {"tag": "success", "scaled_floor": str(int(scaled_floor))}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
