"""Killable SymPy worker for one exact algebraic sign."""

from __future__ import annotations

import hashlib
import sys

from sympy import QQ, Poly, Symbol

from jacobian.canonical import (
    encode_strict_json,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math._root_isolation import strict_root_count


def _fraction(value: object) -> object:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError("worker coefficient must be a fraction string")
    numerator, denominator = value.split("/")
    return QQ(
        parse_canonical_integer(numerator),
        parse_canonical_integer(denominator),
    )


def main() -> None:
    input_bytes = sys.stdin.buffer.read()
    payload = loads_strict_json(input_bytes)
    presentation = payload["presentation"]
    coefficients = presentation["coefficients_descending"]
    root_index = payload["real_root_index"]
    value_coefficients = payload["coefficients_descending"]
    x = Symbol("x")
    defining = Poly.from_list(
        [parse_canonical_integer(coefficient) for coefficient in coefficients],
        gens=x,
        domain=QQ,
    )
    representative = Poly.from_list(
        [_fraction(coefficient) for coefficient in value_coefficients],
        gens=x,
        domain=QQ,
    )
    if representative.is_zero:
        sign = 0
    else:
        roots_seen = 0
        sign = None
        for (lower, upper), _multiplicity in (defining * representative).intervals():
            if not strict_root_count(defining, lower, upper):
                continue
            if roots_seen != root_index:
                roots_seen += 1
                continue
            sample = lower if lower == upper else (lower + upper) / 2
            exact_value = representative.eval(sample)
            sign = 1 if exact_value > 0 else -1 if exact_value < 0 else 0
            break
        if sign is None:
            raise ValueError("selected real embedding was not isolated")
    sys.stdout.buffer.write(
        encode_strict_json(
            {
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                "sign": sign,
            }
        )
    )


if __name__ == "__main__":
    main()
