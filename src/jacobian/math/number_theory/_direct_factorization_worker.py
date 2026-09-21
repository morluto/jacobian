"""Isolated SymPy adapter for one admitted direct factorization request."""

from __future__ import annotations

import sys

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import (
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)


def main() -> int:
    try:
        payload = loads_strict_json(sys.stdin.buffer.read())
        value = parse_canonical_integer(payload["value"])
        if value == 0:
            raise ValueError("zero has no finite direct factorization")
        from sympy import factorint

        factors = sorted(factorint(abs(value)).items())
        sys.stdout.buffer.write(
            encode_worker_result_frame(
                {
                    "factors": [
                        [format_canonical_integer(int(prime)), int(power)]
                        for prime, power in factors
                    ]
                }
            )
        )
        return 0
    except (KeyError, TypeError, ValueError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
