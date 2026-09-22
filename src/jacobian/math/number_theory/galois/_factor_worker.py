"""One deterministic finite-field factorization inside a killable worker."""

import json
import sys

from jacobian._worker_protocol import encode_worker_result_frame


def main() -> int:
    from sympy.polys.domains import ZZ
    from sympy.polys.galoistools import gf_berlekamp, gf_sqf_list

    prime, coefficients = json.loads(sys.stdin.buffer.read())
    unit, squarefree = gf_sqf_list(list(reversed(coefficients)), prime, ZZ)
    factors = []
    for polynomial, multiplicity in squarefree:
        for factor in gf_berlekamp(polynomial, prime, ZZ):
            factors.append((tuple(int(c) for c in reversed(factor)), int(multiplicity)))
    factors.sort(key=lambda item: (len(item[0]), item[0]))
    sys.stdout.buffer.write(
        encode_worker_result_frame([int(unit), factors])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
