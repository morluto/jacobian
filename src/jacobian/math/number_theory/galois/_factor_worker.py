"""One deterministic finite-field factorization inside a killable worker."""

import json
import sys


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
    sys.stdout.write(json.dumps([int(unit), factors], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
