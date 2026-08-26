"""Isolated SymPy adapter for one admitted direct factorization request."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read())
        value = int(payload["value"])
        if value == 0:
            raise ValueError("zero has no finite direct factorization")
        from sympy import factorint

        factors = sorted(factorint(abs(value)).items())
        sys.stdout.write(
            json.dumps(
                {"factors": [[str(prime), int(power)] for prime, power in factors]},
                separators=(",", ":"),
            )
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
