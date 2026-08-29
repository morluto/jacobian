"""Isolated worker for exact binomial and permutation evaluation."""

from __future__ import annotations

import json
import math
import sys

from jacobian.canonical import format_canonical_integer


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read())
        operation = payload["op"]
        n = int(payload["n"])
        k = int(payload["k"])
        if operation == "comb":
            value = math.comb(n, k)
        elif operation == "perm":
            value = math.perm(n, k)
        else:
            return 2
        sys.stdout.write(format_canonical_integer(value))
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
