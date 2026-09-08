"""Killable worker for one already-admitted exact discrepancy decision."""

import json
import sys

from jacobian.math.combinatorics.discrepancy.bounded_coloring._z3 import solve


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read())
        rows = tuple((tuple(row[0]), row[1], row[2]) for row in payload["constraints"])
        reply = solve(
            payload["variable_count"], rows, payload["work_limit"], payload["deadline"]
        )
        sys.stdout.write(json.dumps(reply, separators=(",", ":")))
        return 0
    except (KeyError, TypeError, ValueError, MemoryError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
