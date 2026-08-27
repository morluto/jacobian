"""Isolated Z3 adapter for one canonical SAT request."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.logic._sat import SatSolveRequest, _solve_sat_kernel


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        request = SatSolveRequest.model_validate(payload)
        response = _solve_sat_kernel(cnf=request.cnf, timeout_ms=request.timeout_ms)
        sys.stdout.write(json.dumps(response, separators=(",", ":")))
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
