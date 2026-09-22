"""Isolated maintained-backend adapter for discrepancy minimization."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from jacobian._execution import request_execution
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem
from jacobian.math.combinatorics.discrepancy.operations import (
    compute_optimal_discrepancy,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            return 2
        deadline = payload.pop("_deadline", None)
        if not isinstance(deadline, (int, float)):
            return 2
        set_system = FiniteSetSystem.model_validate(payload)
        with request_execution(time.monotonic(), outer_deadline=float(deadline)):
            result = compute_optimal_discrepancy(set_system)
        sys.stdout.write(
            json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
