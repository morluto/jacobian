"""Isolated maintained-backend adapter for discrepancy minimization."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.discrepancy_theory._models import DiscrepancyOptimumRequest
from jacobian.math.discrepancy_theory._operations import compute_optimal_discrepancy


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        request = DiscrepancyOptimumRequest.model_validate(payload)
        result = compute_optimal_discrepancy(request)
        sys.stdout.write(
            json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
