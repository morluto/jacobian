"""Isolated Z3 adapter for the exact maximum-cut operation."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs.optimization._maximum_cut import (
    GraphMaximumCutRequest,
    compute_maximum_cut,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        request = GraphMaximumCutRequest.model_validate(payload)
        result = compute_maximum_cut(request)
        sys.stdout.write(
            json.dumps(
                result.model_dump(mode="json"),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
