"""Isolated Z3 adapter for one SMT-core extraction or replay request."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.logic._unsat_core import (
    SmtUnsatCoreRequest,
    _unsat_core_worker_kernel,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("worker payload must be an object")
        request = SmtUnsatCoreRequest.model_validate(payload["request"])
        raw_selected_indices = payload.get("selected_indices")
        if raw_selected_indices is None:
            selected_indices = None
        elif isinstance(raw_selected_indices, list) and all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in raw_selected_indices
        ):
            selected_indices = tuple(raw_selected_indices)
        else:
            raise ValueError("worker payload has invalid selected indices")
        response = _unsat_core_worker_kernel(
            request,
            selected_indices=selected_indices,
        )
        sys.stdout.write(json.dumps(response, separators=(",", ":")))
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
