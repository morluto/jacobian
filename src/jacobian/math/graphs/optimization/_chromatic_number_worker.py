"""Isolated Z3 adapter for the bounded chromatic-number operation."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    request_execution,
)
from jacobian._worker_errors import bind_worker_deadline, worker_execution_errors
from jacobian.math.graphs.optimization._chromatic_number import (
    _search_chromatic_number_kernel,
)
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberRequest,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
        bind_worker_deadline(payload)
        request = GraphChromaticNumberRequest.model_validate(payload)
        result = _search_chromatic_number_kernel(request)
        sys.stdout.write(
            json.dumps(
                result.model_dump(mode="json", exclude={"vertices"}),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


if __name__ == "__main__":
    with worker_execution_errors(), request_execution(time.monotonic()):
        raise SystemExit(main())
