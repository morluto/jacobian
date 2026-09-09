"""Isolated worker for the bounded chromatic bipartition operation."""

from __future__ import annotations

import json
import sys
import time

from jacobian._execution import request_execution
from jacobian._worker_errors import bind_worker_deadline, worker_execution_errors
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    _find_chromatic_bipartition_kernel,
)
from jacobian.process import encode_worker_result_frame


def main() -> int:
    try:
        with request_execution(time.monotonic()), worker_execution_errors():
            payload = json.loads(sys.stdin.buffer.read())
            bind_worker_deadline(payload)
            request = ChromaticBipartitionRequest.model_validate(payload)
            result = _find_chromatic_bipartition_kernel(request)
            sys.stdout.buffer.write(
                encode_worker_result_frame(result.model_dump(mode="json"))
            )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
