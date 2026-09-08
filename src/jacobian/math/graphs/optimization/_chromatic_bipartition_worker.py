"""Isolated worker for the bounded chromatic bipartition operation."""

from __future__ import annotations

import json
import sys

from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    _find_chromatic_bipartition_kernel,
)


def main() -> int:
    try:
        request = ChromaticBipartitionRequest.model_validate(
            json.loads(sys.stdin.buffer.read())
        )
        result = _find_chromatic_bipartition_kernel(request)
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
