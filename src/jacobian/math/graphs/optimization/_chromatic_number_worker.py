"""Isolated Z3 adapter for the bounded chromatic-number operation."""

from __future__ import annotations

import json
import sys
from typing import Any

from jacobian.math.graphs.optimization._chromatic_number import (
    _search_chromatic_number_kernel,
)
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberRequest,
)


def main() -> int:
    try:
        payload: Any = json.loads(sys.stdin.buffer.read())
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
    except (TypeError, ValueError, json.JSONDecodeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
