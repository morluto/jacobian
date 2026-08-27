"""One-shot isolated SymPy factorization worker."""

from __future__ import annotations

import json
import sys

from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _factorize_certified_in_process,
)


def main() -> int:
    request = CertifiedFactorizationRequest.model_validate(json.load(sys.stdin))
    response: dict[str, object] = {
        "ok": True,
        "result": _factorize_certified_in_process(request).model_dump(mode="json"),
    }
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
