"""One-shot isolated SymPy factorization worker."""

from __future__ import annotations

import hashlib
import sys

from jacobian.canonical import encode_strict_json
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _factorize_certified_in_process,
)


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    request = CertifiedFactorizationRequest.model_validate_json(
        input_bytes, strict=True
    )
    response: dict[str, object] = {
        "ok": True,
        "result": _factorize_certified_in_process(request).model_dump(mode="json"),
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
    }
    sys.stdout.buffer.write(encode_strict_json(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
