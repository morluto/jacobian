"""Bounded worker entry point for one rational cyclotomic kernel call."""

from __future__ import annotations

import sys
from hashlib import sha256

from jacobian.math.matrices.cyclic_linear._kernel_process import (
    _decode_requests,
    _encode_response,
)
from jacobian.math.matrices.cyclic_linear.operations import _cyclotomic_kernel_child


def main() -> None:
    input_data = sys.stdin.buffer.read()
    requests = _decode_requests(input_data)
    results = tuple(
        _cyclotomic_kernel_child(order, degree, matrix_coordinates, common_denominator)
        for order, degree, matrix_coordinates, common_denominator in requests
    )
    sys.stdout.buffer.write(_encode_response(sha256(input_data).digest(), results))


if __name__ == "__main__":
    main()
