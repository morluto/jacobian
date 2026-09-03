"""Bounded worker entry point for one rational cyclotomic kernel call."""

from __future__ import annotations

import pickle
import sys
from hashlib import sha256

from jacobian.math.matrices.cyclic_linear.operations import _cyclotomic_kernel_child


def main() -> None:
    input_data = sys.stdin.buffer.read()
    requests = pickle.loads(input_data)
    results = tuple(
        _cyclotomic_kernel_child(order, degree, matrix_coordinates, common_denominator)
        for order, degree, matrix_coordinates, common_denominator in requests
    )
    pickle.dump((sha256(input_data).digest(), results), sys.stdout.buffer)


if __name__ == "__main__":
    main()
