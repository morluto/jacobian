"""Bounded worker entry point for one rational cyclotomic kernel call."""

from __future__ import annotations

import pickle
import sys

from jacobian.math.matrices.cyclic_linear.operations import _cyclotomic_kernel_child


def main() -> None:
    order, degree, matrix_coordinates, common_denominator = pickle.load(
        sys.stdin.buffer
    )
    result = _cyclotomic_kernel_child(
        order, degree, matrix_coordinates, common_denominator
    )
    pickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    main()
