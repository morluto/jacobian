"""Pre-execution bounds for exact integer Smith certificate replay.

Replay owns a separate envelope from Smith decomposition: all four authored
matrices contribute to admission, including transformation heights. No
mathematical conclusion follows from a refusal of this envelope.
"""

from dataclasses import dataclass
from typing import NoReturn

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_DIMENSION,
    SmithNormalFormCertificate,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_SMITH_VERIFICATION_WORK = 50_000_000_000
MAX_SMITH_VERIFICATION_INTERMEDIATE_BITS = 131_072
MAX_SMITH_VERIFICATION_ALLOCATION_BITS = 16_777_216


@dataclass(frozen=True)
class SmithVerificationPlan:
    work: int
    intermediate_bits: int
    allocation_bits: int


def _resource(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("certificate",),
        code=f"matrix.smith_verification.{reason}",
        message=message,
    )


def _shape_error() -> NoReturn:
    raise OperationDomainValidationError(
        location=("certificate",),
        code="matrix.smith_verification.shape",
        message="Smith replay requires compatible canonical integer matrices",
    )


def _matrix_height(matrix: IntegerMatrix) -> tuple[int, int]:
    if (
        type(matrix.row_count) is not int
        or type(matrix.column_count) is not int
        or matrix.row_count < 0
        or matrix.column_count < 0
    ):
        _shape_error()
    if max(matrix.row_count, matrix.column_count) > MAX_CERTIFIED_SNF_DIMENSION:
        _resource("dimension", "Smith replay supports at most 32 rows and columns")
    if len(matrix.entries) != matrix.row_count:
        _shape_error()
    height = 1
    storage = 0
    for row in matrix.entries:
        request_checkpoint("during Smith certificate replay admission")
        if len(row) != matrix.column_count:
            _shape_error()
        for value in row:
            if type(value) is not int:
                _shape_error()
            bits = max(1, value.bit_length())
            if bits > MAX_SMITH_VERIFICATION_INTERMEDIATE_BITS:
                _resource(
                    "height", "Smith replay input exceeds its bit-height envelope"
                )
            height = max(height, bits)
            storage += bits
    return height, storage


def _product_bound(
    rows: int, inner: int, columns: int, left_bits: int, right_bits: int
) -> tuple[int, int, int]:
    # Each dot product has inner terms of magnitude < 2^(left_bits+right_bits).
    bits = left_bits + right_bits + max(0, (inner - 1).bit_length()) if inner else 1
    operations = rows * inner * columns
    return bits, 2 * operations * bits**2, rows * columns * bits


def _determinant_bound(size: int, height: int) -> tuple[int, int, int]:
    if size == 0:
        return 1, 1, 0
    # Every stored entry of row-pivoted Bareiss is a minor. The Leibniz
    # bound k! * 2^(k*height) bounds every minor for k <= size. Bounding
    # each log2(i) upward avoids floating-point arithmetic. The two
    # products and subtraction before exact division fit 2*minor_bits+1.
    minor_bits = size * height + sum((i - 1).bit_length() for i in range(2, size + 1))
    numerator_bits = 2 * minor_bits + 1
    updates = sum(i * i for i in range(1, size))
    # Two products, subtraction, remainder and quotient per exact division,
    # plus pivoting and determinant sign. Six quadratic-bit units per
    # update cover these five arithmetic operations conservatively.
    work = (6 * updates + size * size + 1) * numerator_bits**2
    return numerator_bits, work, size * size * minor_bits


def admit_smith_verification(
    certificate: SmithNormalFormCertificate,
) -> SmithVerificationPlan:
    request_checkpoint("before Smith certificate replay admission")
    matrices = (
        certificate.source,
        certificate.diagonal,
        certificate.left_transformation,
        certificate.right_transformation,
    )
    if any(not isinstance(matrix, IntegerMatrix) for matrix in matrices):
        _shape_error()
    bounds = tuple(_matrix_height(matrix) for matrix in matrices)
    rows, columns = certificate.source.row_count, certificate.source.column_count
    expected_shapes = (
        (rows, columns),
        (rows, columns),
        (rows, rows),
        (columns, columns),
    )
    if any(
        (matrix.row_count, matrix.column_count) != shape
        for matrix, shape in zip(matrices, expected_shapes, strict=True)
    ):
        _shape_error()
    source_height, diagonal_height, left_height, right_height = (
        bound[0] for bound in bounds
    )
    first = _product_bound(rows, rows, columns, left_height, source_height)
    second = _product_bound(rows, columns, columns, first[0], right_height)
    left_det = _determinant_bound(rows, left_height)
    right_det = _determinant_bound(columns, right_height)
    input_storage = sum(bound[1] for bound in bounds)
    # Input scans/conversion, two products, comparison, and two determinants.
    work = input_storage + sum(
        bound[1] for bound in (first, second, left_det, right_det)
    )
    work += rows * columns * max(second[0], diagonal_height)
    # Conservatively retain all copies/products and both elimination arrays,
    # including backend conversion and temporary multiply/divide numerators.
    allocation = 4 * input_storage + 4 * sum(
        bound[2] for bound in (first, second, left_det, right_det)
    )
    intermediate = max(bound[0] for bound in (first, second, left_det, right_det))
    if intermediate > MAX_SMITH_VERIFICATION_INTERMEDIATE_BITS:
        _resource(
            "intermediate",
            "Smith replay intermediate bit-height bound exceeds its envelope",
        )
    if allocation > MAX_SMITH_VERIFICATION_ALLOCATION_BITS:
        _resource("allocation", "Smith replay allocation bound exceeds its envelope")
    if work > MAX_SMITH_VERIFICATION_WORK:
        _resource("work", "Smith replay exact arithmetic exceeds its work envelope")
    return SmithVerificationPlan(work, intermediate, allocation)
