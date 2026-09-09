"""Prime-field linear-code canonicalization under coordinate actions."""

from itertools import permutations
from math import factorial
from typing import Literal, Self

from pydantic import model_validator
from sympy import isprime

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.operations import _backend_group, _full_permutation_form

MAX_CODE_CANONICALIZATION_ACTION_ORDER = 100_000
MAX_CODE_CANONICALIZATION_RREF_WORK = 20_000_000


class LinearCodeCanonicalizationRequest(StrictModel):
    encoder: PrimeFieldLinearEncoder
    action: Literal["FULL_SYMMETRIC", "SUPPLIED_GROUP"] = "FULL_SYMMETRIC"
    permutation_group: PermutationGroup | None = None

    @model_validator(mode="after")
    def bind_action(self) -> Self:
        if (self.action == "SUPPLIED_GROUP") != (self.permutation_group is not None):
            raise ValueError("SUPPLIED_GROUP requires exactly one permutation_group")
        if self.permutation_group is not None and self.permutation_group.degree != len(
            self.encoder.coordinate_axis
        ):
            raise ValueError("permutation-group degree must match the coordinate axis")
        return self


class LinearCodeCanonicalizationResult(StrictModel):
    source: LinearCodeCanonicalizationRequest
    canonical_encoder: PrimeFieldLinearEncoder
    transporter: tuple[int, ...]
    transported_axis: tuple[str, ...]
    orbit_size: ExactInteger
    stabilizer_size: ExactInteger


def _rref(
    matrix: tuple[tuple[int, ...], ...], prime: int
) -> tuple[tuple[int, ...], ...]:
    rows = [list(row) for row in matrix]
    pivot = 0
    width = len(rows[0]) if rows else 0
    for column in range(width):
        selected = next(
            (row for row in range(pivot, len(rows)) if rows[row][column] % prime),
            None,
        )
        if selected is None:
            continue
        rows[pivot], rows[selected] = rows[selected], rows[pivot]
        inverse = pow(rows[pivot][column], -1, prime)
        rows[pivot] = [value * inverse % prime for value in rows[pivot]]
        for row in range(len(rows)):
            if row == pivot:
                continue
            factor = rows[row][column]
            rows[row] = [
                (left - factor * right) % prime
                for left, right in zip(rows[row], rows[pivot], strict=True)
            ]
        pivot += 1
        if pivot == len(rows):
            break
    return tuple(tuple(row) for row in rows)


def canonicalize_linear_code(
    request: LinearCodeCanonicalizationRequest,
) -> LinearCodeCanonicalizationResult:
    encoder = request.encoder
    if not isprime(encoder.field_order):
        raise OperationDomainValidationError(
            location=("encoder", "field_order"),
            code="code.canonicalization.prime_field",
            message="linear-code canonicalization requires a prime field order",
        )
    source_rref = _rref(encoder.generator_matrix, encoder.field_order)
    if any(not any(row) for row in source_rref):
        raise OperationDomainValidationError(
            location=("encoder", "generator_matrix"),
            code="code.canonicalization.full_row_rank",
            message="linear-code canonicalization requires a full-row-rank encoder",
        )
    width = len(encoder.coordinate_axis)
    if request.action == "FULL_SYMMETRIC":
        order = factorial(width)
        if order > MAX_CODE_CANONICALIZATION_ACTION_ORDER:
            raise OperationResourceAdmissionError(
                location=("encoder", "coordinate_axis"),
                code="code.canonicalization.factorial_bound",
                message="full symmetric coordinate action exceeds its factorial bound",
            )
        elements = tuple(permutations(range(width)))
    else:
        assert request.permutation_group is not None
        backend = _backend_group(request.permutation_group)
        order = int(backend.order())
        if order > MAX_CODE_CANONICALIZATION_ACTION_ORDER:
            raise OperationResourceAdmissionError(
                location=("permutation_group",),
                code="code.canonicalization.group_order_bound",
                message="generated coordinate group exceeds its enumeration bound",
            )
        elements = tuple(
            sorted(
                _full_permutation_form(element, width) for element in backend.elements
            )
        )
    work = order * max(1, len(encoder.message_axis)) * max(1, width * width)
    if work > MAX_CODE_CANONICALIZATION_RREF_WORK:
        raise OperationResourceAdmissionError(
            location=("encoder",),
            code="code.canonicalization.rref_work_bound",
            message="coordinate-action RREF traversal exceeds its admitted work bound",
        )
    candidates: list[
        tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]
    ] = []
    distinct: set[tuple[tuple[int, ...], ...]] = set()
    for element in elements:
        permuted = tuple(
            tuple(row[element[column]] for column in range(width))
            for row in encoder.generator_matrix
        )
        reduced = _rref(permuted, encoder.field_order)
        distinct.add(reduced)
        candidates.append(
            (
                reduced,
                tuple(element),
                tuple(
                    encoder.coordinate_axis[element[column]] for column in range(width)
                ),
            )
        )
    matrix, transporter, transported_axis = min(candidates)
    orbit_size = len(distinct)
    return LinearCodeCanonicalizationResult(
        source=request,
        canonical_encoder=PrimeFieldLinearEncoder(
            field_order=encoder.field_order,
            message_axis=tuple(f"m{index}" for index in range(len(matrix))),
            coordinate_axis=transported_axis,
            generator_matrix=matrix,
        ),
        transporter=transporter,
        transported_axis=transported_axis,
        orbit_size=orbit_size,
        stabilizer_size=order // orbit_size,
    )


__all__ = ["canonicalize_linear_code"]
