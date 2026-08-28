"""Typed wire contracts for coding theory operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

# Each operation selects one exact kernel path. Result construction itself
# never re-enters that path.
EXACT_ENUMERATION_PASSES = 1
SYNDROME_BFS_PASSES = 1
MAX_EXACT_CODEWORD_EVALUATIONS = 131_072
MAX_COVERING_RADIUS_STATES_PER_PASS = 65_536
MAX_COVERING_RADIUS_TRANSITIONS = 2_000_000

_WeightCount = Annotated[
    StrictInt,
    Field(ge=1, le=MAX_EXACT_CODEWORD_EVALUATIONS),
]


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _validate_prime_field_matrix(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
) -> int:
    from sympy import isprime

    if not isprime(field_order):
        raise _error(
            "code_theory.field_order_not_prime",
            "field_order must be prime for this prime-field operation",
        )
    width = len(generator_matrix[0])
    if width == 0 or width > 256:
        raise _error(
            "code_theory.generator_width_out_of_bounds",
            "generator rows must have between one and 256 entries",
        )
    if any(len(row) != width for row in generator_matrix):
        raise _error(
            "code_theory.generator_rows_unequal",
            "generator matrix rows must have equal length",
        )
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise _error(
            "code_theory.generator_entry_not_canonical",
            "generator entries must be canonical field residues",
        )
    return width


def _matrix_rank_mod_prime(
    matrix: tuple[tuple[int, ...], ...],
    field_order: int,
) -> int:
    rows = [list(row) for row in matrix]
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(pivot_row, row_count)
                if rows[index][column] % field_order != 0
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % field_order, -1, field_order)
        rows[pivot_row] = [value * inverse % field_order for value in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index == pivot_row:
                continue
            factor = row[column] % field_order
            if factor == 0:
                continue
            rows[index] = [
                (left - factor * right) % field_order
                for left, right in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


class LinearCodeRequest(StrictModel):
    """A linear code given by its generator matrix over one bounded prime field."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_bounded_prime_field_matrix(self) -> Self:
        _validate_prime_field_matrix(self.field_order, self.generator_matrix)
        return self


class MinimumDistanceResult(StrictModel):
    """An exact minimum-distance claim in its canonical source coordinates.

    Construction checks the bounded scalar relation only. For the zero code,
    the kernel uses the documented empty-code convention.
    """

    request: LinearCodeRequest
    minimum_distance: int = Field(ge=0, le=256)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def require_bounded_distance(self) -> Self:
        width = len(self.request.generator_matrix[0])
        if self.minimum_distance > width:
            raise _error(
                "code_theory.minimum_distance_out_of_bounds",
                "minimum distance must lie in [0, code length]",
            )
        return self

    @classmethod
    def _from_kernel(cls, *, request: LinearCodeRequest, minimum_distance: int) -> Self:
        """Build a result produced by the owner-local enumeration kernel."""

        return cls(request=request, minimum_distance=minimum_distance)


class WeightDistributionResult(StrictModel):
    """An exact weight-distribution claim in canonical source coordinates.

    Construction checks only row shape, bounds, and canonical ordering.
    """

    request: LinearCodeRequest
    weights: tuple[tuple[StrictInt, _WeightCount], ...] = Field(
        min_length=1,
        max_length=257,
    )
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def require_structural_weight_rows(self) -> Self:
        width = len(self.request.generator_matrix[0])
        seen_weights: list[int] = []
        for weight, _count in self.weights:
            if not 0 <= weight <= width:
                raise _error(
                    "code_theory.weight_out_of_bounds",
                    "weight rows must lie in [0, code length]",
                )
            if seen_weights and weight <= seen_weights[-1]:
                raise _error(
                    "code_theory.weights_not_strictly_ascending",
                    "weight rows must be strictly ascending and unique",
                )
            seen_weights.append(weight)
        return self

    @classmethod
    def _from_kernel(
        cls, *, request: LinearCodeRequest, weights: tuple[tuple[int, int], ...]
    ) -> Self:
        """Build a profile produced by the owner-local enumeration kernel."""

        return cls(request=request, weights=weights)


class CoveringRadiusRequest(StrictModel):
    """A linear code whose syndrome graph fits declared exact-work bounds."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_prime_field_matrix(self) -> Self:
        _validate_prime_field_matrix(self.field_order, self.generator_matrix)
        return self


class CoveringRadiusResult(StrictModel):
    """An exact covering-radius claim in canonical source coordinates.

    Construction checks the bounded scalar relation only.
    """

    request: CoveringRadiusRequest
    covering_radius: int = Field(ge=0, le=256)
    method: Literal["SYNDROME_BFS"] = "SYNDROME_BFS"

    @model_validator(mode="after")
    def require_bounded_radius(self) -> Self:
        width = len(self.request.generator_matrix[0])
        if self.covering_radius > width:
            raise _error(
                "code_theory.covering_radius_out_of_bounds",
                "covering radius must lie in [0, code length]",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, request: CoveringRadiusRequest, covering_radius: int
    ) -> Self:
        """Build a result produced by the owner-local syndrome kernel."""

        return cls(request=request, covering_radius=covering_radius)
