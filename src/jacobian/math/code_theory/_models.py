"""Typed wire contracts for coding theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_EXACT_CODEWORDS = 65_536
MAX_COVERING_RADIUS_STATES = 65_536
MAX_COVERING_RADIUS_TRANSITIONS = 2_000_000


def _validate_prime_field_matrix(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
) -> int:
    from sympy import isprime

    if not isprime(field_order):
        raise ValueError("field_order must be prime for this prime-field operation")
    width = len(generator_matrix[0])
    if width == 0 or width > 256:
        raise ValueError("generator rows must have between one and 256 entries")
    if any(len(row) != width for row in generator_matrix):
        raise ValueError("generator matrix rows must have equal length")
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise ValueError("generator entries must be canonical field residues")
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
        if self.field_order ** len(self.generator_matrix) > MAX_EXACT_CODEWORDS:
            raise ValueError("generator matrix exceeds the exact enumeration bound")
        return self


class MinimumDistanceResult(StrictModel):
    minimum_distance: int = Field(ge=0, le=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"


class WeightDistributionResult(StrictModel):
    weights: tuple[tuple[int, int], ...] = Field(max_length=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"


class CoveringRadiusRequest(StrictModel):
    """A linear code whose syndrome graph fits declared exact-work bounds."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_bounded_syndrome_graph(self) -> Self:
        width = _validate_prime_field_matrix(
            self.field_order,
            self.generator_matrix,
        )
        rank = _matrix_rank_mod_prime(self.generator_matrix, self.field_order)
        syndrome_dimension = width - rank
        state_count = self.field_order**syndrome_dimension
        if state_count > MAX_COVERING_RADIUS_STATES:
            raise ValueError("syndrome space exceeds the exact state bound")
        move_count_bound = min(
            width * (self.field_order - 1),
            max(state_count - 1, 0),
        )
        if state_count * move_count_bound > MAX_COVERING_RADIUS_TRANSITIONS:
            raise ValueError("syndrome graph exceeds the exact transition bound")
        return self


class CoveringRadiusResult(StrictModel):
    covering_radius: int = Field(ge=0, le=256)
    method: Literal["SYNDROME_BFS"] = "SYNDROME_BFS"
