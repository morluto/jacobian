"""Typed wire contracts for coding theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

# One source-bound call performs its exhaustive work twice: the domain
# function computes the claimed value and the retained-source result
# validator replays it. The enumeration and transition budgets below are
# therefore totals charged across both passes per accepted call;
# MAX_COVERING_RADIUS_STATES_PER_PASS stays a per-pass bound because each
# BFS allocates its visited set independently.
EXACT_ENUMERATION_PASSES = 2
SYNDROME_BFS_PASSES = 2
MAX_EXACT_CODEWORD_EVALUATIONS = 131_072
MAX_COVERING_RADIUS_STATES_PER_PASS = 65_536
MAX_COVERING_RADIUS_TRANSITIONS = 4_000_000


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
        if (
            EXACT_ENUMERATION_PASSES * self.field_order ** len(self.generator_matrix)
            > MAX_EXACT_CODEWORD_EVALUATIONS
        ):
            raise _error(
                "code_theory.enumeration_work_exceeded",
                "generator matrix exceeds the exact enumeration bound",
            )
        return self


class MinimumDistanceResult(StrictModel):
    """The exact minimum nonzero codeword weight bound to its source code.

    Retains the canonical source code (prime field order and generator
    matrix) so validation replays the exact enumeration: the claimed
    distance lies in ``[0, n]`` and equals the minimum nonzero Hamming
    weight over the distinct generated codewords of the retained
    generator. For the zero code (rank 0) the generated codeword set is
    ``{0}``, so no nonzero codeword exists; the claimed distance is then
    the code length ``n`` by the empty-code convention and is not the
    weight of any generated word.
    """

    request: LinearCodeRequest
    minimum_distance: int = Field(ge=0, le=256)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.code_theory.operations import (
            minimum_distance as replay_minimum_distance,
        )

        width = len(self.request.generator_matrix[0])
        if self.minimum_distance > width:
            raise _error(
                "code_theory.minimum_distance_out_of_bounds",
                "minimum distance must lie in [0, code length]",
            )
        if (
            replay_minimum_distance(
                self.request.generator_matrix, self.request.field_order
            )
            != self.minimum_distance
        ):
            raise _error(
                "code_theory.minimum_distance_replay_mismatch",
                "minimum distance must be the exact enumeration of the "
                "retained source code",
            )
        return self


class WeightDistributionResult(StrictModel):
    """The exact weight distribution bound to its source code.

    Retains the canonical source code so validation replays the defining
    relations: rows are canonically ordered with positive counts and weights
    in ``[0, n]``, counts sum to the number of distinct generated codewords
    (``q^rank``), and the profile equals the exact enumeration.
    """

    request: LinearCodeRequest
    weights: tuple[tuple[int, int], ...] = Field(max_length=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.code_theory._models import _matrix_rank_mod_prime
        from jacobian.math.code_theory.operations import (
            weight_distribution as replay_weight_distribution,
        )

        width = len(self.request.generator_matrix[0])
        seen_weights: list[int] = []
        for weight, count in self.weights:
            if not 0 <= weight <= width:
                raise _error(
                    "code_theory.weight_out_of_bounds",
                    "weight rows must lie in [0, code length]",
                )
            if count < 1:
                raise _error(
                    "code_theory.weight_count_not_positive",
                    "weight counts must be positive",
                )
            if seen_weights and weight <= seen_weights[-1]:
                raise _error(
                    "code_theory.weights_not_strictly_ascending",
                    "weight rows must be strictly ascending and unique",
                )
            seen_weights.append(weight)
        rank = _matrix_rank_mod_prime(
            self.request.generator_matrix, self.request.field_order
        )
        expected_total = self.request.field_order**rank
        if sum(count for _weight, count in self.weights) != expected_total:
            raise _error(
                "code_theory.weight_count_total_mismatch",
                "weight counts must sum to the distinct generated codeword count",
            )
        replayed = replay_weight_distribution(
            self.request.generator_matrix, self.request.field_order
        )
        if tuple((w, c) for w, c in replayed) != self.weights:
            raise _error(
                "code_theory.weight_distribution_replay_mismatch",
                "weight distribution must be the exact enumeration of the "
                "retained source code",
            )
        return self


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
        if state_count > MAX_COVERING_RADIUS_STATES_PER_PASS:
            raise _error(
                "code_theory.syndrome_state_bound_exceeded",
                "syndrome space exceeds the exact state bound",
            )
        move_count_bound = min(
            width * (self.field_order - 1),
            max(state_count - 1, 0),
        )
        if (
            SYNDROME_BFS_PASSES * state_count * move_count_bound
            > MAX_COVERING_RADIUS_TRANSITIONS
        ):
            raise _error(
                "code_theory.syndrome_transition_bound_exceeded",
                "syndrome graph exceeds the exact transition bound",
            )
        return self


class CoveringRadiusResult(StrictModel):
    """The exact covering radius bound to its source code.

    Retains the canonical source code (the same bounded syndrome-graph
    request) so validation replays the exact BFS: the radius lies in
    ``[0, n]`` and equals the maximum minimum-error weight over the
    retained source's syndrome graph.
    """

    request: CoveringRadiusRequest
    covering_radius: int = Field(ge=0, le=256)
    method: Literal["SYNDROME_BFS"] = "SYNDROME_BFS"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.code_theory.operations import (
            covering_radius as replay_covering_radius,
        )

        width = len(self.request.generator_matrix[0])
        if self.covering_radius > width:
            raise _error(
                "code_theory.covering_radius_out_of_bounds",
                "covering radius must lie in [0, code length]",
            )
        if (
            replay_covering_radius(
                self.request.generator_matrix, self.request.field_order
            )
            != self.covering_radius
        ):
            raise _error(
                "code_theory.covering_radius_replay_mismatch",
                "covering radius must be the exact syndrome BFS of the "
                "retained source code",
            )
        return self
