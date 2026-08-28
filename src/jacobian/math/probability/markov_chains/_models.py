"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MARKOV_STATES = 32
MAX_MIXING_STEPS = 256
DEFAULT_MIXING_STEPS = 64


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Markov-chain contracts."""

    return PydanticCustomError(f"markov_chain.{reason}", message)


class TransitionMatrixRequest(StrictModel):
    """A finite stochastic transition matrix with rational entries."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=MAX_MARKOV_STATES
    )

    @model_validator(mode="after")
    def require_stochastic_square_matrix(self) -> Self:
        dimension = len(self.matrix)
        if any(len(row) != dimension for row in self.matrix):
            raise _validation_error(
                "transition_matrix_not_square", "transition matrix must be square"
            )
        for row in self.matrix:
            values = tuple(value.as_fraction() for value in row)
            if any(value < 0 for value in values):
                raise _validation_error(
                    "transition_probability_negative",
                    "transition probabilities must be nonnegative",
                )
            if sum(values) != 1:
                raise _validation_error(
                    "transition_row_not_stochastic",
                    "each transition row must sum to one",
                )
        return self


class StationaryDistributionRequest(TransitionMatrixRequest):
    """A transition matrix whose exact stationary solutions fit the wire contract."""


class ExtremeStationaryDistribution(StrictModel):
    """One canonical extreme point supported on a closed class."""

    closed_class: tuple[int, ...] = Field(min_length=1)
    distribution: tuple[CanonicalRational, ...] = Field(min_length=1)


class StationaryDistributionResult(StrictModel):
    """Extreme points of the finite chain's stationary-distribution simplex."""

    transition_matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=MAX_MARKOV_STATES
    )
    """The source transition matrix whose stationary simplex was computed."""

    extreme_distributions: tuple[ExtremeStationaryDistribution, ...] = Field(
        min_length=1
    )
    unique: bool
    method: Literal["CLOSED_CLASS_EXACT_LINEAR_SYSTEM"] = (
        "CLOSED_CLASS_EXACT_LINEAR_SYSTEM"
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        transition_matrix: tuple[tuple[CanonicalRational, ...], ...],
        extreme_distributions: tuple[ExtremeStationaryDistribution, ...],
        unique: bool,
    ) -> Self:
        """Construct a family whose stationary equations the kernel solved."""

        return cls.model_construct(
            transition_matrix=transition_matrix,
            extreme_distributions=extreme_distributions,
            unique=unique,
        )

    @model_validator(mode="after")
    def bind_stationary_family(self) -> Self:
        TransitionMatrixRequest(matrix=self.transition_matrix)
        dimension = len(self.transition_matrix)
        dimensions = {len(item.distribution) for item in self.extreme_distributions}
        if dimensions != {dimension}:
            raise _validation_error(
                "stationary_dimension_mismatch",
                "stationary distributions must share the source matrix dimension",
            )
        classes = tuple(item.closed_class for item in self.extreme_distributions)
        if classes != tuple(sorted(classes)) or len(classes) != len(set(classes)):
            raise _validation_error(
                "closed_classes_not_canonical",
                "closed classes must be unique and sorted",
            )
        if self.unique != (len(self.extreme_distributions) == 1):
            raise _validation_error(
                "stationary_unique_mismatch",
                "unique must match the number of extreme distributions",
            )
        for item in self.extreme_distributions:
            values = tuple(value.as_fraction() for value in item.distribution)
            if any(value < 0 for value in values) or sum(values) != 1:
                raise _validation_error(
                    "stationary_distribution_invalid",
                    "each stationary distribution must be nonnegative and normalized",
                )
            support = tuple(index for index, value in enumerate(values) if value > 0)
            if support != item.closed_class:
                raise _validation_error(
                    "stationary_support_mismatch",
                    "each extreme distribution must be supported on its closed class",
                )
        return self


class ErgodicDecisionResult(StrictModel):
    is_ergodic: bool
    is_irreducible: bool
    is_aperiodic: bool


class MixingTimeRequest(TransitionMatrixRequest):
    """A bounded exact search for worst-case total-variation mixing time."""

    epsilon: CanonicalRational
    max_steps: StrictInt = Field(
        default=DEFAULT_MIXING_STEPS, ge=1, le=MAX_MIXING_STEPS
    )


class MixingTimeResult(StrictModel):
    status: Literal["FOUND", "NOT_ERGODIC", "BOUND_EXCEEDED"]
    epsilon: CanonicalRational
    max_steps: StrictInt = Field(ge=1, le=MAX_MIXING_STEPS)
    steps_examined: StrictInt = Field(ge=0, le=MAX_MIXING_STEPS + 1)
    mixing_time: StrictInt | None = Field(default=None, ge=0, le=MAX_MIXING_STEPS)
    max_total_variation_distance: CanonicalRational | None = None
    method: Literal["SYMPY_EXACT_MATRIX_POWERS"] = "SYMPY_EXACT_MATRIX_POWERS"

    @model_validator(mode="after")
    def bind_search_result(self) -> Self:
        distance = (
            None
            if self.max_total_variation_distance is None
            else self.max_total_variation_distance.as_fraction()
        )
        if distance is not None and not 0 <= distance <= 1:
            raise _validation_error(
                "mixing_distance_out_of_range",
                "total-variation distance must lie in [0, 1]",
            )
        if self.status == "FOUND":
            if (
                self.mixing_time is None
                or distance is None
                or distance > self.epsilon.as_fraction()
                or self.steps_examined != self.mixing_time + 1
            ):
                raise _validation_error(
                    "mixing_found_payload_invalid",
                    "a found result requires its first satisfactory step and distance",
                )
        elif self.status == "BOUND_EXCEEDED":
            if (
                self.mixing_time is not None
                or distance is None
                or distance <= self.epsilon.as_fraction()
                or self.steps_examined != self.max_steps + 1
            ):
                raise _validation_error(
                    "mixing_bound_exceeded_payload_invalid",
                    "a bound-exceeded result requires the terminal unsatisfactory distance",
                )
        elif (
            self.mixing_time is not None
            or distance is not None
            or self.steps_examined != 0
        ):
            raise _validation_error(
                "mixing_nonergodic_payload_invalid",
                "a non-ergodic result has no mixing-time search value",
            )
        return self


class CommunicatingClassesResult(StrictModel):
    """The communicating-class decomposition of a Markov chain."""

    transition_matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=MAX_MARKOV_STATES
    )
    """The source transition matrix whose support graph is decomposed."""

    classes: tuple[tuple[tuple[int, ...], bool], ...]
    """Each entry is (state_indices, is_closed). States are 0-indexed."""

    state_class: tuple[int, ...]
    """Class index of each state (0-indexed)."""

    @classmethod
    def _from_kernel(
        cls,
        *,
        transition_matrix: tuple[tuple[CanonicalRational, ...], ...],
        classes: tuple[tuple[tuple[int, ...], bool], ...],
        state_class: tuple[int, ...],
    ) -> Self:
        """Construct output whose SCC relation was established by the kernel."""

        return cls.model_construct(
            transition_matrix=transition_matrix,
            classes=classes,
            state_class=state_class,
        )

    @model_validator(mode="after")
    def require_partition_validity(self) -> Self:
        TransitionMatrixRequest(matrix=self.transition_matrix)
        dimension = len(self.transition_matrix)
        if any(len(row) != dimension for row in self.transition_matrix):
            raise _validation_error(
                "decomposition_matrix_not_square", "transition matrix must be square"
            )
        all_states: list[int] = []
        for states, _ in self.classes:
            all_states.extend(states)
        if sorted(all_states) != list(range(len(all_states))):
            raise _validation_error(
                "decomposition_classes_not_partition",
                "classes must partition all state indices",
            )
        if len(all_states) != dimension:
            raise _validation_error(
                "decomposition_class_dimension_mismatch",
                "classes must partition the transition matrix states",
            )
        if len(self.state_class) != len(all_states):
            raise _validation_error(
                "decomposition_state_class_length",
                "state_class must have one entry per state",
            )
        for class_index, (states, _) in enumerate(self.classes):
            for state in states:
                if self.state_class[state] != class_index:
                    raise _validation_error(
                        "decomposition_state_class_mismatch",
                        "state_class must match classes",
                    )
        return self
