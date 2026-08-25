"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from math import factorial
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Markov-chain contracts."""

    return PydanticCustomError(f"markov_chain.{reason}", message)


class TransitionMatrixRequest(StrictModel):
    """A finite stochastic transition matrix with rational entries."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=32
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

    @model_validator(mode="after")
    def require_bounded_stationary_height(self) -> Self:
        dimension = len(self.matrix)
        cleared_row_bounds: list[int] = []
        for column in range(dimension - 1):
            entries = tuple(self.matrix[row][column] for row in range(dimension))
            denominator_digits = sum(len(value.den) for value in entries)
            cleared_row_bounds.append(
                max(
                    max(len(value.num.lstrip("-")), len(value.den))
                    + 1
                    + denominator_digits
                    - len(value.den)
                    for value in entries
                )
            )
        cleared_row_bounds.append(1)  # normalization: sum(pi_i) = 1

        # Leibniz bounds both det(A) and every Cramer numerator: each term is
        # a product with one cleared integer from every row, and there are at
        # most dimension! terms. Reduction can only decrease coordinate height.
        determinant_digits = sum(cleared_row_bounds) + len(str(factorial(dimension)))
        if determinant_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _validation_error(
                "stationary_height_exceeds_bound",
                "stationary distribution rational height exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound",
            )
        return self


class ExtremeStationaryDistribution(StrictModel):
    """One canonical extreme point supported on a closed class."""

    closed_class: tuple[int, ...] = Field(min_length=1)
    distribution: tuple[CanonicalRational, ...] = Field(min_length=1)


class StationaryDistributionResult(StrictModel):
    """Extreme points of the finite chain's stationary-distribution simplex."""

    extreme_distributions: tuple[ExtremeStationaryDistribution, ...] = Field(
        min_length=1
    )
    unique: bool
    method: Literal["CLOSED_CLASS_EXACT_LINEAR_SYSTEM"] = (
        "CLOSED_CLASS_EXACT_LINEAR_SYSTEM"
    )

    @model_validator(mode="after")
    def bind_stationary_family(self) -> Self:
        dimensions = {len(item.distribution) for item in self.extreme_distributions}
        if len(dimensions) != 1:
            raise _validation_error(
                "stationary_dimension_mismatch",
                "stationary distributions must share one dimension",
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
    max_steps: StrictInt = Field(default=64, ge=1, le=256)

    @model_validator(mode="after")
    def require_bounded_search(self) -> Self:
        if len(self.matrix) > 32:
            raise _validation_error(
                "mixing_state_limit", "mixing-time search supports at most 32 states"
            )
        if not 0 < self.epsilon.as_fraction() <= 1:
            raise _validation_error(
                "mixing_epsilon_out_of_range", "epsilon must lie in (0, 1]"
            )
        for value in (self.epsilon, *(item for row in self.matrix for item in row)):
            if max(len(value.num.lstrip("-")), len(value.den)) > 32:
                raise _validation_error(
                    "mixing_component_digits_exceed_limit",
                    "mixing-time rational components support at most 32 digits",
                )
        matrix_digits = max(
            max(len(value.num.lstrip("-")), len(value.den))
            for row in self.matrix
            for value in row
        )
        state_count = len(self.matrix)
        # A common denominator for the transition matrix has at most n**2 * d
        # digits. Exact powers contribute one such denominator per step, while
        # Cramer's rule bounds the stationary denominator by n times that size.
        rational_height_digits = matrix_digits * (
            state_count**3 + self.max_steps * state_count**2
        )
        if rational_height_digits > MAX_CANONICAL_RATIONAL_DIGITS - 1_024:
            raise _validation_error(
                "mixing_result_height_exceeds_bound",
                "mixing-time matrix height and max_steps can exceed the exact "
                "rational result bound",
            )
        return self


class MixingTimeResult(StrictModel):
    status: Literal["FOUND", "NOT_ERGODIC", "BOUND_EXCEEDED"]
    epsilon: CanonicalRational
    max_steps: StrictInt = Field(ge=1, le=256)
    steps_examined: StrictInt = Field(ge=0, le=257)
    mixing_time: StrictInt | None = Field(default=None, ge=0, le=256)
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
        min_length=1, max_length=32
    )
    """The source transition matrix whose support graph is decomposed."""

    classes: tuple[tuple[tuple[int, ...], bool], ...]
    """Each entry is (state_indices, is_closed). States are 0-indexed."""

    state_class: tuple[int, ...]
    """Class index of each state (0-indexed)."""

    @model_validator(mode="after")
    def require_partition_validity(self) -> Self:
        dimension = len(self.transition_matrix)
        if any(len(row) != dimension for row in self.transition_matrix):
            raise _validation_error(
                "decomposition_matrix_not_square", "transition matrix must be square"
            )
        for row in self.transition_matrix:
            values = tuple(value.as_fraction() for value in row)
            if any(value < 0 for value in values):
                raise _validation_error(
                    "decomposition_probability_negative",
                    "transition probabilities must be nonnegative",
                )
            if sum(values) != 1:
                raise _validation_error(
                    "decomposition_row_not_stochastic",
                    "each transition row must sum to one",
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

    @model_validator(mode="after")
    def bind_decomposition(self) -> Self:
        import networkx as nx

        matrix = self.transition_matrix
        dimension = len(matrix)
        graph: nx.DiGraph[int] = nx.DiGraph()
        graph.add_nodes_from(range(dimension))
        for i in range(dimension):
            for j in range(dimension):
                if matrix[i][j].as_fraction() > 0:
                    graph.add_edge(i, j)
        sccs = list(nx.strongly_connected_components(graph))
        condensation = nx.condensation(graph, sccs)
        scc_list = list(nx.topological_sort(condensation))
        expected_classes: list[tuple[tuple[int, ...], bool]] = []
        expected_state_class = [0] * dimension
        for scc_idx, scc_node in enumerate(scc_list):
            scc = sccs[scc_node] if isinstance(scc_node, int) else scc_node
            states = tuple(sorted(scc))
            is_closed = True
            for state in states:
                for target in range(dimension):
                    if target not in scc and matrix[state][target].as_fraction() > 0:
                        is_closed = False
                        break
                if not is_closed:
                    break
            expected_classes.append((states, is_closed))
            for state in states:
                expected_state_class[state] = scc_idx
        if self.classes != tuple(expected_classes):
            raise _validation_error(
                "decomposition_scc_classes_mismatch",
                "classes must be the exact SCC decomposition of the transition matrix",
            )
        if self.state_class != tuple(expected_state_class):
            raise _validation_error(
                "decomposition_scc_state_class_mismatch",
                "state_class must match the SCC decomposition",
            )
        return self
