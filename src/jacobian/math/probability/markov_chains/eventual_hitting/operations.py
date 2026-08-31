"""Eventual hitting probability kernel."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.markov_chains.eventual_hitting._models import (
    EventualHittingProfileResult,
)
from jacobian.math.probability.markov_chains.values import (
    TransitionMatrix,
    TransitionMatrixAdmissionError,
    _decimal_digits,
    as_canonical_transition_matrix,
    require_transition_matrix,
)

__all__ = ["compute_eventual_hitting_profile"]


def _json_array_size(item_size: int, count: int) -> int:
    return 2 + max(count - 1, 0) + item_size * count


def _json_array_sizes(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _reject(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location, code=f"markov_chain.eventual_hitting.{code}", message=message
    )


def _admit_eventual_hitting(  # noqa: C901
    matrix: TransitionMatrix,
    target_states: tuple[int, ...],
    *,
    enforce_transport_limit: bool,
) -> tuple[set[int], set[int], tuple[int, ...]]:
    try:
        require_transition_matrix(matrix)
    except TransitionMatrixAdmissionError as exc:
        _reject(exc.location, exc.reason, str(exc))

    n = len(matrix)
    if type(target_states) is not tuple or not target_states:
        _reject(
            ("target_states",),
            "targets_must_be_nonempty",
            "target_states must be nonempty",
        )
    if any(type(state) is not int for state in target_states):
        _reject(
            ("target_states",),
            "targets_must_be_integers",
            "target states must be integers",
        )
    if tuple(sorted(set(target_states))) != target_states:
        _reject(
            ("target_states",),
            "targets_must_be_strictly_increasing",
            "target_states must be strictly increasing",
        )
    if any(state < 0 or state >= n for state in target_states):
        _reject(
            ("target_states",),
            "target_out_of_range",
            "target states must be in 0..n-1",
        )

    target_set = set(target_states)
    reverse: dict[int, set[int]] = {state: set() for state in range(n)}
    for source, row in enumerate(matrix):
        for destination, probability in enumerate(row):
            if probability > 0:
                reverse[destination].add(source)
    can_reach_target = set(target_set)
    frontier = list(target_set)
    while frontier:
        current = frontier.pop()
        for predecessor in reverse[current]:
            if predecessor not in can_reach_target:
                can_reach_target.add(predecessor)
                frontier.append(predecessor)

    transient = tuple(
        state
        for state in range(n)
        if state in can_reach_target and state not in target_set
    )
    matrix_digits = max(
        (
            max(_decimal_digits(value.numerator), _decimal_digits(value.denominator))
            for row in matrix
            for value in row
        ),
        default=1,
    )
    if matrix_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            ("matrix",),
            "source_height_exceeds_bound",
            "transition probabilities exceed the exact rational source bound",
        )
    transient_count = len(transient)
    # Charge common-denominator growth row by row; combining the largest row
    # width with the largest row term count can reject profiles where those
    # maxima occur in different rows.
    row_growth = 0
    for state in transient:
        row_values = [
            value
            for destination, value in enumerate(matrix[state])
            if destination in can_reach_target and value != 0
        ]
        row_terms = len(row_values)
        row_digits = max(
            (
                max(
                    _decimal_digits(value.numerator), _decimal_digits(value.denominator)
                )
                for value in row_values
            ),
            default=1,
        )
        row_growth += row_terms * row_digits + _decimal_digits(max(row_terms, 1))
    result_height = (
        transient_count * row_growth + _decimal_digits(factorial(transient_count)) + 1
    )
    if result_height > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            ("matrix",),
            "result_height_exceeds_bound",
            "eventual hitting probabilities exceed the exact rational result bound",
        )

    matrix_row_bytes = [
        _json_array_sizes(
            [
                len(
                    encode_strict_json(
                        CanonicalRational.from_fraction(value).model_dump(mode="json")
                    )
                )
                for value in row
            ]
        )
        for row in matrix
    ]
    matrix_bytes = _json_array_sizes(matrix_row_bytes)
    index_bytes = max(1, len(str(max(n - 1, 0))))
    target_bytes = _json_array_size(index_bytes, len(target_states))
    transient_probability_bytes = strict_json_object_size(
        (("num", result_height + 2), ("den", result_height + 2))
    )
    zero_probability_bytes = len(
        encode_strict_json(
            CanonicalRational.from_fraction(Fraction(0)).model_dump(mode="json")
        )
    )
    one_probability_bytes = len(
        encode_strict_json(
            CanonicalRational.from_fraction(Fraction(1)).model_dump(mode="json")
        )
    )
    probability_bytes = _json_array_sizes(
        [
            transient_probability_bytes
            if state in transient
            else one_probability_bytes
            if state in target_set
            else zero_probability_bytes
            for state in range(n)
        ]
    )
    classification_bytes = _json_array_size(index_bytes, n)
    result_bytes = strict_json_object_size(
        (
            ("matrix", matrix_bytes),
            ("target_states", target_bytes),
            ("hitting_probabilities", probability_bytes),
            ("zero_states", classification_bytes),
            ("proper_states", classification_bytes),
            ("almost_sure_states", classification_bytes),
        )
    )
    if enforce_transport_limit and result_bytes > CanonicalLimits().max_output_bytes:
        _reject(
            ("matrix",),
            "result_exceeds_output_bound",
            "eventual hitting profile exceeds the canonical output bound",
        )
    return target_set, can_reach_target, transient


def compute_eventual_hitting_profile(
    matrix: TransitionMatrix,
    target_states: tuple[int, ...],
    *,
    enforce_transport_limit: bool = False,
) -> EventualHittingProfileResult:
    """Return the eventual hitting probability profile for a Markov chain.

    For each state i, compute h(i) = P_i(ever hit the target set A).
    """
    target_set, can_reach_target, transient = _admit_eventual_hitting(
        matrix, target_states, enforce_transport_limit=enforce_transport_limit
    )
    n = len(matrix)
    h = [Fraction(0)] * n
    for i in target_set:
        h[i] = Fraction(1)

    non_target = list(transient)

    if not non_target:
        source_matrix = as_canonical_transition_matrix(matrix)
        almost_sure = tuple(i for i in range(n) if i in target_set)
        return EventualHittingProfileResult(
            matrix=source_matrix,
            target_states=target_states,
            hitting_probabilities=tuple(
                CanonicalRational.from_fraction(h[i]) for i in range(n)
            ),
            zero_states=tuple(i for i in range(n) if i not in can_reach_target),
            proper_states=(),
            almost_sure_states=almost_sure,
        )

    a_matrix = [[Fraction(0)] * len(non_target) for _ in range(len(non_target))]
    b_vector = [Fraction(0)] * len(non_target)
    for row_idx, i in enumerate(non_target):
        a_matrix[row_idx][row_idx] = Fraction(1)
        for col_idx, j in enumerate(non_target):
            a_matrix[row_idx][col_idx] -= matrix[i][j]
        for j in target_set:
            b_vector[row_idx] += matrix[i][j]

    from jacobian.math.probability.markov_chains._flint import solve_linear_system

    solution = solve_linear_system(a_matrix, b_vector)
    if solution is None:
        _reject(
            ("matrix",),
            "transient_system_singular",
            "the target-reachable transient system must be nonsingular",
        )
    for idx, i in enumerate(non_target):
        h[i] = solution[idx]

    zero_states = tuple(i for i in range(n) if i not in can_reach_target)
    proper_states = tuple(i for i in range(n) if 0 < h[i] < 1)
    almost_sure = tuple(i for i in range(n) if h[i] == 1)

    return EventualHittingProfileResult(
        matrix=as_canonical_transition_matrix(matrix),
        target_states=target_states,
        hitting_probabilities=tuple(
            CanonicalRational.from_fraction(h[i]) for i in range(n)
        ),
        zero_states=zero_states,
        proper_states=proper_states,
        almost_sure_states=almost_sure,
    )

