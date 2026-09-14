"""Exact finite-state decision for progressions in Kempner digit sets."""

from __future__ import annotations

from collections import deque
from math import ceil, log10
from typing import NamedTuple

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import (
    MAX_KEMPNER_ARITY,
    MAX_KEMPNER_BASE,
    MAX_KEMPNER_INTEGER_DIGITS,
    KempnerArithmeticProgressionRequest,
    KempnerArithmeticProgressionResult,
    KempnerContainsProgression,
    KempnerDigitSet,
    KempnerProgressionFree,
)

MAX_CARRY_GRAPH_STATES = 1_000_000
MAX_CARRY_GRAPH_WORK = 5_000_000
MAX_CARRY_PREDECESSOR_ALLOCATION = 128_000_000
MAX_CARRY_RESULT_ALLOCATION = 2_000_000


class _State(NamedTuple):
    carries: tuple[int, ...]
    seen_terms: int
    pending_zero_terms: int
    difference_seen: bool


class _Admission(NamedTuple):
    state_bound: int
    transition_work: int
    predecessor_allocation: int


def _state_bound(arity: int) -> int:
    """Return a conservative finite-state count without unbounded growth."""

    if arity > 12:
        return MAX_CARRY_GRAPH_STATES + 1
    bound = 2
    for _ in range(arity):
        bound *= 4
        if bound > MAX_CARRY_GRAPH_STATES:
            return bound
    for carry_index in range(1, arity):
        bound *= carry_index + 1
        if bound > MAX_CARRY_GRAPH_STATES:
            return bound
    return bound


def _require_canonical_digit_set(digit_set: KempnerDigitSet) -> None:
    """Reject a non-canonical or out-of-domain digit set at the boundary."""

    if (
        not isinstance(digit_set.base, int)
        or isinstance(digit_set.base, bool)
        or not 2 <= digit_set.base <= MAX_KEMPNER_BASE
        or not isinstance(digit_set.allowed_digits, tuple)
        or not digit_set.allowed_digits
        or any(
            not isinstance(digit, int)
            or isinstance(digit, bool)
            or digit < 0
            or digit >= digit_set.base
            for digit in digit_set.allowed_digits
        )
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_progression.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )
    if (
        digit_set.allowed_digits != tuple(sorted(set(digit_set.allowed_digits)))
        or len(digit_set.allowed_digits) >= digit_set.base
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_progression.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )


def _digit_level_progression(
    allowed: tuple[int, ...], arity: int
) -> tuple[int, ...] | None:
    """Return a one-digit arithmetic progression of the requested arity, if any.

    Every value is an allowed digit, so each is a positive member of the
    family; detecting one avoids charging the carry graph for a trivial result.
    """

    allowed_set = set(allowed)
    for first in allowed:
        if first < 1:
            continue
        for second in allowed:
            if second <= first:
                continue
            difference = second - first
            values = tuple(first + index * difference for index in range(arity))
            if all(value in allowed_set for value in values):
                return values
    return None


def _require_admission(digit_set: KempnerDigitSet, arity: int) -> _Admission:
    _require_canonical_digit_set(digit_set)
    state_bound = _state_bound(arity)
    base = digit_set.base
    transition_work = state_bound * base * base
    # A predecessor stores a state, two source digits, and a pointer.  This is
    # deliberately charged before BFS; the kernel never grows past the charge.
    predecessor_allocation = state_bound * (8 * (arity + 4) + 24)
    if state_bound > MAX_CARRY_GRAPH_STATES:
        raise OperationResourceAdmissionError(
            location=("arity",),
            code="number_theory.kempner_progression.state_graph",
            message=(
                "the derived carry-state graph exceeds the admitted finite-state "
                "envelope"
            ),
        )
    if transition_work > MAX_CARRY_GRAPH_WORK:
        raise OperationResourceAdmissionError(
            location=("digit_set", "base"),
            code="number_theory.kempner_progression.transition_work",
            message="the derived digit transition work exceeds the admitted budget",
        )
    if predecessor_allocation > MAX_CARRY_PREDECESSOR_ALLOCATION:
        raise OperationResourceAdmissionError(
            location=("arity",),
            code="number_theory.kempner_progression.predecessors",
            message="the derived predecessor storage exceeds the admitted budget",
        )
    return _Admission(
        state_bound=state_bound,
        transition_work=transition_work,
        predecessor_allocation=predecessor_allocation,
    )


def _next_state(
    state: _State,
    *,
    x: int,
    y: int,
    base: int,
    allowed: frozenset[int],
    arity: int,
) -> _State | None:
    carries = list(state.carries)
    seen_terms = state.seen_terms
    pending = state.pending_zero_terms
    for index in range(arity):
        if index == 0:
            digit = x
        else:
            total = x + index * y + carries[index - 1]
            digit = total % base
            carries[index - 1] = total // base
        bit = 1 << index
        if digit == 0:
            if 0 not in allowed:
                pending |= bit
            continue
        if digit not in allowed or pending & bit:
            return None
        seen_terms |= bit
    return _State(tuple(carries), seen_terms, pending, state.difference_seen or y != 0)


def _accepting(state: _State, *, arity: int) -> bool:
    return (
        not any(state.carries)
        and state.difference_seen
        and state.seen_terms == (1 << arity) - 1
    )


def _reconstruct(
    terminal: _State,
    predecessors: dict[_State, tuple[_State | None, int, int]],
    *,
    base: int,
) -> tuple[int, int, int]:
    digits: list[tuple[int, int]] = []
    state = terminal
    while True:
        previous, x, y = predecessors[state]
        if previous is None:
            break
        digits.append((x, y))
        state = previous
    digits.reverse()
    first = sum(x * base**position for position, (x, _) in enumerate(digits))
    difference = sum(y * base**position for position, (_, y) in enumerate(digits))
    # The source-reachable predecessor path fixes the actual witness width: the
    # BFS accepts at the shallowest depth, so its digit count bounds the output
    # far below the worst-case state count.
    return first, difference, len(digits)


def decide_kempner_arithmetic_progression(
    digit_set: KempnerDigitSet,
    arity: int,
) -> KempnerArithmeticProgressionResult:
    """Decide exactly whether a Kempner family contains a nontrivial AP."""

    if not isinstance(arity, int) or isinstance(arity, bool) or arity < 3:
        raise OperationDomainValidationError(
            location=("arity",),
            code="number_theory.kempner_progression.arity",
            message="arity must be an integer of at least three",
        )
    if arity > MAX_KEMPNER_ARITY:
        raise OperationResourceAdmissionError(
            location=("arity",),
            code="number_theory.kempner_progression.arity_bound",
            message=f"arity must be at most {MAX_KEMPNER_ARITY}",
        )
    _require_canonical_digit_set(digit_set)
    if tuple(digit_set.allowed_digits) == (0,):
        # The only digit is 0, so there is no positive member and every arity
        # is trivially progression-free; do not charge the carry-graph envelope.
        return KempnerArithmeticProgressionResult(
            digit_set=digit_set,
            arity=arity,
            conclusion=KempnerProgressionFree(status="PROGRESSION_FREE"),
        )
    if len(digit_set.allowed_digits) == 1:
        # A singleton nonzero digit admits only repdigits
        # d*(base**k-1)/(base-1), which contain no three-term (hence no longer)
        # nontrivial progression: 2*base**j = base**i + base**k forces i=j=k.
        # Every arity is trivially progression-free; do not charge the
        # carry-graph envelope, whose source-blind bound would reject sparse
        # cases such as base 64 with a single allowed digit.
        return KempnerArithmeticProgressionResult(
            digit_set=digit_set,
            arity=arity,
            conclusion=KempnerProgressionFree(status="PROGRESSION_FREE"),
        )
    digit_progression = _digit_level_progression(digit_set.allowed_digits, arity)
    if digit_progression is not None:
        first = digit_progression[0]
        difference = digit_progression[1] - digit_progression[0]
        return KempnerArithmeticProgressionResult(
            digit_set=digit_set,
            arity=arity,
            conclusion=KempnerContainsProgression(
                status="CONTAINS_PROGRESSION",
                indices=tuple(range(arity)),
                values=digit_progression,
                first_term=first,
                common_difference=difference,
            ),
        )
    admission = _require_admission(digit_set, arity)
    base = digit_set.base
    allowed = frozenset(digit_set.allowed_digits)
    start = _State((0,) * (arity - 1), 0, 0, False)
    queue: deque[_State] = deque((start,))
    predecessors: dict[_State, tuple[_State | None, int, int]] = {start: (None, 0, 0)}
    terminal: _State | None = None
    while queue:
        request_checkpoint("during Kempner progression BFS")
        state = queue.popleft()
        if _accepting(state, arity=arity):
            terminal = state
            break
        for x in range(base):
            for y in range(base):
                candidate = _next_state(
                    state,
                    x=x,
                    y=y,
                    base=base,
                    allowed=allowed,
                    arity=arity,
                )
                if candidate is None or candidate in predecessors:
                    continue
                predecessors[candidate] = (state, x, y)
                if len(predecessors) > admission.state_bound:
                    raise RuntimeError("Kempner BFS exceeded its admitted state bound")
                queue.append(candidate)
    if terminal is None:
        return KempnerArithmeticProgressionResult(
            digit_set=digit_set,
            arity=arity,
            conclusion=KempnerProgressionFree(status="PROGRESSION_FREE"),
        )
    first, difference, witness_depth = _reconstruct(terminal, predecessors, base=base)
    witness_digits = ceil(witness_depth * log10(base)) + 1
    result_allocation = arity * (witness_digits + 24) + 256
    if (
        witness_digits > MAX_KEMPNER_INTEGER_DIGITS
        or result_allocation > MAX_CARRY_RESULT_ALLOCATION
    ):
        raise OperationResourceAdmissionError(
            location=("arity",),
            code="number_theory.kempner_progression.result_size",
            message="the derived exact witness output exceeds the admitted budget",
        )
    values = tuple(first + index * difference for index in range(arity))
    if first < 1 or difference < 1 or values[-1] != first + (arity - 1) * difference:
        raise RuntimeError("Kempner BFS produced an invalid progression witness")
    return KempnerArithmeticProgressionResult(
        digit_set=digit_set,
        arity=arity,
        conclusion=KempnerContainsProgression(
            status="CONTAINS_PROGRESSION",
            indices=tuple(range(arity)),
            values=values,
            first_term=first,
            common_difference=difference,
        ),
    )


def compute_kempner_arithmetic_progression(
    request: KempnerArithmeticProgressionRequest,
) -> KempnerArithmeticProgressionResult:
    return decide_kempner_arithmetic_progression(request.digit_set, request.arity)


KEMPNER_ARITHMETIC_PROGRESSION_OPERATION = MathTool(
    operation_id="number_theory.kempner_set.arithmetic_progression.decide",
    title="Decide fixed-arity arithmetic progressions in a Kempner digit set",
    description=(
        "Decide exactly whether positive integers whose canonical base-b digits "
        "lie in one proper digit subset contain a nontrivial fixed-arity "
        "arithmetic progression; return the shortest-padded, lexicographically "
        "first witness when one exists."
    ),
    request_type=KempnerArithmeticProgressionRequest,
    result_type=KempnerArithmeticProgressionResult,
    run=compute_kempner_arithmetic_progression,
    tags=("number-theory", "digit-restriction", "arithmetic-progression", "exact"),
    discovery_terms=(
        "Kempner set arithmetic progression",
        "restricted digits progression",
    ),
    examples=(
        OperationExample(
            name="binary_three_term_progression",
            description=(
                "Decide whether the positive ternary digit family {1,2} contains "
                "a 3-term progression; the digit subset must be a proper "
                "canonical subset of its base alphabet."
            ),
            input={
                "digit_set": {"base": "3", "allowed_digits": ["1", "2"]},
                "arity": "3",
            },
        ),
    ),
)


__all__ = [
    "KEMPNER_ARITHMETIC_PROGRESSION_OPERATION",
    "compute_kempner_arithmetic_progression",
    "decide_kempner_arithmetic_progression",
]
