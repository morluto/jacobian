"""Admission quantities shared by symbolic-dynamics wire contracts and kernels."""

from __future__ import annotations

from jacobian.math.symbolic_dynamics.values import (
    MAX_ENUMERATED_BLOCKS,
    MAX_PERIOD,
    MAX_PRESENTATION_CELLS,
    AdjacencyShift,
    ForbiddenBlockShift,
)

MAX_ZETA_REPLAY_PERIOD = MAX_PERIOD
MAX_ZETA_COEFFICIENT_DIGITS = 128
MAX_ZETA_RESULT_DIGITS = 100_000
MAX_ZETA_WORK = 13_000_000


def enumeration_size(alphabet_size: int, block_length: int) -> int:
    """Bound one full alphabet-word enumeration before materializing it."""

    if block_length < 0:
        raise ValueError("block length must be nonnegative")
    size = 1
    for _ in range(block_length):
        size *= alphabet_size
        if size > MAX_ENUMERATED_BLOCKS:
            raise ValueError("requested block enumeration exceeds the work bound")
    return size


def _contains(word: tuple[str, ...], factor: tuple[str, ...]) -> bool:
    return any(
        word[start : start + len(factor)] == factor
        for start in range(len(word) - len(factor) + 1)
    )


def normalize_forbidden_blocks(
    shift: ForbiddenBlockShift,
) -> tuple[tuple[str, ...], ...]:
    """Return the canonical antichain of forbidden factors."""

    rank = {symbol: index for index, symbol in enumerate(shift.alphabet)}
    ordered = sorted(
        set(shift.forbidden_blocks),
        key=lambda block: (len(block), tuple(rank[symbol] for symbol in block)),
    )
    minimal: list[tuple[str, ...]] = []
    for block in ordered:
        if not any(_contains(block, forbidden) for forbidden in minimal):
            minimal.append(block)
    return tuple(minimal)


def presentation_memory(shift: ForbiddenBlockShift) -> int:
    return max(
        0,
        max(
            (len(block) - 1 for block in normalize_forbidden_blocks(shift)),
            default=0,
        ),
    )


def require_bounded_support(shift: ForbiddenBlockShift) -> int:
    memory = presentation_memory(shift)
    enumeration_size(len(shift.alphabet), memory + 1)
    return memory


def require_bounded_presentation(shift: ForbiddenBlockShift, memory: int) -> None:
    state_count = enumeration_size(len(shift.alphabet), memory)
    enumeration_size(len(shift.alphabet), memory + 1)
    if state_count * state_count > MAX_PRESENTATION_CELLS:
        raise ValueError("presentation adjacency exceeds the result bound")


def require_zeta_budget(shift: AdjacencyShift, replay_period: int) -> None:
    """Admit determinant, trace replay, and their derivable exact result size."""

    if not 1 <= replay_period <= MAX_ZETA_REPLAY_PERIOD:
        raise ValueError("replay period is outside the supported bounds")
    states = len(shift.matrix)
    maximum_entry = max(entry for row in shift.matrix for entry in row)
    # The degree-k coefficient is a sum of principal k-minors. Its absolute
    # value is at most C(states,k) k! M^k <= (states M)^k.
    coefficient_digits = states * len(str(states * max(1, maximum_entry))) + 1
    if coefficient_digits > MAX_ZETA_COEFFICIENT_DIGITS:
        raise ValueError("zeta polynomial exceeds the coefficient digit bound")
    work = states**4 + states**3 * replay_period
    if work > MAX_ZETA_WORK:
        raise ValueError("zeta determinant and replay exceed the work bound")
    maximum_row_sum = max(sum(row) for row in shift.matrix)
    count_bound = states * max(1, maximum_row_sum) ** replay_period
    count_digits = len(str(count_bound))
    result_digits = (
        2 * (states + 1) * coefficient_digits + 2 * replay_period * count_digits
    )
    if result_digits > MAX_ZETA_RESULT_DIGITS:
        raise ValueError("zeta result exceeds the aggregate digit bound")


__all__ = [
    "MAX_ZETA_REPLAY_PERIOD",
    "enumeration_size",
    "normalize_forbidden_blocks",
    "presentation_memory",
    "require_bounded_presentation",
    "require_bounded_support",
    "require_zeta_budget",
]
