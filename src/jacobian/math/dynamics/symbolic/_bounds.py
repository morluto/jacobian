"""Request-scoped bounds used by symbolic-dynamics native operations."""

from __future__ import annotations

from jacobian.math.dynamics.symbolic.values import (
    MAX_ADJACENCY_STATES,
    MAX_ALPHABET_SIZE,
    MAX_ENUMERATED_BLOCKS,
    MAX_FORBIDDEN_BLOCK_LENGTH,
    MAX_FORBIDDEN_BLOCKS,
    MAX_PRESENTATION_CELLS,
    MAX_PRESENTATION_TRANSITIONS,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
    LabeledTransition,
)

MAX_ZETA_COEFFICIENT_DIGITS = 128
MAX_ZETA_RESULT_DIGITS = 100_000
MAX_ZETA_WORK = 13_000_000
MAX_PERIODIC_PROFILE_DIGITS = 100_000
MAX_PERIODIC_PROFILE_WORK = 10_000_000
MAX_PRESENTATION_VERIFICATION_WORK = 2_500_000


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


def _presentation_carrier_work(
    presentation: BlockPresentation,
) -> tuple[int, list[int]]:
    """Check nested carrier axes and count their bounded scan cost."""

    carrier_work = len(presentation.alphabet)
    for block in presentation.state_blocks:
        if type(block) is not tuple or len(block) > MAX_FORBIDDEN_BLOCK_LENGTH:
            raise ValueError("presentation state-block axis is malformed")
        carrier_work += 1 + len(block)
    forbidden_lengths: list[int] = []
    for block in presentation.forbidden_blocks:
        if type(block) is not tuple or len(block) > MAX_FORBIDDEN_BLOCK_LENGTH:
            raise ValueError("presentation forbidden-block axis is malformed")
        length = len(block)
        forbidden_lengths.append(length)
        carrier_work += 1 + length
    for row in presentation.adjacency_matrix:
        if type(row) is not tuple or len(row) != len(presentation.state_blocks):
            raise ValueError("presentation adjacency axis is malformed")
        carrier_work += 1 + len(row)
    for transition in presentation.transitions:
        if type(transition) is not LabeledTransition:
            raise ValueError("presentation transition axis is malformed")
        carrier_work += 3
    return carrier_work, forbidden_lengths


def _contains_work(word_length: int, factor_length: int) -> int:
    return max(0, word_length - factor_length + 1) * factor_length


def require_bounded_presentation_verification(
    presentation: BlockPresentation,
) -> None:
    """Admit one complete serialized presentation check before scanning it."""

    state_count = len(presentation.state_blocks)
    if state_count > MAX_ADJACENCY_STATES:
        raise ValueError("presentation state axis exceeds the supported bound")
    if len(presentation.transitions) > MAX_PRESENTATION_TRANSITIONS:
        raise ValueError("presentation transition axis exceeds the supported bound")
    if len(presentation.forbidden_blocks) > MAX_FORBIDDEN_BLOCKS:
        raise ValueError(
            "presentation forbidden-block axis exceeds the supported bound"
        )
    if presentation.memory > MAX_FORBIDDEN_BLOCK_LENGTH:
        raise ValueError("presentation memory exceeds the supported bound")
    if len(presentation.alphabet) > MAX_ALPHABET_SIZE:
        raise ValueError("presentation alphabet exceeds the supported bound")
    if len(presentation.adjacency_matrix) != state_count:
        raise ValueError("presentation adjacency axis exceeds the supported bound")

    # The verifier reconstructs the complete occurring state axis and scans
    # every source rule.  Check the nested carrier axes before those scans so
    # hostile tuple subclasses cannot turn a bounded claim into unbounded
    # iteration, and charge these scans to the same admission budget.
    carrier_work, forbidden_lengths = _presentation_carrier_work(presentation)

    required_memory = max((length - 1 for length in forbidden_lengths), default=0)
    if presentation.memory < required_memory:
        raise ValueError("presentation memory cannot encode its forbidden rules")

    candidate_states = enumeration_size(len(presentation.alphabet), presentation.memory)
    candidate_extensions = enumeration_size(
        len(presentation.alphabet), presentation.memory + 1
    )

    # Account for canonicalization and every forbidden-factor scan performed
    # while deriving support and labeled overlap edges.
    normalization_work = sum(1 + length for length in forbidden_lengths)
    normalization_work += sum(
        _contains_work(word_length, factor_length)
        for word_length in forbidden_lengths
        for factor_length in forbidden_lengths
    )
    rule_scan_work = (candidate_states + candidate_extensions) * sum(
        _contains_work(word_length, factor_length)
        for word_length in (presentation.memory, presentation.memory + 1)
        for factor_length in forbidden_lengths
    )
    # Candidate construction, graph construction, SCC discovery, and the two
    # reachability traversals each visit the bounded support axes.
    support_work = 4 * (candidate_states + candidate_extensions)
    state_cells = state_count * state_count
    overlap_work = state_cells * max(1, presentation.memory)
    expected_work = state_cells * (
        len(presentation.alphabet) if not presentation.memory else 1
    )
    expected_rule_work = state_cells * sum(
        _contains_work(presentation.memory + 1, factor_length)
        for factor_length in forbidden_lengths
    )
    state_rule_work = state_count * sum(
        _contains_work(presentation.memory, factor_length)
        for factor_length in forbidden_lengths
    )
    work = (
        carrier_work
        + normalization_work
        + rule_scan_work
        + support_work
        + len(presentation.transitions) * max(1, presentation.memory)
        + state_cells
        + overlap_work
        + expected_work
        + expected_rule_work
        + state_rule_work
    )
    if work > MAX_PRESENTATION_VERIFICATION_WORK:
        raise ValueError("presentation verification exceeds the work bound")


def require_zeta_budget(shift: AdjacencyShift) -> None:
    """Admit the determinant and its derivable exact result size."""

    states = len(shift.matrix)
    maximum_entry = max(
        (entry for row in shift.matrix for entry in row),
        default=0,
    )
    # The degree-k coefficient is a sum of principal k-minors. Its absolute
    # value is at most C(states,k) k! M^k <= (states M)^k.
    coefficient_digits = states * len(str(states * max(1, maximum_entry))) + 1
    if coefficient_digits > MAX_ZETA_COEFFICIENT_DIGITS:
        raise ValueError("zeta polynomial exceeds the coefficient digit bound")
    work = states**4
    if work > MAX_ZETA_WORK:
        raise ValueError("zeta determinant exceeds the work bound")
    result_digits = 2 * (states + 1) * coefficient_digits
    if result_digits > MAX_ZETA_RESULT_DIGITS:
        raise ValueError("zeta result exceeds the aggregate digit bound")


__all__ = [
    "MAX_PERIODIC_PROFILE_DIGITS",
    "MAX_PERIODIC_PROFILE_WORK",
    "MAX_PRESENTATION_VERIFICATION_WORK",
    "enumeration_size",
    "normalize_forbidden_blocks",
    "presentation_memory",
    "require_bounded_presentation",
    "require_bounded_presentation_verification",
    "require_bounded_support",
    "require_zeta_budget",
]
