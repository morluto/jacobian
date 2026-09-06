"""Exact bounded native kernels for symbolic dynamics."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.dynamics.symbolic._bounds import (
    MAX_PERIODIC_PROFILE_DIGITS,
    MAX_PERIODIC_PROFILE_WORK,
    _contains,
    enumeration_size,
    normalize_forbidden_blocks,
    presentation_memory,
    require_bounded_presentation,
    require_bounded_presentation_verification,
    require_bounded_support,
    require_zeta_budget,
)
from jacobian.math.dynamics.symbolic.values import (
    MAX_ADJACENCY_ENTRY,
    MAX_ADJACENCY_STATES,
    MAX_ALPHABET_SIZE,
    MAX_FORBIDDEN_BLOCK_LENGTH,
    MAX_FORBIDDEN_BLOCKS,
    MAX_PERIOD,
    MAX_PRESENTATION_TRANSITIONS,
    MAX_SYMBOL_LENGTH,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
    LabeledTransition,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_polynomial_from_sympy,
)
from jacobian.math.polynomials.values import RationalFunction, RationalPolynomial


def _admit[T](
    admission: Callable[[], T],
    *,
    location: tuple[str | int, ...],
    code: str,
) -> T:
    try:
        return admission()
    except OperationDomainValidationError:
        raise
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(exc),
        ) from exc


def block_language(
    shift: ForbiddenBlockShift, block_length: int
) -> tuple[tuple[str, ...], ...]:
    if block_length < 0:
        raise OperationDomainValidationError(
            location=("block_length",),
            code="symbolic_dynamics.block_length_negative",
            message="block length must be nonnegative",
        )
    _admit(
        lambda: enumeration_size(len(shift.alphabet), block_length),
        location=("block_length",),
        code="symbolic_dynamics.block_enumeration_not_admitted",
    )
    forbidden = normalize_forbidden_blocks(shift)
    if () in forbidden:
        return ()
    memory = _admit(
        lambda: require_bounded_support(shift),
        location=("shift",),
        code="symbolic_dynamics.shift_support_not_admitted",
    )
    states, _, left_infinite, right_infinite = _presentation_support(shift, memory)
    if block_length == 0:
        return (
            ((),)
            if right_infinite
            and (not shift.two_sided or left_infinite & right_infinite)
            else ()
        )
    candidates = _locally_allowed_words(shift, block_length)
    if memory == 0:
        return candidates if right_infinite else ()
    if block_length < memory:
        supported_states = (
            left_infinite & right_infinite if shift.two_sided else right_infinite
        )
        supported_prefixes = {state[:block_length] for state in supported_states}
        return tuple(word for word in candidates if word in supported_prefixes)
    state_index = set(states)
    occurring = []
    for word in candidates:
        starts = (word[:memory],)
        ends = (word[-memory:],)
        if any(
            start in state_index
            and end in right_infinite
            and (not shift.two_sided or start in left_infinite)
            for start in starts
            for end in ends
        ):
            occurring.append(word)
    return tuple(occurring)


def _locally_allowed_words(
    shift: ForbiddenBlockShift, length: int
) -> tuple[tuple[str, ...], ...]:
    forbidden = normalize_forbidden_blocks(shift)
    return tuple(
        word
        for word in itertools.product(shift.alphabet, repeat=length)
        if not any(_contains(word, excluded) for excluded in forbidden)
    )


def _presentation_support(
    shift: ForbiddenBlockShift, memory: int
) -> tuple[
    tuple[tuple[str, ...], ...],
    nx.DiGraph[tuple[str, ...]],
    set[tuple[str, ...]],
    set[tuple[str, ...]],
]:
    import networkx as nx

    states = _locally_allowed_words(shift, memory)
    extensions = _locally_allowed_words(shift, memory + 1)
    graph: nx.DiGraph[tuple[str, ...]] = nx.DiGraph()
    graph.add_nodes_from(states)
    for word in extensions:
        source = word[:memory] if memory else ()
        target = word[-memory:] if memory else ()
        if source in graph and target in graph:
            graph.add_edge(source, target)
    cyclic = {
        node
        for component in nx.strongly_connected_components(graph)
        for node in component
        if len(component) > 1 or graph.has_edge(node, node)
    }
    right_infinite = _reachable_from(graph, cyclic, reverse=True)
    left_infinite = _reachable_from(graph, cyclic, reverse=False)
    return states, graph, left_infinite, right_infinite


def _reachable_from(
    graph: nx.DiGraph[tuple[str, ...]],
    starts: set[tuple[str, ...]],
    *,
    reverse: bool,
) -> set[tuple[str, ...]]:
    reached = set(starts)
    pending = list(starts)
    while pending:
        node = pending.pop()
        neighbors = graph.predecessors(node) if reverse else graph.successors(node)
        for neighbor in neighbors:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _presentation_from_states_and_words(
    shift: ForbiddenBlockShift,
    memory: int,
    states: tuple[tuple[str, ...], ...],
    extension_words: tuple[tuple[str, ...], ...],
) -> BlockPresentation:
    state_index = {state: index for index, state in enumerate(states)}
    transitions: list[LabeledTransition] = []
    for word in extension_words:
        source_block = word[:memory] if memory else ()
        target_block = word[-memory:] if memory else ()
        source = state_index.get(source_block)
        target = state_index.get(target_block)
        if source is not None and target is not None:
            transitions.append(
                LabeledTransition(
                    source=source,
                    target=target,
                    appended_symbol=word[-1],
                )
            )
    size = len(states)
    adjacency = [[0] * size for _ in range(size)]
    for transition in transitions:
        adjacency[transition.source][transition.target] += 1
    return BlockPresentation(
        alphabet=shift.alphabet,
        memory=memory,
        state_blocks=states,
        forbidden_blocks=normalize_forbidden_blocks(shift),
        transitions=tuple(transitions),
        adjacency_matrix=tuple(tuple(row) for row in adjacency),
        two_sided=shift.two_sided,
    )


def finite_type_presentation(shift: ForbiddenBlockShift) -> BlockPresentation:
    memory = presentation_memory(shift)
    _admit(
        lambda: require_bounded_presentation(shift, memory),
        location=("shift",),
        code="symbolic_dynamics.presentation_not_admitted",
    )
    states = block_language(shift, memory)
    extensions = block_language(shift, memory + 1)
    return _presentation_from_states_and_words(shift, memory, states, extensions)


def higher_block_presentation(
    shift: ForbiddenBlockShift, block_length: int
) -> BlockPresentation:
    if block_length < 1:
        raise OperationDomainValidationError(
            location=("block_length",),
            code="symbolic_dynamics.higher_block_length_nonpositive",
            message="higher-block length must be positive",
        )
    required_memory = presentation_memory(shift)
    if block_length < required_memory:
        raise OperationDomainValidationError(
            location=("block_length",),
            code="symbolic_dynamics.higher_block_below_memory",
            message="higher-block length is below the presentation memory",
        )
    _admit(
        lambda: require_bounded_presentation(shift, block_length),
        location=("shift", "block_length"),
        code="symbolic_dynamics.presentation_not_admitted",
    )
    states = block_language(shift, block_length)
    extensions = block_language(shift, block_length + 1)
    return _presentation_from_states_and_words(shift, block_length, states, extensions)


def adjacency_shift(
    matrix: tuple[tuple[int, ...], ...], *, two_sided: bool = True
) -> AdjacencyShift:
    return AdjacencyShift(matrix=matrix, two_sided=two_sided)


def adjacency_shift_from_presentation(
    presentation: BlockPresentation,
) -> AdjacencyShift:
    """Map a verified presentation source to its ordered adjacency target."""
    if not verify_block_presentation(presentation):
        raise OperationDomainValidationError(
            location=("presentation",),
            code="symbolic_dynamics.presentation_adjacency_invalid",
            message=(
                "presentation adjacency must be a complete count of its labeled "
                "overlap transitions"
            ),
        )
    return adjacency_shift(
        presentation.adjacency_matrix,
        two_sided=presentation.two_sided,
    )


def _verify_presentation_symbols(
    alphabet: tuple[Any, ...],
    state_blocks: tuple[Any, ...],
    forbidden_blocks: tuple[Any, ...],
    memory: int,
) -> set[str] | None:
    """Validate exact symbol/block containers before hashing or comparison."""

    if any(
        type(symbol) is not str or not symbol or len(symbol) > MAX_SYMBOL_LENGTH
        for symbol in alphabet
    ):
        return None
    alphabet_set = set(alphabet)
    if len(alphabet_set) != len(alphabet):
        return None
    rank = {symbol: index for index, symbol in enumerate(alphabet)}
    for block in (*state_blocks, *forbidden_blocks):
        if type(block) is not tuple or len(block) > MAX_FORBIDDEN_BLOCK_LENGTH:
            return None
        if any(
            type(symbol) is not str
            or not symbol
            or len(symbol) > MAX_SYMBOL_LENGTH
            or symbol not in alphabet_set
            for symbol in block
        ):
            return None
    ordered_forbidden = sorted(
        forbidden_blocks,
        key=lambda block: (len(block), tuple(rank[symbol] for symbol in block)),
    )
    minimal_forbidden: list[tuple[str, ...]] = []
    for factor in ordered_forbidden:
        if not any(_contains(factor, prior) for prior in minimal_forbidden):
            minimal_forbidden.append(factor)
    if forbidden_blocks != tuple(minimal_forbidden):
        return None
    if any(type(block) is not tuple or len(block) != memory for block in state_blocks):
        return None
    if len(set(state_blocks)) != len(state_blocks):
        return None
    if tuple(state_blocks) != tuple(
        sorted(state_blocks, key=lambda block: tuple(rank[symbol] for symbol in block))
    ):
        return None
    if any(
        _contains(block, factor)
        for block in state_blocks
        for factor in forbidden_blocks
    ):
        return None
    return alphabet_set


def _verify_presentation_matrix(matrix: tuple[Any, ...], state_count: int) -> bool:
    """Validate exact bounded adjacency rows without semantic recomputation."""

    for row in matrix:
        if type(row) is not tuple or len(row) != state_count:
            return False
        if any(
            type(entry) is not int or entry < 0 or entry > MAX_ADJACENCY_ENTRY
            for entry in row
        ):
            return False
    return True


def _verify_presentation_transitions(transitions: tuple[Any, ...]) -> bool:
    """Validate transition records before the bounded relation scan."""

    for transition in transitions:
        if type(transition) is not LabeledTransition:
            return False
        if (
            type(transition.source) is not int
            or type(transition.target) is not int
            or type(transition.appended_symbol) is not str
            or not transition.appended_symbol
            or len(transition.appended_symbol) > MAX_SYMBOL_LENGTH
        ):
            return False
    return True


def _verify_presentation_carrier(claim: BlockPresentation) -> set[str] | None:
    """Check the bounded axes and return the canonical alphabet set."""

    alphabet = claim.alphabet
    state_blocks = claim.state_blocks
    forbidden_blocks = claim.forbidden_blocks
    matrix = claim.adjacency_matrix
    transitions = claim.transitions
    if (
        type(alphabet) is not tuple
        or type(state_blocks) is not tuple
        or type(forbidden_blocks) is not tuple
        or type(matrix) is not tuple
        or type(transitions) is not tuple
        or type(claim.memory) is not int
        or claim.memory < 0
        or claim.memory > MAX_FORBIDDEN_BLOCK_LENGTH
        or type(claim.two_sided) is not bool
    ):
        return None
    if (
        not 1 <= len(alphabet) <= MAX_ALPHABET_SIZE
        or len(state_blocks) > MAX_ADJACENCY_STATES
        or len(forbidden_blocks) > MAX_FORBIDDEN_BLOCKS
        or len(matrix) != len(state_blocks)
        or len(transitions) > MAX_PRESENTATION_TRANSITIONS
    ):
        return None
    try:
        require_bounded_presentation_verification(claim)
    except (TypeError, ValueError):
        return None
    alphabet_set = _verify_presentation_symbols(
        alphabet, state_blocks, forbidden_blocks, claim.memory
    )
    if alphabet_set is None:
        return None
    if not _verify_presentation_matrix(matrix, len(state_blocks)):
        return None
    if not _verify_presentation_transitions(transitions):
        return None
    return alphabet_set


def _actual_presentation_edges(
    claim: BlockPresentation, alphabet_set: set[str]
) -> tuple[dict[tuple[int, int, str], int], list[list[int]]] | None:
    """Collect supplied transitions while checking their local overlap labels."""

    size = len(claim.state_blocks)
    actual_edges: dict[tuple[int, int, str], int] = {}
    counts = [[0] * size for _ in range(size)]
    for transition in claim.transitions:
        source = transition.source
        target = transition.target
        symbol = transition.appended_symbol
        if (
            type(source) is not int
            or type(target) is not int
            or not 0 <= source < size
            or not 0 <= target < size
            or type(symbol) is not str
            or symbol not in alphabet_set
        ):
            return None
        source_block = claim.state_blocks[source]
        target_block = claim.state_blocks[target]
        if claim.memory:
            if target_block[:-1] != source_block[1:] or target_block[-1] != symbol:
                return None
        elif source_block != () or target_block != ():
            return None
        key = (source, target, symbol)
        actual_edges[key] = actual_edges.get(key, 0) + 1
        counts[source][target] += 1
    return actual_edges, counts


def _expected_occurring_state_blocks(
    claim: BlockPresentation,
) -> tuple[tuple[str, ...], ...]:
    """Derive the canonical state axis from the complete source shift."""

    import networkx as nx

    alphabet = claim.alphabet
    forbidden = claim.forbidden_blocks

    def allowed(word: tuple[str, ...]) -> bool:
        return not any(_contains(word, factor) for factor in forbidden)

    candidates = tuple(
        word
        for word in itertools.product(alphabet, repeat=claim.memory)
        if allowed(word)
    )
    candidate_set = set(candidates)
    extensions = (
        word
        for word in itertools.product(alphabet, repeat=claim.memory + 1)
        if allowed(word)
    )
    graph: nx.DiGraph[tuple[str, ...]] = nx.DiGraph()
    graph.add_nodes_from(candidates)
    for word in extensions:
        source = word[: claim.memory] if claim.memory else ()
        target = word[-claim.memory :] if claim.memory else ()
        if source in candidate_set and target in candidate_set:
            graph.add_edge(source, target)
    cyclic = {
        node
        for component in nx.strongly_connected_components(graph)
        for node in component
        if len(component) > 1 or graph.has_edge(node, node)
    }
    right_infinite = _reachable_from(graph, cyclic, reverse=True)
    left_infinite = _reachable_from(graph, cyclic, reverse=False)
    return tuple(
        state
        for state in candidates
        if state in right_infinite and (not claim.two_sided or state in left_infinite)
    )


def _expected_presentation_edges(
    claim: BlockPresentation,
) -> dict[tuple[int, int, str], int]:
    """Derive every allowed labeled overlap from the retained source rules."""

    expected_edges: dict[tuple[int, int, str], int] = {}
    forbidden = claim.forbidden_blocks
    for source_index, source_block in enumerate(claim.state_blocks):
        for target_index, target_block in enumerate(claim.state_blocks):
            if claim.memory:
                if source_block[1:] != target_block[:-1]:
                    continue
                extension = (*source_block, target_block[-1])
                if any(_contains(extension, factor) for factor in forbidden):
                    continue
                expected_edges[(source_index, target_index, target_block[-1])] = 1
            else:
                for symbol in claim.alphabet:
                    if not any(_contains((symbol,), factor) for factor in forbidden):
                        expected_edges[(source_index, target_index, symbol)] = 1
    return expected_edges


def _verify_block_presentation(claim: BlockPresentation) -> bool:
    """Check the complete serialized transition relation and its counts."""

    if type(claim) is not BlockPresentation:
        return False
    try:
        alphabet_set = _verify_presentation_carrier(claim)
        if alphabet_set is None:
            return False
        if claim.state_blocks != _expected_occurring_state_blocks(claim):
            return False
        actual = _actual_presentation_edges(claim, alphabet_set)
        if actual is None:
            return False
        actual_edges, counts = actual
        return actual_edges == _expected_presentation_edges(claim) and (
            claim.adjacency_matrix == tuple(tuple(row) for row in counts)
        )
    except Exception:
        return False


def verify_block_presentation(claim: BlockPresentation) -> bool:
    """Verify all overlap transitions and their serialized adjacency counts."""

    try:
        return _verify_block_presentation(claim)
    except Exception:
        return False


def _mobius_sieve(limit: int) -> tuple[int, ...]:
    """Return mu(0)..mu(limit) by a bounded Eratosthenes sieve."""

    values = [1] * (limit + 1)
    is_prime = [True] * (limit + 1)
    for prime in range(2, limit + 1):
        if not is_prime[prime]:
            continue
        for multiple in range(prime, limit + 1, prime):
            is_prime[multiple] = False
            values[multiple] *= -1
        square = prime * prime
        for multiple in range(square, limit + 1, square):
            values[multiple] = 0
    return tuple(values)


def periodic_point_profile(
    shift: AdjacencyShift, max_period: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not 1 <= max_period <= MAX_PERIOD:
        raise OperationDomainValidationError(
            location=("max_period",),
            code="symbolic_dynamics.period_out_of_bounds",
            message="max period is outside the supported bounds",
        )
    matrix = shift.matrix
    if not matrix:
        empty = (0,) * max_period
        return empty, empty, empty
    states = len(matrix)
    matrix_work = states**3 * max_period
    divisor_work = 3 * max_period * (max_period.bit_length() + 1)
    if matrix_work + divisor_work > MAX_PERIODIC_PROFILE_WORK:
        raise OperationDomainValidationError(
            location=("shift", "max_period"),
            code="symbolic_dynamics.periodic_profile_work_bound",
            message="periodic-point matrix powering exceeds the work bound",
        )
    maximum_row_sum = max(sum(row) for row in matrix)
    count_bits = (
        states.bit_length() + max_period * (max(1, maximum_row_sum) - 1).bit_length()
    )
    count_digits = max(1, (count_bits * 30_103 + 99_999) // 100_000)
    aggregate_digits = 3 * max_period * count_digits
    if aggregate_digits > MAX_PERIODIC_PROFILE_DIGITS:
        raise OperationDomainValidationError(
            location=("shift", "max_period"),
            code="symbolic_dynamics.periodic_profile_output_bound",
            message="periodic-point profile exceeds the output digit bound",
        )
    from jacobian.math.dynamics.symbolic._flint import matrix_power_traces

    fixed = matrix_power_traces(matrix, max_period)
    mobius = _mobius_sieve(max_period)
    exact_values = [0] * max_period
    for divisor in range(1, max_period + 1):
        multiplier = mobius[divisor]
        if multiplier:
            for period in range(divisor, max_period + 1, divisor):
                exact_values[period - 1] += multiplier * fixed[period // divisor - 1]
    exact = tuple(exact_values)
    if any(count < 0 or count % period for period, count in enumerate(exact, 1)):
        raise RuntimeError("periodic-point inversion violated orbit integrality")
    orbits = tuple(count // period for period, count in enumerate(exact, 1))
    return tuple(fixed), exact, orbits


def _determinant_coefficients(shift: AdjacencyShift) -> tuple[int, ...]:
    """Return ascending coefficients of ``det(I - t A)``."""

    from sympy import Matrix

    characteristic = Matrix(shift.matrix).charpoly()
    # det(lambda I - A) = sum(c_k lambda^(n-k)), so the same coefficient
    # sequence in ascending powers of t is det(I - t A).
    return tuple(int(coefficient) for coefficient in characteristic.all_coeffs())


def artin_mazur_zeta(
    shift: AdjacencyShift,
) -> tuple[RationalPolynomial, RationalFunction]:
    """Return ``det(I-tA)`` and ``1/det(I-tA)``."""

    _admit(
        lambda: require_zeta_budget(shift),
        location=("shift",),
        code="symbolic_dynamics.zeta_not_admitted",
    )
    from sympy import QQ, Poly, Symbol

    variable = Symbol("t")
    coefficients = _determinant_coefficients(shift)
    determinant = Poly.from_dict(
        {(exponent,): coefficient for exponent, coefficient in enumerate(coefficients)},
        variable,
        domain=QQ,
    )
    return (
        rational_polynomial_from_sympy(
            determinant, ("t",), maximum_terms=MAX_ADJACENCY_STATES + 1
        ),
        rational_function_from_sympy(
            1 / determinant.as_expr(),
            ("t",),
            maximum_terms=MAX_ADJACENCY_STATES + 1,
        ),
    )


__all__ = [
    "adjacency_shift",
    "adjacency_shift_from_presentation",
    "artin_mazur_zeta",
    "block_language",
    "enumeration_size",
    "finite_type_presentation",
    "higher_block_presentation",
    "normalize_forbidden_blocks",
    "periodic_point_profile",
    "verify_block_presentation",
]
