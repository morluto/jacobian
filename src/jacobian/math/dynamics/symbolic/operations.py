"""Exact bounded native kernels for symbolic dynamics."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

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
    require_bounded_support,
    require_zeta_budget,
)
from jacobian.math.dynamics.symbolic.values import (
    MAX_ADJACENCY_STATES,
    MAX_PERIOD,
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
    "artin_mazur_zeta",
    "block_language",
    "enumeration_size",
    "finite_type_presentation",
    "higher_block_presentation",
    "normalize_forbidden_blocks",
    "periodic_point_profile",
]
