"""Exact symbol-level Parikh profiles for accepted DFA words."""

from math import comb
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_RESULT_DIGITS,
    MAX_DFA_ALPHABET,
)

MAX_SYMBOL_PARIKH_LENGTH = 1_000
MAX_SYMBOL_PARIKH_DP_WORK = 2_000_000
MAX_SYMBOL_PARIKH_CELLS = 43_000


class SymbolParikhProfileRequest(StrictModel):
    dfa: DFA
    word_length: StrictInt = Field(ge=0, le=MAX_SYMBOL_PARIKH_LENGTH)


class SymbolParikhCell(StrictModel):
    symbol_counts: tuple[StrictInt, ...] = Field(max_length=MAX_DFA_ALPHABET)
    multiplicity: ExactInteger = Field(ge=1)


class SymbolParikhProfileResult(StrictModel):
    dfa: DFA
    alphabet: tuple[StrictInt, ...] = Field(max_length=MAX_DFA_ALPHABET)
    word_length: StrictInt = Field(ge=0, le=MAX_SYMBOL_PARIKH_LENGTH)
    cells: tuple[SymbolParikhCell, ...] = Field(max_length=MAX_SYMBOL_PARIKH_CELLS)
    total_accepted_words: ExactInteger

    @classmethod
    def _from_kernel(
        cls,
        request: SymbolParikhProfileRequest,
        *,
        cells: tuple[SymbolParikhCell, ...],
        total_accepted_words: ExactInteger,
    ) -> Self:
        """Construct a profile after the trusted DP established its invariants."""

        return cls.model_construct(
            dfa=request.dfa,
            alphabet=tuple(range(request.dfa.alphabet_size)),
            word_length=request.word_length,
            cells=cells,
            total_accepted_words=total_accepted_words,
        )

    @model_validator(mode="after")
    def require_canonical_cells(self) -> Self:
        if self.alphabet != tuple(range(self.dfa.alphabet_size)):
            raise ValueError("symbol-Parikh alphabet must be the DFA's ordered axis")
        vectors = tuple(cell.symbol_counts for cell in self.cells)
        if vectors != tuple(sorted(set(vectors))):
            raise ValueError(
                "symbol-Parikh cells must be lexicographically sorted and unique"
            )
        for vector in vectors:
            if len(vector) != self.dfa.alphabet_size:
                raise ValueError(
                    "symbol-Parikh vectors must use the complete alphabet axis"
                )
            if any(count < 0 for count in vector) or sum(vector) != self.word_length:
                raise ValueError(
                    "symbol-Parikh vectors must be nonnegative and sum to word_length"
                )
        if self.total_accepted_words != sum(cell.multiplicity for cell in self.cells):
            raise ValueError(
                "total_accepted_words must equal the sum of cell multiplicities"
            )
        return self


def _build_transition_index(
    dfa: DFA,
) -> dict[tuple[int, int], int]:
    return {
        (transition.source, transition.symbol): transition.target
        for transition in dfa.transitions
    }


def _reachable_states_without_index(dfa: DFA) -> set[int]:
    reachable = {dfa.initial_state}
    frontier = [dfa.initial_state]
    while frontier:
        source = frontier.pop()
        for transition in dfa.transitions:
            if transition.source != source or transition.target in reachable:
                continue
            reachable.add(transition.target)
            frontier.append(transition.target)
    return reachable


def _letter_maps_on_reachable(
    dfa: DFA, reachable: set[int]
) -> tuple[dict[int, dict[int, int]], int]:
    by_source: dict[int, dict[int, int]] = {}
    scanned = 0
    for transition in dfa.transitions:
        scanned += 1
        if transition.source not in reachable:
            continue
        by_source.setdefault(transition.source, {})[transition.symbol] = (
            transition.target
        )
    return by_source, scanned


def _letter_actions_commute(
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> tuple[bool, int]:
    compared = 0
    commute = True
    alphabet = range(alphabet_size)
    for source in reachable:
        outgoing = by_source.get(source, {})
        for first in alphabet:
            image_first = outgoing.get(first)
            first_reachable = image_first is not None and image_first in reachable
            first_outgoing = (
                by_source.get(image_first, {})
                if first_reachable and image_first is not None
                else {}
            )
            for second in alphabet:
                compared += 1
                if not commute:
                    continue
                if not first_reachable:
                    commute = False
                    continue
                image_second = outgoing.get(second)
                if image_second is None or image_second not in reachable:
                    commute = False
                    continue
                if first_outgoing.get(second) != by_source.get(image_second, {}).get(
                    first
                ):
                    commute = False
    return commute, compared


def _reachable_letter_graph(
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {state: set() for state in reachable}
    for source in reachable:
        outgoing = by_source.get(source, {})
        for symbol in range(alphabet_size):
            target = outgoing.get(symbol)
            if target in reachable:
                graph[source].add(target)
    return graph


def _cyclic_components(graph: dict[int, set[int]]) -> set[int]:
    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    cyclic: set[int] = set()

    def strongconnect(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for target in graph[vertex]:
            if target not in indices:
                strongconnect(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif target in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[target])
        if lowlink[vertex] == indices[vertex]:
            component: list[int] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == vertex:
                    break
            looping = len(component) > 1 or vertex in graph[vertex]
            if looping:
                cyclic.update(component)

    for state in graph:
        if state not in indices:
            strongconnect(state)
    return cyclic


def _persistent_reachable_states(
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> set[int]:
    graph = _reachable_letter_graph(by_source, reachable, alphabet_size)
    cyclic = _cyclic_components(graph)
    persistent = set(cyclic)
    frontier = list(cyclic)
    while frontier:
        source = frontier.pop()
        for target in graph[source]:
            if target in persistent:
                continue
            persistent.add(target)
            frontier.append(target)
    return persistent


def _longest_transient_walk(
    by_source: dict[int, dict[int, int]],
    transient: set[int],
    alphabet_size: int,
    initial_state: int,
) -> int:
    if initial_state not in transient:
        return -1
    longest = dict.fromkeys(transient, -1)
    longest[initial_state] = 0
    for _ in range(len(transient)):
        changed = False
        for source in transient:
            if longest[source] < 0:
                continue
            outgoing = by_source.get(source, {})
            for symbol in range(alphabet_size):
                target = outgoing.get(symbol)
                if target not in transient:
                    continue
                candidate = longest[source] + 1
                if candidate > longest[target]:
                    longest[target] = candidate
                    changed = True
        if not changed:
            break
    return max(longest.values(), default=-1)


def _noncommuting_analysis_work(reachable_count: int, alphabet_size: int) -> int:
    """Charge persistent-graph, SCC/closure, and transient-walk phases."""

    edge_probes = reachable_count * alphabet_size
    scc_and_closure = 2 * (reachable_count + edge_probes)
    transient_relaxations = reachable_count * reachable_count * alphabet_size
    return edge_probes + scc_and_closure + transient_relaxations


def _partition_find(parent: dict[int, int], state: int) -> int:
    root = state
    while parent[root] != root:
        root = parent[root]
    while parent[state] != root:
        parent[state], state = root, parent[state]
    return root


def _partition_union(parent: dict[int, int], left: int, right: int) -> bool:
    root_left = _partition_find(parent, left)
    root_right = _partition_find(parent, right)
    if root_left == root_right:
        return False
    parent[root_right] = root_left
    return True


def _union_letter_commutators(
    parent: dict[int, int],
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> int:
    probes = 0
    alphabet = range(alphabet_size)
    for source in reachable:
        outgoing = by_source.get(source, {})
        for first in alphabet:
            image_first = outgoing.get(first)
            first_reachable = image_first is not None and image_first in reachable
            first_outgoing = (
                by_source.get(image_first, {})
                if first_reachable and image_first is not None
                else {}
            )
            for second in alphabet:
                probes += 1
                image_second = outgoing.get(second)
                if (
                    not first_reachable
                    or image_first is None
                    or image_second is None
                    or image_second not in reachable
                ):
                    continue
                ab = first_outgoing.get(second)
                ba = by_source.get(image_second, {}).get(first)
                if ab in reachable and ba in reachable:
                    _partition_union(parent, ab, ba)
    return probes


def _close_letter_congruence(
    parent: dict[int, int],
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> int:
    probes = 0
    alphabet = range(alphabet_size)
    changed = True
    while changed:
        changed = False
        for symbol in alphabet:
            images: dict[int, int] = {}
            for source in reachable:
                probes += 1
                target = by_source.get(source, {}).get(symbol)
                if target is None or target not in reachable:
                    continue
                root = _partition_find(parent, source)
                seen = images.get(root)
                if seen is None:
                    images[root] = target
                elif _partition_union(parent, seen, target):
                    changed = True
    return probes


def _parikh_vector_state_bound(
    by_source: dict[int, dict[int, int]],
    reachable: set[int],
    alphabet_size: int,
) -> tuple[int, int]:
    """Bound how many states one symbol-count vector can occupy.

    Words with the same Parikh vector differ by adjacent letter swaps. The
    congruence generated by those commutators, closed under letter actions,
    therefore bounds per-vector state multiplicity by the size of an action
    factor rather than by every persistent state.
    """

    parent = {state: state for state in reachable}
    probes = _union_letter_commutators(
        parent, by_source, reachable, alphabet_size
    ) + _close_letter_congruence(parent, by_source, reachable, alphabet_size)
    sizes: dict[int, int] = {}
    for state in reachable:
        root = _partition_find(parent, state)
        sizes[root] = sizes.get(root, 0) + 1
    return max(sizes.values(), default=1), probes


def _states_per_layer(
    *,
    length: int,
    reachable: set[int],
    commute: bool,
    persistent: set[int],
    max_transient_step: int,
    states_per_vector: int,
) -> list[int]:
    reachable_count = len(reachable)
    persistent_bound = min(len(persistent), states_per_vector)
    transient_bound = min(reachable_count, states_per_vector)
    bounds: list[int] = []
    for step in range(length + 1):
        if commute:
            bounds.append(1)
        elif step > max_transient_step:
            bounds.append(persistent_bound)
        else:
            bounds.append(transient_bound)
    return bounds


def _extend_profile_layer(
    layer: dict[tuple[int, tuple[int, ...]], int],
    transitions: dict[tuple[int, int], int],
    reachable: set[int],
    alphabet_size: int,
) -> dict[tuple[int, tuple[int, ...]], int]:
    next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
    for (state, counts), multiplicity in layer.items():
        for symbol in range(alphabet_size):
            target = transitions[(state, symbol)]
            if target not in reachable:
                continue
            updated = (*counts[:symbol], counts[symbol] + 1, *counts[symbol + 1 :])
            key = (target, updated)
            next_layer[key] = next_layer.get(key, 0) + multiplicity
    return next_layer


def _collect_profile(
    layer: dict[tuple[int, tuple[int, ...]], int],
    accepting: set[int],
) -> dict[tuple[int, ...], int]:
    profile: dict[tuple[int, ...], int] = {}
    for (state, counts), multiplicity in layer.items():
        if state in accepting:
            profile[counts] = profile.get(counts, 0) + multiplicity
    return profile


def symbol_parikh_profile(
    request: SymbolParikhProfileRequest,
) -> SymbolParikhProfileResult:
    dfa = request.dfa
    length = request.word_length
    alphabet_size = dfa.alphabet_size
    if alphabet_size == 0:
        total = int(length == 0 and dfa.initial_state in dfa.accepting_states)
        cells = (SymbolParikhCell(symbol_counts=(), multiplicity=1),) if total else ()
        return SymbolParikhProfileResult._from_kernel(
            request,
            cells=cells,
            total_accepted_words=total,
        )
    output_bound = comb(length + alphabet_size - 1, alphabet_size - 1)
    transition_count = dfa.state_count * alphabet_size
    # The layer at step t contains weak compositions of t, so only layers
    # t=0..length-1 are extended.  The final layer is materialized separately
    # into profile cells and must not be charged as another transition layer.
    # Reachability scans the raw transition axis once per discovered source.
    # Admit its declared-dimension upper bound before doing that scan or
    # building the full transition index.
    preflight_scan_work = dfa.state_count * transition_count
    if output_bound > MAX_SYMBOL_PARIKH_CELLS or (
        transition_count + preflight_scan_work > MAX_SYMBOL_PARIKH_DP_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.profile_bound",
            message="symbol-Parikh DP or output exceeds its exact admitted bound",
        )
    if length * max(1, len(str(alphabet_size))) > MAX_COUNT_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.count_bound",
            message="symbol-Parikh multiplicities exceed the exact integer digit bound",
        )
    reachable = _reachable_states_without_index(dfa)
    by_source, commute_scan_work = _letter_maps_on_reachable(dfa, reachable)
    commute, commute_compare_work = _letter_actions_commute(
        by_source, reachable, alphabet_size
    )
    commute_preflight_work = commute_scan_work + commute_compare_work
    reachable_count = len(reachable)
    analysis_work = (
        0 if commute else _noncommuting_analysis_work(reachable_count, alphabet_size)
    )
    if commute:
        states_per_vector = 1
        vector_state_work = 0
    else:
        states_per_vector, _ = _parikh_vector_state_bound(
            by_source, reachable, alphabet_size
        )
        vector_state_work = (
            reachable_count * alphabet_size * alphabet_size
            + reachable_count * reachable_count * alphabet_size
        )
    persistent = (
        reachable
        if commute
        else _persistent_reachable_states(by_source, reachable, alphabet_size)
    )
    transient = set() if commute else reachable - persistent
    max_transient_step = (
        -1
        if commute
        else _longest_transient_walk(
            by_source, transient, alphabet_size, dfa.initial_state
        )
    )
    layer_states = _states_per_layer(
        length=length,
        reachable=reachable,
        commute=commute,
        persistent=persistent,
        max_transient_step=max_transient_step,
        states_per_vector=states_per_vector,
    )
    # A DP layer has at most one entry per word of that length.  Capping the
    # composition bound by the possible word count keeps unreachable states
    # from charging combinations that the layer cannot contain. Transient
    # tree prefixes occupy only early layers, so later layers charge the
    # persistent (cyclic) fragment. Persistent occupancy is further capped by
    # the per-vector action-factor bound, not by every persistent state.
    extension_cells = 0
    possible_word_count = 1
    for step in range(length):
        layer_composition_bound = comb(step + alphabet_size - 1, alphabet_size - 1)
        extension_cells += min(
            layer_states[step] * layer_composition_bound,
            possible_word_count,
        )
        possible_word_count *= alphabet_size
    extension_coordinate_work = extension_cells * alphabet_size * max(1, alphabet_size)
    output_materialization_cells = min(
        layer_states[length] * output_bound,
        possible_word_count,
    )
    output_materialization_work = output_materialization_cells
    # Result cells are unique accepting count vectors, not the whole final
    # DP layer.  An empty accepting set therefore constructs no cells.
    if not set(dfa.accepting_states).intersection(reachable):
        collected_cells = 0
    else:
        collected_cells = min(
            output_bound,
            possible_word_count,
            output_materialization_cells,
        )
    cell_construction_work = collected_cells * max(1, alphabet_size)
    result_reduce_work = 2 * collected_cells
    # The transition index is built from every DFA edge, including edges from
    # states that are unreachable from the initial state.
    transition_index_work = transition_count
    reachability_scan_work = len(reachable) * transition_count
    work_bound = (
        transition_index_work
        + reachability_scan_work
        + commute_preflight_work
        + vector_state_work
        + analysis_work
        + extension_coordinate_work
        + output_materialization_work
        + cell_construction_work
        + result_reduce_work
    )
    if work_bound > MAX_SYMBOL_PARIKH_DP_WORK:
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.profile_bound",
            message="symbol-Parikh DP or output exceeds its exact admitted bound",
        )
    transitions = _build_transition_index(dfa)
    zero = (0,) * alphabet_size
    layer: dict[tuple[int, tuple[int, ...]], int] = {(dfa.initial_state, zero): 1}
    for _step in range(length):
        layer = _extend_profile_layer(layer, transitions, reachable, alphabet_size)
    accepting = set(dfa.accepting_states)
    profile = _collect_profile(layer, accepting)
    total = sum(profile.values())
    return SymbolParikhProfileResult._from_kernel(
        request,
        cells=tuple(
            SymbolParikhCell(symbol_counts=counts, multiplicity=multiplicity)
            for counts, multiplicity in sorted(profile.items())
        ),
        total_accepted_words=total,
    )
