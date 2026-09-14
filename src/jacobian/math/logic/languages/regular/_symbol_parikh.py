"""Exact symbol-level Parikh profiles for accepted DFA words."""

import heapq
from collections.abc import Iterable
from itertools import islice
from math import comb
from typing import Annotated, Self, cast

from pydantic import ConfigDict, Field, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_RESULT_DIGITS,
    MAX_DFA_ALPHABET,
    MAX_DFA_STATES,
    MAX_DFA_TRANSITIONS,
)

MAX_SYMBOL_PARIKH_LENGTH = 1_000
MAX_SYMBOL_PARIKH_DP_WORK = 2_000_000
MAX_SYMBOL_PARIKH_CELLS = 43_000
_CHECKPOINT_STRIDE = 512


class SymbolParikhProfileRequest(StrictModel):
    """Request for the complete accepted-word symbol-count profile."""

    dfa: DFA = Field(
        description=(
            "Complete deterministic finite automaton; symbol coordinates use its "
            "ordered zero-based alphabet axis."
        )
    )
    word_length: StrictInt = Field(
        ge=0,
        le=MAX_SYMBOL_PARIKH_LENGTH,
        description="Exact nonnegative length of every counted accepted word.",
    )


class SymbolParikhCell(StrictModel):
    """One canonical symbol-count vector and its positive exact multiplicity."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        # A validation-bypassed cell must not be trusted when nested in a
        # public result, so revalidate instances of this model as fields.
        revalidate_instances="always",
    )

    symbol_counts: tuple[Annotated[StrictInt, Field(ge=0)], ...] = Field(
        max_length=MAX_DFA_ALPHABET,
        description=(
            "Dense nonnegative counts on the DFA alphabet axis; coordinates sum "
            "to the requested word length."
        ),
    )
    multiplicity: ExactInteger = Field(
        ge=1,
        description="Positive exact number of accepted words with this vector.",
    )


class SymbolParikhProfileResult(StrictModel):
    """Complete canonical map from symbol-count vectors to accepted words."""

    dfa: DFA = Field(description="The source DFA retained for composition.")
    alphabet: tuple[StrictInt, ...] = Field(
        max_length=MAX_DFA_ALPHABET,
        description="Ordered zero-based alphabet axis retained by every vector.",
    )
    word_length: StrictInt = Field(
        ge=0,
        le=MAX_SYMBOL_PARIKH_LENGTH,
        description="Exact length shared by every profile vector.",
    )
    cells: tuple[SymbolParikhCell, ...] = Field(
        max_length=MAX_SYMBOL_PARIKH_CELLS,
        description="Lexicographically sorted, unique nonzero profile cells.",
    )
    total_accepted_words: ExactInteger = Field(
        description="Exact sum of all cell multiplicities.",
    )

    @classmethod
    def _from_kernel(
        cls,
        dfa: DFA,
        word_length: int,
        *,
        cells: tuple[SymbolParikhCell, ...],
        total_accepted_words: ExactInteger,
    ) -> Self:
        """Construct a profile after the trusted DP established its invariants."""

        return cls.model_construct(
            dfa=dfa,
            alphabet=tuple(range(dfa.alphabet_size)),
            word_length=word_length,
            cells=cells,
            total_accepted_words=total_accepted_words,
        )

    @model_validator(mode="after")
    def require_canonical_cells(self) -> Self:
        # A model_construct DFA is trusted when nested, so rerun the DFA's own
        # contract before checking profile invariants against its fields.
        try:
            source = DFA.model_validate(
                {
                    "state_count": getattr(self.dfa, "state_count", None),
                    "alphabet_size": getattr(self.dfa, "alphabet_size", None),
                    "transitions": getattr(self.dfa, "transitions", None),
                    "initial_state": getattr(self.dfa, "initial_state", None),
                    "accepting_states": getattr(self.dfa, "accepting_states", None),
                }
            )
        except (ValidationError, AttributeError, TypeError) as exc:
            raise ValueError(
                "symbol-Parikh source must be a canonical total DFA"
            ) from exc
        if self.alphabet != tuple(range(source.alphabet_size)):
            raise ValueError("symbol-Parikh alphabet must be the DFA's ordered axis")
        vectors = tuple(cell.symbol_counts for cell in self.cells)
        for cell in self.cells:
            # A validation-bypassed nested cell can carry a non-positive
            # multiplicity because the direct constructor does not revalidate
            # existing instances; enforce the documented bound explicitly.
            if cell.multiplicity < 1:
                raise ValueError("symbol-Parikh cell multiplicities must be positive")
        if vectors != tuple(sorted(set(vectors))):
            raise ValueError(
                "symbol-Parikh cells must be lexicographically sorted and unique"
            )
        for vector in vectors:
            if len(vector) != source.alphabet_size:
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
        # Retain the canonical revalidated DFA so a validation-bypassed nested
        # carrier cannot survive into the public result.
        return self.model_copy(update={"dfa": source})


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


def _states_at_exact_depth(
    dfa: DFA,
    reachable: set[int],
    alphabet_size: int,
    length: int,
) -> set[int]:
    """Return the reachable states that are live after exactly ``length`` steps."""

    outgoing: dict[int, list[int]] = {}
    for transition in dfa.transitions:
        if transition.source in reachable and transition.target in reachable:
            outgoing.setdefault(transition.source, []).append(transition.target)
    frontier = {dfa.initial_state}
    for _ in range(length):
        request_checkpoint("during symbol-Parikh exact-depth reachability")
        if not frontier:
            return set()
        next_frontier: set[int] = set()
        for state in frontier:
            next_frontier.update(outgoing.get(state, ()))
        frontier = next_frontier
    return frontier


def _extend_profile_layer(
    layer: dict[tuple[int, tuple[int, ...]], int],
    transitions: dict[tuple[int, int], int],
    reachable: set[int],
    alphabet_size: int,
) -> dict[tuple[int, tuple[int, ...]], int]:
    next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
    visited = 0
    for (state, counts), multiplicity in layer.items():
        for symbol in range(alphabet_size):
            if visited % _CHECKPOINT_STRIDE == 0:
                request_checkpoint("during symbol-Parikh DP extension")
            visited += 1
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
    for index, ((state, counts), multiplicity) in enumerate(layer.items()):
        if index % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during symbol-Parikh profile collection")
        if state in accepting:
            profile[counts] = profile.get(counts, 0) + multiplicity
    return profile


def _compute_symbol_parikh_profile(
    dfa: DFA,
    length: int,
) -> SymbolParikhProfileResult:
    alphabet_size = dfa.alphabet_size
    if alphabet_size == 0:
        total = int(length == 0 and dfa.initial_state in dfa.accepting_states)
        cells = (SymbolParikhCell(symbol_counts=(), multiplicity=1),) if total else ()
        return SymbolParikhProfileResult._from_kernel(
            dfa,
            length,
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
    # Result cells are unique accepting count vectors in the final layer. Only
    # accepting states that are live after exactly `length` steps can
    # contribute. When no accepting state is even graph-reachable the exact-
    # depth traversal is skipped entirely; otherwise its cost is charged below.
    reachable_accepting = set(dfa.accepting_states).intersection(reachable)
    exact_depth_work = 0
    if not reachable_accepting:
        collected_cells = 0
    elif reachable_accepting == reachable:
        # Every reachable state accepts, so the exact-depth traversal cannot
        # sharpen `collected_cells`: any state it would exclude is already
        # bounded out above, and the cap below is the materialization bound.
        # Skipping it avoids charging traversal work that cannot change the
        # result and would otherwise reject an admitted request.
        collected_cells = min(
            output_bound,
            possible_word_count,
            output_materialization_cells,
        )
    else:
        exact_depth_work = len(dfa.transitions) + length * reachable_count * max(
            1, alphabet_size
        )
        accepting_at_depth = reachable_accepting.intersection(
            _states_at_exact_depth(dfa, reachable, alphabet_size, length)
        )
        if not accepting_at_depth:
            collected_cells = 0
        else:
            collected_cells = min(
                output_bound,
                possible_word_count,
                output_materialization_cells,
            )
    cell_construction_work = collected_cells * max(1, alphabet_size)
    result_reduce_work = 2 * collected_cells
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
        + exact_depth_work
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
    for step in range(length):
        if step % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during symbol-Parikh DP")
        layer = _extend_profile_layer(layer, transitions, reachable, alphabet_size)
    accepting = set(dfa.accepting_states)
    profile = _collect_profile(layer, accepting)
    total = sum(profile.values())
    materialized_cells: list[SymbolParikhCell] = []
    for index, (counts, multiplicity) in enumerate(
        _sorted_profile_items_with_checkpoints(profile)
    ):
        if index % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during symbol-Parikh cell materialization")
        materialized_cells.append(
            SymbolParikhCell(symbol_counts=counts, multiplicity=multiplicity)
        )
    return SymbolParikhProfileResult._from_kernel(
        dfa,
        length,
        cells=tuple(materialized_cells),
        total_accepted_words=total,
    )


def _symbol_parikh_profile_request(
    request: SymbolParikhProfileRequest,
) -> SymbolParikhProfileResult:
    return symbol_parikh_profile(request.dfa, request.word_length)


def _bounded_dfa_tuple(
    value: object,
    limit: int,
    item_fields: tuple[str, ...] | None = None,
) -> tuple[object, ...]:
    """Materialize a DFA container only when its bounded size is known first.

    ``__len__`` is not trusted: at most ``limit + 1`` items are consumed, so a
    lying or infinite iterable cannot allocate past the carrier bound.  When
    ``item_fields`` is given, already-constructed nested models are rewritten
    as plain field maps so strict validation revalidates them instead of
    accepting a validation-bypassed instance.
    """

    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise PydanticCustomError(
            "regular_language.symbol_parikh.dfa_contract",
            "dfa containers must be bounded tuples",
        )
    if type(value) is tuple:
        if len(value) > limit:
            raise PydanticCustomError(
                "regular_language.symbol_parikh.dfa_contract",
                "dfa container exceeds its admitted length",
            )
        items = list(value)
    else:
        if getattr(value, "__len__", None) is None:
            raise PydanticCustomError(
                "regular_language.symbol_parikh.dfa_contract",
                "dfa containers must be bounded tuples",
            )
        try:
            items = list(islice(cast(Iterable[object], value), limit + 1))
        except TypeError:
            raise PydanticCustomError(
                "regular_language.symbol_parikh.dfa_contract",
                "dfa containers must be bounded tuples",
            ) from None
        if len(items) > limit:
            raise PydanticCustomError(
                "regular_language.symbol_parikh.dfa_contract",
                "dfa container exceeds its admitted length",
            )
    if item_fields is None:
        return tuple(items)
    return tuple(
        item
        if isinstance(item, dict)
        else {field: getattr(item, field, None) for field in item_fields}
        for item in items
    )


def _sorted_profile_items_with_checkpoints(
    profile: dict[tuple[int, ...], int],
) -> list[tuple[tuple[int, ...], int]]:
    """Return the profile items in canonical order without a long frozen sort.

    ``sorted`` evaluates its whole argument before the caller can checkpoint,
    so materialize through a heap and checkpoint throughout the pops.
    """

    request_checkpoint("before symbol-Parikh cell ordering")
    heap = list(profile.items())
    heapq.heapify(heap)
    ordered: list[tuple[tuple[int, ...], int]] = []
    while heap:
        ordered.append(heapq.heappop(heap))
        if len(ordered) % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during symbol-Parikh cell ordering")
    return ordered


def symbol_parikh_profile(dfa: DFA, word_length: int) -> SymbolParikhProfileResult:
    """Return the accepted-word symbol Parikh profile for one exact length."""

    if not isinstance(dfa, DFA):
        raise OperationDomainValidationError(
            location=("dfa",),
            code="regular_language.symbol_parikh.dfa_type",
            message="dfa must be a canonical DFA value",
        )
    try:
        dfa = DFA.model_validate(
            {
                "state_count": getattr(dfa, "state_count", None),
                "alphabet_size": getattr(dfa, "alphabet_size", None),
                "transitions": _bounded_dfa_tuple(
                    getattr(dfa, "transitions", None),
                    MAX_DFA_TRANSITIONS,
                    item_fields=("source", "symbol", "target"),
                ),
                "initial_state": getattr(dfa, "initial_state", None),
                "accepting_states": _bounded_dfa_tuple(
                    getattr(dfa, "accepting_states", None), MAX_DFA_STATES
                ),
            }
        )
    except (ValidationError, PydanticCustomError) as exc:
        raise OperationDomainValidationError(
            location=("dfa",),
            code="regular_language.symbol_parikh.dfa_contract",
            message="dfa must be a total deterministic automaton",
        ) from exc
    if (
        type(word_length) is not int
        or word_length < 0
        or word_length > MAX_SYMBOL_PARIKH_LENGTH
    ):
        raise OperationDomainValidationError(
            location=("word_length",),
            code="regular_language.symbol_parikh.word_length",
            message=(
                "word_length must be an integer from 0 through "
                f"{MAX_SYMBOL_PARIKH_LENGTH}"
            ),
        )
    return _compute_symbol_parikh_profile(dfa, word_length)
