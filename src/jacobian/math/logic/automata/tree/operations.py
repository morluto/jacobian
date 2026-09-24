"""Domain-owned bottom-up tree automaton kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from itertools import product
from math import prod
from typing import Literal

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountResult,
    RankedTreePositionsResult,
    RankedTreeSubtreeResult,
    TreeAutomatonBooleanProductResult,
    TreeAutomatonComplementResult,
    TreeAutomatonCompletionResult,
    TreeAutomatonMinimizeResult,
    TreeAutomatonTrimResult,
    TreeContextPlugRequest,
    TreeContextPlugResult,
    TreeContextStateMapRequest,
    TreeContextStateMapResult,
    TreeDeterminizeResult,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
    _plug_tree_context,
    _tree_context_state_map,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_RUN_TREE_DEPTH,
    MAX_RUN_TREE_NODES,
    MAX_TA_ARITY,
    MAX_TA_STATES,
    MAX_TA_SYMBOLS,
    MAX_TA_TRANSITIONS,
    MAX_TREE_AUTOMATON_WORK,
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    ReachableStateProfile,
    TreeAutomatonTransition,
    TreeStateChartEntry,
    _build_reachable_state_profile,
    _reject_tree,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "ReachableStateProfile",
    "accepted_tree_count",
    "boolean_product_tree_automata",
    "complement_tree_automaton",
    "complete_deterministic_tree_automaton",
    "determinize_tree_automaton",
    "map_tree_context_states",
    "minimize_tree_automaton",
    "plug_tree_context_operation",
    "ranked_tree_positions",
    "ranked_tree_subtree",
    "reachable_state_profile",
    "run_tree_automaton",
    "tree_state_chart",
    "trim_tree_automaton",
    "verify_accepted_tree_count",
    "verify_determinization",
    "verify_reachable_state_profile",
    "verify_tree_run",
    "verify_trim_tree_automaton",
]


def _preflight_context(context: FiniteTreeContext) -> tuple[int, int]:
    """Bound typed context structure before serialization or value validation."""
    arity = getattr(context, "arity", None)
    frames = getattr(context, "frames", None)
    if type(arity) is not tuple or len(arity) > MAX_TA_SYMBOLS:
        raise OperationDomainValidationError(
            location=("context", "arity"),
            code="tree_context.input_signature",
            message="context arity must be a bounded tuple",
        )
    if any(type(rank) is not int or not 0 <= rank <= MAX_TA_ARITY for rank in arity):
        raise OperationDomainValidationError(
            location=("context", "arity"),
            code="tree_context.input_signature",
            message="context arities must be integers in the supported range",
        )
    if type(frames) is not tuple:
        raise OperationDomainValidationError(
            location=("context", "frames"),
            code="tree_context.input_frames",
            message="context frames must be a tuple",
        )
    if len(frames) > MAX_RUN_TREE_DEPTH:
        raise OperationResourceAdmissionError(
            location=("context", "frames"),
            code="tree_context.depth_bound",
            message="context spine exceeds the supported depth",
        )
    nodes = len(frames)
    depth_bound = len(frames)
    for frame_index, frame in enumerate(frames):
        if type(frame) is not TreeContextFrame:
            raise OperationDomainValidationError(
                location=("context", "frames", frame_index),
                code="tree_context.input_frame",
                message="context frames must be canonical frame values",
            )
        symbol = getattr(frame, "symbol", None)
        hole_child = getattr(frame, "hole_child", None)
        siblings = getattr(frame, "siblings", None)
        if (
            type(symbol) is not int
            or not 0 <= symbol < len(arity)
            or type(hole_child) is not int
            or type(siblings) is not tuple
        ):
            raise OperationDomainValidationError(
                location=("context", "frames", frame_index),
                code="tree_context.input_frame",
                message="context frame fields must be canonical",
            )
        rank = arity[symbol]
        if not 0 <= hole_child < rank or len(siblings) != rank - 1:
            raise OperationDomainValidationError(
                location=("context", "frames", frame_index),
                code="tree_context.frame_rank",
                message="context frame children do not match the symbol arity",
            )
        for sibling_index, sibling in enumerate(siblings):
            stack = [(sibling, frame_index + 2)]
            while stack:
                node, depth = stack.pop()
                nodes += 1
                if nodes > MAX_RUN_TREE_NODES:
                    raise OperationResourceAdmissionError(
                        location=(
                            "context",
                            "frames",
                            frame_index,
                            "siblings",
                            sibling_index,
                        ),
                        code="tree_context.node_bound",
                        message="context size exceeds the supported node bound",
                    )
                if depth > MAX_RUN_TREE_DEPTH:
                    raise OperationResourceAdmissionError(
                        location=(
                            "context",
                            "frames",
                            frame_index,
                            "siblings",
                            sibling_index,
                        ),
                        code="tree_context.depth_bound",
                        message="context exceeds the supported tree depth",
                    )
                depth_bound = max(depth_bound, depth)
                if (
                    type(node) is not RankedTree
                    or type(getattr(node, "symbol", None)) is not int
                    or type(getattr(node, "children", None)) is not tuple
                    or not 0 <= node.symbol < len(arity)
                    or len(node.children) != arity[node.symbol]
                ):
                    raise OperationDomainValidationError(
                        location=(
                            "context",
                            "frames",
                            frame_index,
                            "siblings",
                            sibling_index,
                        ),
                        code="tree_context.sibling_shape",
                        message="context siblings must be well-ranked canonical trees",
                    )
                stack.extend((child, depth + 1) for child in node.children)
    return nodes, depth_bound


def _preflight_tree(tree: RankedTree) -> tuple[int, int]:
    """Bound a ranked tree before serialization or alphabet traversal."""
    nodes = 0
    depth_bound = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > MAX_RUN_TREE_NODES:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_context.input_node_bound",
                message="input tree exceeds the supported node bound",
            )
        if depth > MAX_RUN_TREE_DEPTH:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_context.input_depth_bound",
                message="input tree exceeds the supported depth",
            )
        depth_bound = max(depth_bound, depth)
        symbol = getattr(node, "symbol", None)
        children = getattr(node, "children", None)
        if (
            type(node) is not RankedTree
            or type(symbol) is not int
            or not 0 <= symbol < MAX_TA_SYMBOLS
            or type(children) is not tuple
            or len(children) > MAX_TA_ARITY
        ):
            raise OperationDomainValidationError(
                location=("tree",),
                code="tree_context.input_tree_shape",
                message="input tree must use canonical ranked-tree nodes",
            )
        stack.extend((child, depth + 1) for child in children)
    return nodes, depth_bound


def _preflight_complete_automaton(
    automaton: CompleteDeterministicBottomUpTreeAutomaton,
) -> None:
    state_count = getattr(automaton, "state_count", None)
    arity = getattr(automaton, "arity", None)
    transitions = getattr(automaton, "transitions", None)
    final_states = getattr(automaton, "final_states", None)
    if type(transitions) is tuple and len(transitions) > MAX_TA_TRANSITIONS:
        raise OperationResourceAdmissionError(
            location=("automaton", "transitions"),
            code="tree_context.automaton_transition_bound",
            message="automaton transition rows exceed the supported bound",
        )
    if (
        type(state_count) is not int
        or not 1 <= state_count <= MAX_TA_STATES
        or type(arity) is not tuple
        or len(arity) > MAX_TA_SYMBOLS
        or type(transitions) is not tuple
        or len(transitions) > MAX_TA_TRANSITIONS
        or type(final_states) is not tuple
        or len(final_states) > MAX_TA_STATES
    ):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_context.automaton_shape",
            message="automaton axes and rows must satisfy their bounded shapes",
        )
    if any(type(rank) is not int or not 0 <= rank <= MAX_TA_ARITY for rank in arity):
        raise OperationDomainValidationError(
            location=("automaton", "arity"),
            code="tree_context.automaton_signature",
            message="automaton arities must be integers in the supported range",
        )
    for transition in transitions:
        if (
            type(transition) is not TreeAutomatonTransition
            or type(getattr(transition, "symbol", None)) is not int
            or type(getattr(transition, "child_states", None)) is not tuple
            or type(getattr(transition, "target_state", None)) is not int
            or len(transition.child_states) > MAX_TA_ARITY
        ):
            raise OperationDomainValidationError(
                location=("automaton", "transitions"),
                code="tree_context.automaton_transition_shape",
                message="automaton transitions must be canonical bounded rows",
            )


def plug_tree_context_operation(
    request: TreeContextPlugRequest,
) -> TreeContextPlugResult:
    """Plug a ranked tree into a canonical one-hole ranked-tree context."""
    if type(request) is not TreeContextPlugRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.plug.request_type",
            message="request must be a canonical tree-context plug request",
        )
    if (
        type(request.context) is not FiniteTreeContext
        or type(request.tree) is not RankedTree
    ):
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.plug.input_type",
            message="context and tree must be canonical ranked-tree values",
        )
    tree_nodes, tree_depth = _preflight_tree(request.tree)
    context_nodes, context_depth = _preflight_context(request.context)
    if context_nodes + tree_nodes > MAX_RUN_TREE_NODES:
        raise OperationResourceAdmissionError(
            location=("context",),
            code="tree_context.plug.node_bound",
            message="plugged tree would exceed the supported node bound",
        )
    if context_depth + tree_depth > MAX_RUN_TREE_DEPTH:
        raise OperationResourceAdmissionError(
            location=("context",),
            code="tree_context.plug.depth_bound",
            message="plugged tree would exceed the supported depth bound",
        )
    try:
        context = FiniteTreeContext.model_validate(
            request.context.model_dump(mode="json")
        )
        tree = RankedTree.model_validate(request.tree.model_dump(mode="json"))
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.plug.invalid_value",
            message="context and tree must satisfy their canonical value contracts",
        ) from exc
    admitted_request = TreeContextPlugRequest.model_construct(
        context=context, tree=tree
    )
    return TreeContextPlugResult._from_kernel(
        admitted_request,
        plugged_tree=_plug_tree_context(
            admitted_request.context, admitted_request.tree
        ),
    )


def map_tree_context_states(
    request: TreeContextStateMapRequest,
) -> TreeContextStateMapResult:
    """Evaluate the context-induced state transformation of a complete DTA."""
    if type(request) is not TreeContextStateMapRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.state_map.request_type",
            message="request must be a canonical tree-context state-map request",
        )
    if (
        type(request.automaton) is not CompleteDeterministicBottomUpTreeAutomaton
        or type(request.context) is not FiniteTreeContext
    ):
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.state_map.input_type",
            message="automaton and context must be canonical typed values",
        )
    _preflight_complete_automaton(request.automaton)
    context_nodes, _ = _preflight_context(request.context)
    state_context_work = (
        len(request.automaton.transitions)
        + context_nodes
        + len(request.context.frames) * request.automaton.state_count
    )
    if state_context_work > MAX_TREE_AUTOMATON_WORK:
        raise OperationResourceAdmissionError(
            location=("context",),
            code="tree_context.state_map.work_bound",
            message="context state-map evaluation exceeds its work bound",
        )
    try:
        automaton = CompleteDeterministicBottomUpTreeAutomaton.model_validate(
            request.automaton.model_dump(mode="json")
        )
        context = FiniteTreeContext.model_validate(
            request.context.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="tree_context.state_map.invalid_value",
            message="automaton and context must satisfy their canonical value contracts",
        ) from exc
    admitted_request = TreeContextStateMapRequest.model_construct(
        automaton=automaton, context=context
    )
    return TreeContextStateMapResult._from_kernel(
        admitted_request,
        state_map=_tree_context_state_map(
            admitted_request.automaton, admitted_request.context
        ),
    )


def _complete_deterministic_rows(
    machine: CompleteDeterministicBottomUpTreeAutomaton, side: str
) -> dict[tuple[int, tuple[int, ...]], int]:
    table = {}
    for row in machine.transitions:
        key = (row.symbol, row.child_states)
        if key in table:
            raise OperationDomainValidationError(
                location=(side, "transitions"),
                code="tree_automata.product_nondeterministic",
                message="Boolean products require deterministic input automata",
            )
        table[key] = row.target_state
    required = sum(machine.state_count**rank for rank in machine.arity)
    if len(table) != required:
        raise OperationDomainValidationError(
            location=(side, "transitions"),
            code="tree_automata.product_incomplete",
            message="Boolean products require complete deterministic input automata",
        )
    return table


def _boolean_final(connective: str, left_final: bool, right_final: bool) -> bool:
    if connective == "intersection":
        return left_final and right_final
    if connective == "union":
        return left_final or right_final
    if connective == "difference":
        return left_final and not right_final
    return left_final != right_final


def ranked_tree_positions(
    tree: RankedTree,
) -> RankedTreePositionsResult:
    """Return all node positions as zero-based child-index paths in preorder."""

    if type(tree) is not RankedTree:
        raise OperationDomainValidationError(
            location=("tree",),
            code="tree_automata.positions.tree_type",
            message="tree must be a canonical RankedTree value",
        )

    # The first pass bounds the input structure and prices every path cell
    # without retaining result paths. Only after that plan fits the work and
    # allocation bounds do we materialize the complete position tuple.
    node_count = 0
    coordinate_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if (
            type(node) is not RankedTree
            or type(node.symbol) is not int
            or not 0 <= node.symbol < MAX_TA_SYMBOLS
            or type(node.children) is not tuple
            or len(node.children) > MAX_TA_ARITY
        ):
            raise OperationDomainValidationError(
                location=("tree",),
                code="tree_automata.positions.tree_shape",
                message="tree nodes must satisfy the canonical ranked-tree shape",
            )
        node_count += 1
        if node_count > MAX_RUN_TREE_NODES:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_automata.positions.node_count_bound",
                message="ranked tree node count exceeds the position-operation bound",
            )
        if depth > MAX_RUN_TREE_DEPTH:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_automata.positions.depth_bound",
                message="ranked tree depth exceeds the position-operation bound",
            )
        coordinate_count += depth - 1
        stack.extend((child, depth + 1) for child in node.children)

    work_bound = 2 * node_count + coordinate_count
    if work_bound > MAX_RANKED_TREE_POSITIONS_WORK:
        raise OperationResourceAdmissionError(
            location=("tree",),
            code="tree_automata.positions.work_bound",
            message="ranked tree position traversal exceeds its work bound",
        )
    result_cells = node_count + coordinate_count
    if result_cells > MAX_RANKED_TREE_POSITIONS_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("tree",),
            code="tree_automata.positions.result_cells_bound",
            message="complete ranked-tree positions exceed the result allocation bound",
        )

    positions: list[tuple[int, ...]] = []
    position_stack: list[tuple[RankedTree, tuple[int, ...]]] = [(tree, ())]
    while position_stack:
        node, position = position_stack.pop()
        positions.append(position)
        for child_index in range(len(node.children) - 1, -1, -1):
            position_stack.append(
                (node.children[child_index], (*position, child_index))
            )
        if len(positions) % 512 == 0:
            request_checkpoint("during ranked-tree position materialization")

    return RankedTreePositionsResult._from_kernel(tree=tree, positions=tuple(positions))


def ranked_tree_subtree(
    tree: RankedTree,
    position: tuple[int, ...],
) -> RankedTreeSubtreeResult:
    """Return the source-bound subtree rooted at a child-index position."""

    if type(tree) is not RankedTree or type(position) is not tuple:
        raise OperationDomainValidationError(
            location=("tree",),
            code="tree_automata.subtree.input_type",
            message="tree and position must be canonical ranked-tree values",
        )

    node_count = 0
    stack: list[tuple[RankedTree, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if (
            type(node) is not RankedTree
            or type(node.symbol) is not int
            or not 0 <= node.symbol < MAX_TA_SYMBOLS
            or type(node.children) is not tuple
            or len(node.children) > MAX_TA_ARITY
        ):
            raise OperationDomainValidationError(
                location=("tree",),
                code="tree_automata.subtree.tree_shape",
                message="tree nodes must satisfy the canonical ranked-tree shape",
            )
        node_count += 1
        if node_count > MAX_RUN_TREE_NODES:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_automata.subtree.node_count_bound",
                message="ranked tree exceeds the subtree-operation node bound",
            )
        if depth > MAX_RUN_TREE_DEPTH:
            raise OperationResourceAdmissionError(
                location=("tree",),
                code="tree_automata.subtree.depth_bound",
                message="ranked tree exceeds the subtree-operation depth bound",
            )
        stack.extend((child, depth + 1) for child in node.children)

    current = tree
    for depth, index in enumerate(position):
        if type(index) is not int or index < 0 or index >= len(current.children):
            raise OperationDomainValidationError(
                location=("position", depth),
                code="tree_automata.subtree.position_missing",
                message="position does not identify a node in the source tree",
            )
        current = current.children[index]

    # The result retains the source tree, selected subtree, and address. Admit
    # by retained nodes and address cells instead of serialized transport size.
    work_bound = 2 * node_count + len(position)
    if work_bound > MAX_RANKED_TREE_POSITIONS_WORK:
        raise OperationResourceAdmissionError(
            location=("tree",),
            code="tree_automata.subtree.work_bound",
            message="ranked-tree subtree selection exceeds its traversal bound",
        )
    if 2 * node_count + len(position) > MAX_RANKED_TREE_SUBTREE_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("tree",),
            code="tree_automata.subtree.result_cells_bound",
            message="source-bound ranked-tree subtree exceeds its result allocation bound",
        )
    return RankedTreeSubtreeResult._from_kernel(
        tree=tree, position=position, subtree=current
    )


def boolean_product_tree_automata(
    left: CompleteDeterministicBottomUpTreeAutomaton,
    right: CompleteDeterministicBottomUpTreeAutomaton,
    connective: Literal["intersection", "union", "difference", "symmetric_difference"],
) -> TreeAutomatonBooleanProductResult:
    """Construct the exact direct product for two complete deterministic machines.

    Completeness makes every product state pair total, so each Boolean language
    connective is represented by the corresponding final-state predicate.
    """
    if left.arity != right.arity:
        raise OperationDomainValidationError(
            location=("right", "arity"),
            code="tree_automata.product_signature",
            message="Boolean products require the same ordered ranked signature",
        )

    left_rows = _complete_deterministic_rows(left, "left")
    right_rows = _complete_deterministic_rows(right, "right")
    pair_count = left.state_count * right.state_count
    if pair_count > MAX_TA_STATES:
        raise OperationResourceAdmissionError(
            location=("left", "state_count"),
            code="tree_automata.product_states_bound",
            message=f"the product has {pair_count} state pairs; at most {MAX_TA_STATES} are admitted",
        )

    # Join by symbol and preflight the full output before constructing rows.
    left_by_symbol = defaultdict(list)
    right_by_symbol = defaultdict(list)
    for (symbol, children), target in left_rows.items():
        left_by_symbol[symbol].append((children, target))
    for (symbol, children), target in right_rows.items():
        right_by_symbol[symbol].append((children, target))
    row_count = sum(
        len(left_by_symbol[symbol]) * len(right_by_symbol[symbol])
        for symbol in range(len(left.arity))
    )
    work = sum(
        len(left_by_symbol[symbol])
        * len(right_by_symbol[symbol])
        * (left.arity[symbol] + 1)
        for symbol in range(len(left.arity))
    )
    if row_count > MAX_TA_TRANSITIONS or work > MAX_TREE_AUTOMATON_WORK:
        raise OperationResourceAdmissionError(
            location=("transitions",),
            code="tree_automata.product_transitions_bound",
            message="the direct product transition table exceeds its admitted output or work bound",
        )

    state_pairs = tuple(
        (a, b) for a in range(left.state_count) for b in range(right.state_count)
    )
    state_id = {pair: index for index, pair in enumerate(state_pairs)}
    left_finals, right_finals = set(left.final_states), set(right.final_states)

    transitions = []
    for symbol in range(len(left.arity)):
        for left_children, left_target in left_by_symbol[symbol]:
            for right_children, right_target in right_by_symbol[symbol]:
                paired_children = tuple(
                    state_id[(a, b)]
                    for a, b in zip(left_children, right_children, strict=True)
                )
                transitions.append(
                    TreeAutomatonTransition(
                        symbol=symbol,
                        child_states=paired_children,
                        target_state=state_id[(left_target, right_target)],
                    )
                )
    product_automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=pair_count,
        arity=left.arity,
        transitions=tuple(
            sorted(transitions, key=lambda row: (row.symbol, row.child_states))
        ),
        final_states=tuple(
            index
            for index, (a, b) in enumerate(state_pairs)
            if _boolean_final(connective, a in left_finals, b in right_finals)
        ),
    )
    return TreeAutomatonBooleanProductResult._from_kernel(
        left=left,
        right=right,
        connective=connective,
        product=product_automaton,
        state_pairs=state_pairs,
    )


MAX_DETERMINIZE_WORK = 500_000
MAX_DETERMINIZE_SAMPLE_TREES = 4096
MAX_COMPLEMENT_OUTPUT_CELLS = (
    2 * MAX_TA_TRANSITIONS * (MAX_TA_ARITY + 1) + MAX_TA_STATES * 4 + MAX_TA_SYMBOLS * 2
)
MAX_COMPLETION_OUTPUT_CELLS = MAX_TA_TRANSITIONS * (MAX_TA_ARITY + 2)
MAX_MINIMIZE_WORK = MAX_TREE_AUTOMATON_WORK
MAX_RANKED_TREE_POSITIONS_WORK = 600_000
MAX_RANKED_TREE_POSITIONS_RESULT_CELLS = MAX_RUN_TREE_NODES * (MAX_RUN_TREE_DEPTH + 1)
MAX_RANKED_TREE_SUBTREE_RESULT_CELLS = 2 * MAX_RUN_TREE_NODES + MAX_RUN_TREE_DEPTH


def _tree_automaton_minimization_partition(
    automaton: DeterministicBottomUpTreeAutomaton,
    reachable_states: tuple[int, ...],
    rows: dict[tuple[int, tuple[int, ...]], int],
) -> dict[int, int]:
    final_states = set(automaton.final_states)
    classes = {state: int(state in final_states) for state in reachable_states}
    while True:
        refined_by_signature: dict[tuple[object, ...], list[int]] = {}
        for state in reachable_states:
            signature: list[object] = [state in final_states, classes[state]]
            for symbol, rank in enumerate(automaton.arity):
                for position in range(rank):
                    other_positions = tuple(
                        index for index in range(rank) if index != position
                    )
                    for other_states in product(reachable_states, repeat=rank - 1):
                        children = [0] * rank
                        children[position] = state
                        for index, other_state in zip(
                            other_positions, other_states, strict=True
                        ):
                            children[index] = other_state
                        target = rows.get((symbol, tuple(children)))
                        signature.append(-1 if target is None else classes[target])
            refined_by_signature.setdefault(tuple(signature), []).append(state)
        blocks = sorted(refined_by_signature.values(), key=min)
        refined = {
            state: block_id for block_id, block in enumerate(blocks) for state in block
        }
        if all(refined[state] == classes[state] for state in reachable_states):
            return refined
        classes = refined


def _tree_automaton_minimized_value(
    automaton: DeterministicBottomUpTreeAutomaton,
    reachable_states: tuple[int, ...],
    classes: dict[int, int],
) -> tuple[DeterministicBottomUpTreeAutomaton, tuple[int, ...], tuple[int, ...]]:
    blocks = tuple(sorted(set(classes.values())))
    class_to_quotient = {block: index for index, block in enumerate(blocks)}
    old_to_new = tuple(
        -1 if state not in classes else class_to_quotient[classes[state]]
        for state in range(automaton.state_count)
    )
    representatives = tuple(
        min(state for state in reachable_states if classes[state] == block)
        for block in blocks
    )
    reachable = set(reachable_states)
    quotient_rows: dict[tuple[int, tuple[int, ...]], int] = {}
    for row in automaton.transitions:
        if row.target_state not in reachable or any(
            child not in reachable for child in row.child_states
        ):
            continue
        key = (row.symbol, tuple(old_to_new[child] for child in row.child_states))
        target = old_to_new[row.target_state]
        previous = quotient_rows.setdefault(key, target)
        if previous != target:
            raise RuntimeError(
                "partition refinement did not produce a transition congruence"
            )
    final_states = set(automaton.final_states)
    minimized = DeterministicBottomUpTreeAutomaton(
        state_count=len(blocks),
        arity=automaton.arity,
        transitions=tuple(
            TreeAutomatonTransition(
                symbol=symbol, child_states=children, target_state=target
            )
            for (symbol, children), target in sorted(quotient_rows.items())
        ),
        final_states=tuple(
            index
            for index, representative in enumerate(representatives)
            if representative in final_states
        ),
    )
    return minimized, old_to_new, representatives


def minimize_tree_automaton(
    automaton: DeterministicBottomUpTreeAutomaton,
) -> TreeAutomatonMinimizeResult:
    """Return the smallest reachable quotient of a deterministic tree automaton.

    Missing transitions remain undefined. States are equivalent exactly when
    every ground-tree context with a hole accepts from both or neither state.
    """

    if not isinstance(automaton, DeterministicBottomUpTreeAutomaton):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.minimize_automaton_type",
            message="minimization requires a deterministic bottom-up automaton",
        )
    try:
        automaton = DeterministicBottomUpTreeAutomaton.model_validate(
            automaton.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.minimize_automaton_shape",
            message="automaton must satisfy its deterministic ranked carrier shape",
        ) from exc
    n, transition_count = automaton.state_count, len(automaton.transitions)
    rows = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }
    has_ground_tree_seed = any(
        not row.child_states and automaton.arity[row.symbol] == 0
        for row in automaton.transitions
    )
    if not has_ground_tree_seed:
        minimized = DeterministicBottomUpTreeAutomaton(
            state_count=1, arity=automaton.arity, transitions=(), final_states=()
        )
        return TreeAutomatonMinimizeResult._from_kernel(
            automaton=automaton,
            minimized=minimized,
            old_to_new=(-1,) * n,
            new_to_old=(None,),
            reachable_states=(),
        )
    maximum_rank = max(automaton.arity, default=0)
    reachability_work = (n + 1) * (transition_count + 1) * (maximum_rank + 1)
    context_cells = sum(rank * n ** (rank - 1) for rank in automaton.arity if rank > 0)
    refinement_work = n * n * context_cells
    if reachability_work + refinement_work > MAX_MINIMIZE_WORK:
        raise OperationResourceAdmissionError(
            location=("automaton",),
            code="tree_automata.minimize_work_bound",
            message="the reachable-state and partition-refinement bound is exceeded",
        )
    if transition_count * (maximum_rank + 1) > MAX_TA_TRANSITIONS * (MAX_TA_ARITY + 1):
        raise OperationResourceAdmissionError(
            location=("automaton", "transitions"),
            code="tree_automata.minimize_output_bound",
            message="the quotient transition output exceeds its admitted cell bound",
        )

    reachable: set[int] = set()
    changed = True
    while changed:
        changed = False
        for row in automaton.transitions:
            if (
                all(state in reachable for state in row.child_states)
                and row.target_state not in reachable
            ):
                reachable.add(row.target_state)
                changed = True
    reachable_states = tuple(sorted(reachable))

    classes = _tree_automaton_minimization_partition(automaton, reachable_states, rows)
    minimized, old_to_new, representatives = _tree_automaton_minimized_value(
        automaton, reachable_states, classes
    )
    return TreeAutomatonMinimizeResult._from_kernel(
        automaton=automaton,
        minimized=minimized,
        old_to_new=old_to_new,
        new_to_old=representatives,
        reachable_states=reachable_states,
    )


def complete_deterministic_tree_automaton(
    automaton: DeterministicBottomUpTreeAutomaton,
) -> TreeAutomatonCompletionResult:
    """Fill missing deterministic transitions with one nonfinal sink state."""

    if not isinstance(automaton, DeterministicBottomUpTreeAutomaton):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.completion_automaton_type",
            message="automaton must be a deterministic bottom-up tree automaton",
        )
    try:
        automaton = DeterministicBottomUpTreeAutomaton.model_validate(
            automaton.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.completion_automaton_shape",
            message="automaton must satisfy its deterministic ranked carrier shape",
        ) from exc

    original_row_count = sum(automaton.state_count**rank for rank in automaton.arity)
    if original_row_count > MAX_TA_TRANSITIONS:
        raise OperationResourceAdmissionError(
            location=("automaton", "arity"),
            code="tree_automata.completion_transition_bound",
            message=(
                "even the source-state transition table exceeds the admitted "
                f"{MAX_TA_TRANSITIONS}-row completion bound"
            ),
        )
    missing_count = original_row_count - len(automaton.transitions)
    if missing_count < 0:
        raise OperationDomainValidationError(
            location=("automaton", "transitions"),
            code="tree_automata.completion_transition_count",
            message="the deterministic transition table exceeds its ranked domain",
        )

    sink_state: int | None = None
    completed_state_count = automaton.state_count
    if missing_count:
        if automaton.state_count >= MAX_TA_STATES:
            raise OperationResourceAdmissionError(
                location=("automaton", "state_count"),
                code="tree_automata.completion_state_bound",
                message="completion needs one sink state beyond the admitted state bound",
            )
        sink_state = automaton.state_count
        completed_state_count += 1

    output_row_count = sum(completed_state_count**rank for rank in automaton.arity)
    output_cells = sum(
        completed_state_count**rank * (rank + 2) for rank in automaton.arity
    )
    work_bound = sum(
        completed_state_count**rank * (rank + 1) for rank in automaton.arity
    )
    if (
        output_row_count > MAX_TA_TRANSITIONS
        or output_cells > MAX_COMPLETION_OUTPUT_CELLS
        or work_bound > MAX_TREE_AUTOMATON_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("automaton", "arity"),
            code="tree_automata.completion_output_bound",
            message=(
                "the completed transition table exceeds its admitted row, "
                "cell, or work envelope"
            ),
        )

    existing = {
        (transition.symbol, transition.child_states): transition.target_state
        for transition in automaton.transitions
    }

    def completed_target(symbol: int, children: tuple[int, ...]) -> int:
        target = existing.get((symbol, children))
        if target is not None:
            return target
        if sink_state is None:
            raise RuntimeError(
                "a complete source automaton is missing an admitted transition"
            )
        return sink_state

    transitions = tuple(
        TreeAutomatonTransition(
            symbol=symbol,
            child_states=children,
            target_state=completed_target(symbol, children),
        )
        for symbol, rank in enumerate(automaton.arity)
        for children in product(range(completed_state_count), repeat=rank)
    )
    completed = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=completed_state_count,
        arity=automaton.arity,
        transitions=transitions,
        final_states=automaton.final_states,
    )
    return TreeAutomatonCompletionResult._from_kernel(
        automaton=automaton,
        completed=completed,
        source_to_completed=tuple(range(automaton.state_count)),
        sink_state=sink_state,
    )


def complement_tree_automaton(
    automaton: CompleteDeterministicBottomUpTreeAutomaton,
) -> TreeAutomatonComplementResult:
    """Complement a complete deterministic bottom-up tree automaton.

    The ranked signature is the explicit ``arity`` tuple: symbol IDs are its
    indices. State labels are canonicalized by first occurrence in the
    lexicographically ordered complete transition table, with absent states
    appended in their source order. The result carries both state maps.
    """
    if not isinstance(automaton, CompleteDeterministicBottomUpTreeAutomaton):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.complement_automaton_type",
            message="automaton must be a complete deterministic tree automaton",
        )
    try:
        automaton = CompleteDeterministicBottomUpTreeAutomaton.model_validate(
            automaton.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.complement_automaton_shape",
            message="automaton must satisfy its complete ranked carrier shape",
        ) from exc

    # This arithmetic is cheap even at the carrier's maximum arity. Refuse
    # before building transition keys or expanding any child-state product.
    transition_count = sum(automaton.state_count**arity for arity in automaton.arity)
    if transition_count > MAX_TA_TRANSITIONS:
        raise OperationResourceAdmissionError(
            location=("automaton", "arity"),
            code="tree_automata.complement_transition_bound",
            message=(
                "the complete transition table exceeds the admitted output "
                f"bound of {MAX_TA_TRANSITIONS} rows"
            ),
        )
    transition_cells = sum(
        automaton.state_count**arity * (arity + 1) for arity in automaton.arity
    )
    output_cells = (
        2 * transition_cells + 4 * automaton.state_count + 2 * len(automaton.arity)
    )
    if output_cells > MAX_COMPLEMENT_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("automaton", "arity"),
            code="tree_automata.complement_output_bound",
            message="the complete complement result exceeds its admitted output-cell bound",
        )

    keyed: dict[tuple[int, tuple[int, ...]], int] = {}
    for transition in automaton.transitions:
        key = (transition.symbol, transition.child_states)
        if key in keyed:
            raise OperationDomainValidationError(
                location=("automaton", "transitions"),
                code="tree_automata.complement_nondeterministic",
                message=(
                    "complement requires exactly one target per symbol and "
                    "ordered child-state tuple"
                ),
            )
        keyed[key] = transition.target_state
    if len(keyed) != transition_count:
        raise OperationDomainValidationError(
            location=("automaton", "transitions"),
            code="tree_automata.complement_incomplete",
            message=(
                "complement requires one transition for every symbol and "
                "ordered child-state tuple"
            ),
        )

    canonical_rows = sorted(
        automaton.transitions,
        key=lambda row: (row.symbol, row.child_states, row.target_state),
    )
    state_order: list[int] = []
    seen: set[int] = set()
    for row in canonical_rows:
        for state in (*row.child_states, row.target_state):
            if state not in seen:
                seen.add(state)
                state_order.append(state)
    state_order.extend(
        state for state in range(automaton.state_count) if state not in seen
    )
    old_to_new_list = [0] * automaton.state_count
    for new, old in enumerate(state_order):
        old_to_new_list[old] = new
    old_to_new = tuple(old_to_new_list)
    new_to_old = tuple(state_order)
    source_finals = set(automaton.final_states)
    complement = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=automaton.state_count,
        arity=automaton.arity,
        transitions=tuple(
            sorted(
                (
                    TreeAutomatonTransition(
                        symbol=row.symbol,
                        child_states=tuple(
                            old_to_new[state] for state in row.child_states
                        ),
                        target_state=old_to_new[row.target_state],
                    )
                    for row in canonical_rows
                ),
                key=lambda row: (row.symbol, row.child_states, row.target_state),
            )
        ),
        final_states=tuple(
            state
            for state in range(automaton.state_count)
            if new_to_old[state] not in source_finals
        ),
    )
    return TreeAutomatonComplementResult._from_kernel(
        automaton=automaton,
        complement=complement,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        transition_count=transition_count,
    )


def reachable_state_profile(
    automaton: BottomUpTreeAutomaton,
) -> ReachableStateProfile:
    """Return each reachable state and its canonical minimum-node witness tree."""

    return _build_reachable_state_profile(automaton)


def _productive_states(automaton: BottomUpTreeAutomaton) -> set[int]:
    """Return every state occurring inside some accepting run.

    Backward least fixed point seeded with the final states: a transition
    whose target is productive makes all of its child states productive. The
    pass prices the same saturation it runs and shares the tree-automaton
    work envelope with the reachability admission.
    """

    maximum_arity = max(
        (len(row.child_states) for row in automaton.transitions), default=0
    )
    rounds = automaton.state_count + 1
    if rounds * len(automaton.transitions) * (maximum_arity + 1) > (
        MAX_TREE_AUTOMATON_WORK
    ):
        _reject_tree("tree automaton productivity work bound exceeded")
    useful = set(automaton.final_states)
    for _ in range(rounds):
        grown = set(useful)
        for transition in automaton.transitions:
            if transition.target_state in useful:
                grown.update(transition.child_states)
        if grown == useful:
            return useful
        useful = grown
    raise RuntimeError("tree automaton productivity did not reach a fixed point")


def trim_tree_automaton(
    automaton: BottomUpTreeAutomaton,
) -> TreeAutomatonTrimResult:
    """Restrict an automaton to its reachable and productive states.

    Every accepting run of the source uses reachable productive states only,
    so restriction preserves the accepted language exactly. The empty
    language trims to the canonical one-state automaton with no transitions
    and no final states. Witnesses are replayed on the trimmed automaton, so
    each uses kept states only.
    """

    profile = _build_reachable_state_profile(automaton)
    kept = tuple(sorted(set(profile.reachable_states) & _productive_states(automaton)))
    kept_set = set(kept)
    state_to_new = {state: index for index, state in enumerate(kept)}
    dropped = tuple(
        state for state in range(automaton.state_count) if state not in kept_set
    )
    old_to_new = tuple(
        state_to_new.get(state, -1) for state in range(automaton.state_count)
    )
    if not kept:
        trimmed = BottomUpTreeAutomaton(
            state_count=1,
            arity=automaton.arity,
            transitions=(),
            final_states=(),
        )
        return TreeAutomatonTrimResult._from_kernel(
            automaton=automaton,
            trimmed=trimmed,
            kept_states=(),
            dropped_states=dropped,
            old_to_new=old_to_new,
            new_to_old=(),
            empty_language=True,
            witnesses=(),
        )
    kept_set = set(kept)
    remap = {old: new for new, old in enumerate(kept)}
    trimmed = BottomUpTreeAutomaton(
        state_count=len(kept),
        arity=automaton.arity,
        transitions=tuple(
            sorted(
                (
                    TreeAutomatonTransition(
                        symbol=transition.symbol,
                        child_states=tuple(
                            remap[state] for state in transition.child_states
                        ),
                        target_state=remap[transition.target_state],
                    )
                    for transition in automaton.transitions
                    if transition.target_state in kept_set
                    and all(state in kept_set for state in transition.child_states)
                ),
                key=lambda row: (row.symbol, row.child_states, row.target_state),
            )
        ),
        final_states=tuple(
            sorted(
                remap[state] for state in automaton.final_states if state in kept_set
            )
        ),
    )
    replay = _build_reachable_state_profile(trimmed)
    if set(replay.reachable_states) != set(range(len(kept))):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.trim_replay_mismatch",
            message="a kept state is unreachable in the trimmed automaton",
        )
    if not trimmed.final_states:
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.trim_replay_mismatch",
            message="a nonempty trim must retain a final state",
        )
    return TreeAutomatonTrimResult._from_kernel(
        automaton=automaton,
        trimmed=trimmed,
        kept_states=kept,
        dropped_states=dropped,
        old_to_new=old_to_new,
        new_to_old=kept,
        empty_language=not trimmed.final_states,
        witnesses=replay.witnesses,
    )


def verify_trim_tree_automaton(claim: TreeAutomatonTrimResult) -> bool:
    """Verify a trim against its retained source automaton."""

    try:
        return trim_tree_automaton(claim.automaton) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def run_tree_automaton(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> set[int]:
    """Run a bottom-up tree automaton and return the reachable root states."""

    return set(tree_state_chart(automaton, tree)[-1][1])


def tree_state_chart(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return the canonical postorder position/state chart for a ranked tree."""

    validate_ranked_tree(automaton, tree)
    return _tree_state_chart_unchecked(automaton, tree)


def _tree_state_chart_unchecked(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Build a chart after the owner operation has validated the tree."""

    chart: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def visit(node: RankedTree, position: tuple[int, ...]) -> set[int]:
        child_states = tuple(
            visit(child, (*position, index))
            for index, child in enumerate(node.children)
        )
        states = {
            transition.target_state
            for transition in automaton.transitions
            if transition.symbol == node.symbol
            and len(transition.child_states) == len(child_states)
            and all(
                transition.child_states[index] in states
                for index, states in enumerate(child_states)
            )
        }
        chart.append((position, tuple(sorted(states))))
        return states

    visit(tree, ())
    return tuple(chart)


def accepted_tree_count(
    automaton: BottomUpTreeAutomaton,
    tree_size: int,
) -> int:
    """Count distinct accepted ranked trees, not accepting runs."""

    if type(tree_size) is not int:
        _reject_tree("tree size must be an integer", resource=False)
    if tree_size < 1:
        return 0
    accepted_tree_count_work_bound(automaton, tree_size)
    return _accepted_tree_count_admitted(automaton, tree_size)


def _accepted_tree_count_admitted(
    automaton: BottomUpTreeAutomaton, tree_size: int
) -> int:
    if not any(transition.child_states for transition in automaton.transitions):
        # Only leaves can have a run. Count symbols, not nondeterministic runs.
        if tree_size != 1:
            return 0
        finals = set(automaton.final_states)
        return len(
            {
                transition.symbol
                for transition in automaton.transitions
                if transition.target_state in finals
            }
        )
    transitions_by_symbol = {
        symbol: tuple(
            transition
            for transition in automaton.transitions
            if transition.symbol == symbol
        )
        for symbol in range(len(automaton.arity))
    }
    counts_by_size: list[dict[int, int]] = [{} for _ in range(tree_size + 1)]
    for size in range(1, tree_size + 1):
        size_counts: defaultdict[int, int] = defaultdict(int)
        for symbol, arity in enumerate(automaton.arity):
            _accumulate_symbol_trees(
                arity=arity,
                size=size,
                transitions=transitions_by_symbol[symbol],
                counts_by_size=counts_by_size,
                size_counts=size_counts,
            )
        counts_by_size[size] = dict(size_counts)

    final_mask = sum(1 << state for state in automaton.final_states)
    return sum(
        count
        for state_subset, count in counts_by_size[tree_size].items()
        if state_subset & final_mask
    )


def _accumulate_symbol_trees(
    *,
    arity: int,
    size: int,
    transitions: tuple[TreeAutomatonTransition, ...],
    counts_by_size: list[dict[int, int]],
    size_counts: defaultdict[int, int],
) -> None:
    if not transitions:
        return
    if arity == 0:
        if size == 1:
            root_subset = _target_subset(transitions, ())
            if root_subset:
                size_counts[root_subset] += 1
        return
    for child_sizes in _positive_compositions(size - 1, arity):
        if any(not counts_by_size[value] for value in child_sizes):
            continue
        child_choices = [counts_by_size[value].items() for value in child_sizes]
        for child_items in product(*child_choices):
            child_subsets = tuple(item[0] for item in child_items)
            root_subset = _target_subset(transitions, child_subsets)
            if root_subset:
                size_counts[root_subset] += prod(item[1] for item in child_items)


def _target_subset(
    transitions: tuple[TreeAutomatonTransition, ...],
    child_subsets: tuple[int, ...],
) -> int:
    target_subset = 0
    for transition in transitions:
        if all(
            child_subsets[index] & (1 << child_state)
            for index, child_state in enumerate(transition.child_states)
        ):
            target_subset |= 1 << transition.target_state
    return target_subset


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for remainder in _positive_compositions(total - first, parts - 1):
            yield (first, *remainder)


def verify_tree_run(claim: TreeRunResult) -> bool:
    """Verify a serialized run chart and root claim against its sources."""

    try:
        chart = tree_state_chart(claim.automaton, claim.tree)
        roots = chart[-1][1]
        accepted = bool(set(roots) & set(claim.automaton.final_states))
        typed_chart = tuple(
            TreeStateChartEntry(position=position, states=states)
            for position, states in chart
        )
        return (
            claim.state_chart == typed_chart
            and claim.root_states == roots
            and claim.accepted == accepted
            and claim.node_count == len(chart)
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_reachable_state_profile(claim: ReachableStateProfile) -> bool:
    """Verify reachable states and each claimed witness against the automaton."""

    try:
        return reachable_state_profile(claim.automaton) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_accepted_tree_count(claim: AcceptedTreeCountResult) -> bool:
    """Verify an accepted-tree count against its bounded source automaton."""

    try:
        return accepted_tree_count(claim.automaton, claim.tree_size) == claim.count
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


class _DeterminizeBudgetError(Exception):
    """Internal signal that subset construction exceeded its budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _determinize_images(
    automaton: BottomUpTreeAutomaton,
) -> dict[tuple[int, tuple[int, ...]], int]:
    """Map each source transition row to its target-state bitmask."""

    images: dict[tuple[int, tuple[int, ...]], int] = {}
    for transition in automaton.transitions:
        key = (transition.symbol, transition.child_states)
        images[key] = images.get(key, 0) | (1 << transition.target_state)
    return images


def _image_of_subsets(
    images: dict[tuple[int, tuple[int, ...]], int],
    state_count: int,
    symbol: int,
    children: tuple[tuple[int, ...], ...],
    work: list[int],
) -> tuple[int, ...]:
    """Return the sorted source-state image of one subset tuple.

    ``work`` is a single-cell budget ledger: every elementary source-state
    combination consumes one unit and raises ``_DeterminizeBudgetError`` past
    the work envelope.
    """

    mask = 0
    combos: list[tuple[int, ...]] = [()]
    for subset in children:
        work[0] += max(1, len(combos)) * max(1, len(subset))
        if work[0] > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
        combos = [(*prefix, state) for prefix in combos for state in subset]
        if len(combos) > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
    work[0] += max(1, len(combos))
    if work[0] > MAX_DETERMINIZE_WORK:
        raise _DeterminizeBudgetError("WORK_BUDGET")
    for combo in combos:
        mask |= images.get((symbol, combo), 0)
    return tuple(state for state in range(state_count) if mask & (1 << state))


def _combos_for_expansion(subsets: int, arity: tuple[int, ...]) -> int:
    """Count fresh subset tuples when the newest subset has an index.

    Expanding subset ``subsets - 1`` evaluates exactly the tuples whose
    maximum child index equals it; count them, stopping past the budget.
    """

    total = 0
    for rank in arity:
        if rank == 0:
            continue
        fresh = 1
        for _ in range(rank):
            fresh *= subsets
            if fresh > MAX_DETERMINIZE_WORK:
                return MAX_DETERMINIZE_WORK + 1
        old = 1
        for _ in range(rank):
            old *= subsets - 1
            if old > MAX_DETERMINIZE_WORK:
                break
        total += fresh - old
        if total > MAX_DETERMINIZE_WORK:
            return MAX_DETERMINIZE_WORK + 1
    return total


def _seed_nullary_subsets(
    automaton: BottomUpTreeAutomaton,
    images: dict[tuple[int, tuple[int, ...]], int],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    rows: dict[tuple[int, tuple[int, ...]], int],
    combos: list[int],
    work: list[int],
    max_subset_states: int,
) -> None:
    """Intern the image of every nullary symbol before worklist expansion."""

    for symbol, rank in enumerate(automaton.arity):
        if rank != 0:
            continue
        combos[0] += 1
        work[0] += 1
        if work[0] > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
        image = _image_of_subsets(images, automaton.state_count, symbol, (), work)
        target = _intern_subset(image, subsets, index_of, max_subset_states)
        if target is not None:
            rows[(symbol, ())] = target


def _intern_subset(
    image: tuple[int, ...],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    max_subset_states: int,
) -> int | None:
    """Intern one nonempty image, signalling the state budget when full."""

    if not image:
        return None
    if image not in index_of:
        if len(subsets) >= max_subset_states:
            raise _DeterminizeBudgetError("STATE_BUDGET")
        index_of[image] = len(subsets)
        subsets.append(image)
    return index_of[image]


def _expand_newest_subset(
    automaton: BottomUpTreeAutomaton,
    images: dict[tuple[int, tuple[int, ...]], int],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    rows: dict[tuple[int, tuple[int, ...]], int],
    combos: list[int],
    work: list[int],
    newest: int,
    expanded: int,
    max_subset_states: int,
) -> None:
    """Evaluate every subset tuple whose maximum child index is ``newest``."""

    if _combos_for_expansion(expanded, automaton.arity) > (
        MAX_DETERMINIZE_WORK - work[0]
    ):
        raise _DeterminizeBudgetError("WORK_BUDGET")
    for symbol, rank in enumerate(automaton.arity):
        if rank == 0:
            continue
        for children in product(range(expanded), repeat=rank):
            if max(children) != newest:
                continue
            combos[0] += 1
            image = _image_of_subsets(
                images,
                automaton.state_count,
                symbol,
                tuple(subsets[child] for child in children),
                work,
            )
            target = _intern_subset(image, subsets, index_of, max_subset_states)
            if target is not None:
                key = (symbol, children)
                if key not in rows and len(rows) == 4096:
                    raise _DeterminizeBudgetError("WORK_BUDGET")
                rows[key] = target


def _subset_construction(
    automaton: BottomUpTreeAutomaton,
    max_subset_states: int,
) -> tuple[
    list[tuple[int, ...]], dict[tuple[int, tuple[int, ...]], int], int, str | None
]:
    """Run bounded subset construction, returning subsets and DFA rows.

    Returns ``(subsets, rows, combos_evaluated, truncation_reason)`` where
    rows map ``(symbol, child DFA indices)`` to target DFA indices over the
    discovery-ordered subset list. ``truncation_reason`` is ``None`` exactly
    when every subset tuple over the final subset list was evaluated.
    """

    images = _determinize_images(automaton)
    subsets: list[tuple[int, ...]] = []
    index_of: dict[tuple[int, ...], int] = {}
    rows: dict[tuple[int, tuple[int, ...]], int] = {}
    combos = [0]
    work = [0]
    try:
        _seed_nullary_subsets(
            automaton,
            images,
            subsets,
            index_of,
            rows,
            combos,
            work,
            max_subset_states,
        )
        expanded = 0
        while expanded < len(subsets):
            newest = expanded
            expanded += 1
            _expand_newest_subset(
                automaton,
                images,
                subsets,
                index_of,
                rows,
                combos,
                work,
                newest,
                expanded,
                max_subset_states,
            )
    except _DeterminizeBudgetError as truncated:
        return subsets, rows, combos[0], truncated.reason
    return subsets, rows, combos[0], None


def _canonical_dfa(
    automaton: BottomUpTreeAutomaton,
    subsets: list[tuple[int, ...]],
    rows: dict[tuple[int, tuple[int, ...]], int],
) -> tuple[BottomUpTreeAutomaton, tuple[tuple[int, ...], ...]]:
    """Order subsets lexicographically and build the deterministic machine."""

    order = sorted(range(len(subsets)), key=lambda index: subsets[index])
    renumber = {old: new for new, old in enumerate(order)}
    canonical = tuple(subsets[old] for old in order)
    source_finals = set(automaton.final_states)
    deterministic = BottomUpTreeAutomaton(
        state_count=max(1, len(canonical)),
        arity=automaton.arity,
        transitions=tuple(
            sorted(
                (
                    TreeAutomatonTransition(
                        symbol=symbol,
                        child_states=tuple(renumber[child] for child in children),
                        target_state=renumber[target],
                    )
                    for (symbol, children), target in rows.items()
                    if all(child < len(subsets) for child in children)
                    and target < len(subsets)
                ),
                key=lambda row: (row.symbol, row.child_states, row.target_state),
            )
        ),
        final_states=tuple(
            sorted(
                index
                for index, subset in enumerate(canonical)
                if set(subset) & source_finals
            )
        ),
    )
    return deterministic, canonical


def _replay_dfa_closure(
    automaton: BottomUpTreeAutomaton,
    deterministic: BottomUpTreeAutomaton,
    subset_map: tuple[tuple[int, ...], ...],
) -> int:
    """Replay every DFA row against the subset map; return rows checked."""

    images = _determinize_images(automaton)
    work = [0]
    checked = 0
    for transition in deterministic.transitions:
        image = _image_of_subsets(
            images,
            automaton.state_count,
            transition.symbol,
            tuple(subset_map[child] for child in transition.child_states),
            work,
        )
        if image != subset_map[transition.target_state]:
            raise RuntimeError("determinized row disagrees with subset construction")
        checked += 1
    return checked


def _trees_of_height(
    automaton: BottomUpTreeAutomaton, max_height: int
) -> tuple[RankedTree, ...]:
    """Enumerate all run-admissible ground trees of height at most ``max_height``.

    Trees whose node count exceeds the run envelope are pruned during
    generation; enumeration stops with a resource error past the sample
    budget.
    """

    levels: list[list[tuple[RankedTree, int, int]]] = []
    current: list[tuple[RankedTree, int, int]] = []
    for symbol, rank in enumerate(automaton.arity):
        if rank == 0:
            current.append((RankedTree(symbol=symbol, children=()), 0, 1))
    if len(current) > MAX_DETERMINIZE_SAMPLE_TREES:
        raise OperationResourceAdmissionError(
            location=("automaton", "sample_max_height"),
            code="tree_automata.determinize_sample_bound_exceeded",
            message="the bounded-height tree sample exceeds the admitted budget; "
            "shrink sample_max_height",
        )
    levels.append(sorted(current, key=lambda entry: entry[0].symbol))
    for _ in range(max_height):
        previous = [entry for level in levels for entry in level]
        following: list[tuple[RankedTree, int, int]] = []
        for symbol, rank in enumerate(automaton.arity):
            if rank == 0:
                continue
            for children in product(previous, repeat=rank):
                if max(child[1] for child in children) != len(levels) - 1:
                    continue
                total = 1 + sum(child[2] for child in children)
                if total > MAX_RUN_TREE_NODES:
                    continue
                following.append(
                    (
                        RankedTree(
                            symbol=symbol,
                            children=tuple(child[0] for child in children),
                        ),
                        len(levels),
                        total,
                    )
                )
                if len(previous) + len(following) > MAX_DETERMINIZE_SAMPLE_TREES:
                    raise OperationResourceAdmissionError(
                        location=("automaton", "sample_max_height"),
                        code="tree_automata.determinize_sample_bound_exceeded",
                        message="the bounded-height tree sample exceeds the admitted "
                        "budget; shrink sample_max_height",
                    )
        if not following:
            break
        following.sort(
            key=lambda entry: (
                entry[0].symbol,
                tuple(child.symbol for child in entry[0].children),
                entry[2],
            )
        )
        levels.append(following)
    return tuple(tree for level in levels for tree, _, _ in level)


def determinize_tree_automaton(
    automaton: BottomUpTreeAutomaton,
    max_subset_states: int = 64,
    sample_max_height: int = 3,
) -> TreeDeterminizeResult:
    """Determinize a bottom-up tree automaton by subset construction.

    On success return the complete deterministic machine with its subset
    map, a replayed transition-closure certificate, and acceptance
    agreement on every ground tree of height at most ``sample_max_height``.
    When the powerset exceeds ``max_subset_states`` (or the shared work
    envelope), return the partial construction with ``TRUNCATED`` status
    and no language-equivalence claim.
    """

    if type(max_subset_states) is not int or not 1 <= max_subset_states <= 64:
        raise OperationDomainValidationError(
            location=("max_subset_states",),
            code="tree_automata.determinize_subset_budget",
            message="max_subset_states must be within 1..64",
        )
    if type(sample_max_height) is not int or not 0 <= sample_max_height <= 5:
        raise OperationDomainValidationError(
            location=("sample_max_height",),
            code="tree_automata.determinize_sample_height",
            message="sample_max_height must be within 0..5",
        )
    subsets, rows, combos, truncation = _subset_construction(
        automaton, max_subset_states
    )
    if truncation is not None:
        deterministic, canonical = _canonical_dfa(automaton, subsets, rows)
        return TreeDeterminizeResult._from_kernel(
            automaton=automaton,
            max_subset_states=max_subset_states,
            sample_max_height=sample_max_height,
            status="TRUNCATED",
            truncation_reason=truncation,
            deterministic=deterministic,
            subset_map=canonical,
            equivalence_claim=False,
            closure_rows_checked=0,
            combos_evaluated=combos,
            sample_trees_checked=0,
            sample_agreement=False,
        )
    if not subsets:
        deterministic = BottomUpTreeAutomaton(
            state_count=1, arity=automaton.arity, transitions=(), final_states=()
        )
        return TreeDeterminizeResult._from_kernel(
            automaton=automaton,
            max_subset_states=max_subset_states,
            sample_max_height=sample_max_height,
            status="COMPLETE",
            truncation_reason="NONE",
            deterministic=deterministic,
            subset_map=((),),
            equivalence_claim=True,
            closure_rows_checked=0,
            combos_evaluated=combos,
            sample_trees_checked=0,
            sample_agreement=True,
        )
    deterministic, canonical = _canonical_dfa(automaton, subsets, rows)
    closure_rows = _replay_dfa_closure(automaton, deterministic, canonical)
    sample = _trees_of_height(automaton, sample_max_height)
    source_finals = set(automaton.final_states)
    for tree in sample:
        source_roots = run_tree_automaton(automaton, tree)
        deterministic_roots = run_tree_automaton(deterministic, tree)
        source_accepted = bool(source_roots & source_finals)
        deterministic_accepted = bool(
            deterministic_roots & set(deterministic.final_states)
        )
        if source_accepted != deterministic_accepted:
            raise RuntimeError("determinized automaton disagrees with its source")
    return TreeDeterminizeResult._from_kernel(
        automaton=automaton,
        max_subset_states=max_subset_states,
        sample_max_height=sample_max_height,
        status="COMPLETE",
        truncation_reason="NONE",
        deterministic=deterministic,
        subset_map=canonical,
        equivalence_claim=True,
        closure_rows_checked=closure_rows,
        combos_evaluated=combos,
        sample_trees_checked=len(sample),
        sample_agreement=True,
    )


def verify_determinization(claim: TreeDeterminizeResult) -> bool:
    """Verify a determinization against its retained source automaton."""

    try:
        return (
            determinize_tree_automaton(
                claim.automaton, claim.max_subset_states, claim.sample_max_height
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
