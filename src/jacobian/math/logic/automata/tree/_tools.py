"""Tree automaton operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    RankedTreePositionsRequest,
    RankedTreePositionsResult,
    RankedTreeSubtreeRequest,
    RankedTreeSubtreeResult,
    TreeAutomatonBooleanProductRequest,
    TreeAutomatonBooleanProductResult,
    TreeAutomatonComplementRequest,
    TreeAutomatonComplementResult,
    TreeAutomatonCompletionRequest,
    TreeAutomatonCompletionResult,
    TreeAutomatonMinimizeRequest,
    TreeAutomatonMinimizeResult,
    TreeAutomatonReachabilityRequest,
    TreeAutomatonStateAlgebraRequest,
    TreeAutomatonTrimRequest,
    TreeAutomatonTrimResult,
    TreeDeterminizeRequest,
    TreeDeterminizeResult,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.operations import (
    _accepted_tree_count_admitted,
    _tree_state_chart_unchecked,
    boolean_product_tree_automata,
    complement_tree_automaton,
    complete_deterministic_tree_automaton,
    determinize_tree_automaton,
    minimize_tree_automaton,
    ranked_tree_positions,
    ranked_tree_subtree,
    reachable_state_profile,
    trim_tree_automaton,
)
from jacobian.math.logic.automata.tree.state_algebra import (
    deterministic_tree_automaton_state_algebra,
)
from jacobian.math.logic.automata.tree.values import (
    ReachableStateProfile,
    TreeStateChartEntry,
    _admit_nondeterministic_run_counts,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)
from jacobian.math.universal_algebra.values import FiniteAlgebra


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    node_count = validate_ranked_tree(request.automaton, request.tree)
    chart = _tree_state_chart_unchecked(request.automaton, request.tree)
    states = set(chart[-1][1])
    accepting = set(states) & set(request.automaton.final_states)
    return TreeRunResult._from_kernel(
        request,
        accepted=bool(accepting),
        root_states=tuple(sorted(states)),
        state_chart=tuple(
            TreeStateChartEntry(position=position, states=states)
            for position, states in chart
        ),
        node_count=node_count,
    )


def compute_accepted_tree_count(
    request: AcceptedTreeCountRequest,
) -> AcceptedTreeCountResult:
    estimated_work_bound = accepted_tree_count_work_bound(
        request.automaton, request.tree_size
    )
    return AcceptedTreeCountResult._from_kernel(
        request,
        count=_accepted_tree_count_admitted(request.automaton, request.tree_size),
        estimated_work_bound=estimated_work_bound,
    )


def compute_nondeterministic_run_counts(
    request: NondeterministicRunCountsRequest,
) -> NondeterministicRunCountsResult:
    admission = _admit_nondeterministic_run_counts(request.automaton, request.max_size)
    counts = (
        (0,) * request.max_size
        if admission.zero
        else _nondeterministic_run_counts_admitted(
            request.automaton, request.max_size, admission.reachable
        )
    )
    return NondeterministicRunCountsResult._from_kernel(
        request,
        run_counts_by_size=counts,
        estimated_work_bound=admission.work,
    )


def compute_regular_tree_grammar_to_automaton(
    request: RegularTreeGrammarToAutomatonRequest,
) -> RegularTreeGrammarToAutomatonResult:
    return RegularTreeGrammarToAutomatonResult._from_kernel(
        grammar=request.grammar,
        automaton=regular_tree_grammar_to_automaton(request.grammar),
    )
 def compute_ranked_tree_positions(
    request: RankedTreePositionsRequest,
) -> RankedTreePositionsResult:
    return ranked_tree_positions(request.tree)


def compute_ranked_tree_subtree(
    request: RankedTreeSubtreeRequest,
) -> RankedTreeSubtreeResult:
    return ranked_tree_subtree(request.tree, request.position)


def compute_tree_automaton_boolean_product(
    request: TreeAutomatonBooleanProductRequest,
) -> TreeAutomatonBooleanProductResult:
    return boolean_product_tree_automata(
        request.left, request.right, request.connective
    )


def compute_tree_automaton_reachability(
    request: TreeAutomatonReachabilityRequest,
) -> ReachableStateProfile:
    """Compute the exact reachable-state profile with minimum tree witnesses."""

    return reachable_state_profile(request.automaton)


def compute_tree_automaton_trim(
    request: TreeAutomatonTrimRequest,
) -> TreeAutomatonTrimResult:
    """Restrict an automaton to its reachable and productive states."""

    return trim_tree_automaton(request.automaton)


def compute_tree_automaton_state_algebra(
    request: TreeAutomatonStateAlgebraRequest,
) -> FiniteAlgebra:
    """Interpret each ranked symbol as its complete operation table."""

    return deterministic_tree_automaton_state_algebra(request.automaton)


def compute_tree_automaton_determinize(
    request: TreeDeterminizeRequest,
) -> TreeDeterminizeResult:
    """Determinize an automaton by bounded subset construction."""

    return determinize_tree_automaton(
        request.automaton,
        request.max_subset_states,
        request.sample_max_height,
    )


def compute_tree_automaton_complement(
    request: TreeAutomatonComplementRequest,
) -> TreeAutomatonComplementResult:
    """Complement an admitted complete deterministic bottom-up automaton."""
    return complement_tree_automaton(request.automaton)


def compute_tree_automaton_completion(
    request: TreeAutomatonCompletionRequest,
) -> TreeAutomatonCompletionResult:
    return complete_deterministic_tree_automaton(request.automaton)


def compute_tree_automaton_minimize(
    request: TreeAutomatonMinimizeRequest,
) -> TreeAutomatonMinimizeResult:
    return minimize_tree_automaton(request.automaton)


# Automaton: states {0, 1}, symbols {a (arity 0), f (arity 2)}
# Transitions: a -> 0, f(0, 0) -> 0, f(1, 0) -> 1, f(0, 1) -> 1, f(1, 1) -> 1
# Final states: {0}
_RUN_EXAMPLE = {
    "automaton": {
        "state_count": 2,
        "arity": [0, 2],
        "transitions": [
            {"symbol": 0, "child_states": [], "target_state": 0},
            {"symbol": 1, "child_states": [0, 0], "target_state": 0},
            {"symbol": 1, "child_states": [0, 1], "target_state": 1},
            {"symbol": 1, "child_states": [1, 0], "target_state": 1},
            {"symbol": 1, "child_states": [1, 1], "target_state": 1},
        ],
        "final_states": [0],
    },
    "tree": {
        "symbol": 1,
        "children": [
            {"symbol": 0, "children": []},
            {"symbol": 0, "children": []},
        ],
    },
}

# Nondeterministic automaton for boolean AND-trees: states {0 (false),
# 1 (true), 2 (true alias)}. Symbols: 0=false (arity 0), 1=true (arity 0),
# 2=AND (arity 2). Subset construction reaches {{0}, {1, 2}}.
_DETERMINIZE_EXAMPLE = {
    "state_count": 3,
    "arity": [0, 0, 2],
    "transitions": [
        {"symbol": 0, "child_states": [], "target_state": 0},
        {"symbol": 1, "child_states": [], "target_state": 1},
        {"symbol": 1, "child_states": [], "target_state": 2},
        {"symbol": 2, "child_states": [0, 0], "target_state": 0},
        {"symbol": 2, "child_states": [0, 1], "target_state": 0},
        {"symbol": 2, "child_states": [0, 2], "target_state": 0},
        {"symbol": 2, "child_states": [1, 0], "target_state": 0},
        {"symbol": 2, "child_states": [2, 0], "target_state": 0},
        {"symbol": 2, "child_states": [1, 1], "target_state": 1},
        {"symbol": 2, "child_states": [1, 2], "target_state": 1},
        {"symbol": 2, "child_states": [2, 1], "target_state": 1},
        {"symbol": 2, "child_states": [2, 2], "target_state": 1},
        {"symbol": 2, "child_states": [2, 2], "target_state": 2},
    ],
    "final_states": [1],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="tree_automaton.boolean_product.compute",
        title="Boolean product of deterministic tree automata",
        description=(
            "Construct the exact synchronous product of two complete deterministic "
            "bottom-up automata over the same ranked signature. Choose intersection, "
            "union, left difference, or symmetric difference. A missing transition "
            "is rejected at the input boundary; state-pair and transition-output bounds are admitted "
            "before product expansion."
        ),
        request_type=TreeAutomatonBooleanProductRequest,
        result_type=TreeAutomatonBooleanProductResult,
        run=compute_tree_automaton_boolean_product,
        tags=("tree-automata", "boolean-operations", "product", "exact"),
        discovery_terms=(
            "tree language intersection",
            "tree language union",
            "tree automaton product",
        ),
        examples=(
            OperationExample(
                name="intersect_complete_nullary_machines",
                description=(
                    "Build an intersection product of two complete one-state "
                    "machines over one nullary symbol; Boolean products "
                    "require complete deterministic inputs."
                ),
                input={
                    "left": {
                        "state_count": 1,
                        "arity": [0],
                        "transitions": [
                            {"symbol": 0, "child_states": [], "target_state": 0}
                        ],
                        "final_states": [0],
                    },
                    "right": {
                        "state_count": 1,
                        "arity": [0],
                        "transitions": [
                            {"symbol": 0, "child_states": [], "target_state": 0}
                        ],
                        "final_states": [0],
                    },
                    "connective": "intersection",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.states.reachable.compute",
        title="Compute bottom-up tree-automaton reachable states",
        description="Return the complete least-fixed-point set of states reachable by a "
        "finite ground ranked tree, its complement, and one canonical "
        "minimum-node witness tree per reachable state. A transition is "
        "enabled only when all of its ordered child states are reachable. "
        "When several derivations tie at the minimum node count, each state's "
        "witness is the unique one whose root transition (symbol, "
        "child_states, target_state) is lexicographically smallest, comparing "
        "child_states element-wise as integers, with each child witness "
        "chosen by the same rule recursively.",
        request_type=TreeAutomatonReachabilityRequest,
        result_type=ReachableStateProfile,
        run=compute_tree_automaton_reachability,
        tags=("tree-automata", "reachability", "fixed-point", "exact"),
        examples=(
            OperationExample(
                name="leaf_seed_and_binary_extension",
                description="Find states generated from a leaf and a binary "
                "constructor; every transition's symbol must index the "
                "ranked alphabet and its child_states count must equal "
                "arity[symbol].",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.trim.compute",
        title="Trim a bottom-up tree automaton",
        description="Restrict a bottom-up nondeterministic tree automaton to the "
        "states that are both reachable by a ground tree and productive inside "
        "an accepting run, returning the trimmed automaton, old/new state "
        "maps, and one replayed minimum-node witness tree per kept state on "
        "the trimmed automaton. Restriction preserves the accepted language "
        "exactly; the empty language trims to the canonical one-state "
        "automaton with no transitions and no final states.",
        request_type=TreeAutomatonTrimRequest,
        result_type=TreeAutomatonTrimResult,
        run=compute_tree_automaton_trim,
        tags=("tree-automata", "trim", "reachability", "exact"),
        discovery_terms=("tree automaton trim", "useful states", "trimming"),
        examples=(
            OperationExample(
                name="drop_unreachable_state",
                description="Drop the unreachable state 1 from a two-state automaton.",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.determinize.compute",
        title="Determinize a bottom-up tree automaton",
        description="Run bounded subset construction on a nondeterministic "
        "bottom-up tree automaton. On success return the complete "
        "deterministic machine with its subset map, a replayed "
        "transition-closure certificate, and acceptance agreement on every "
        "ground tree up to a bounded height. When the powerset exceeds the "
        "state-set budget, return the partial construction with TRUNCATED "
        "status and no language-equivalence claim.",
        request_type=TreeDeterminizeRequest,
        result_type=TreeDeterminizeResult,
        run=compute_tree_automaton_determinize,
        tags=("tree-automata", "determinization", "subset-construction", "exact"),
        discovery_terms=("determinize", "subset construction", "powerset"),
        examples=(
            OperationExample(
                name="determinize_and_trees",
                description="Determinize a nondeterministic automaton for "
                "boolean AND-trees with an aliased true state.",
                input={
                    "automaton": _DETERMINIZE_EXAMPLE,
                    "max_subset_states": 64,
                    "sample_max_height": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.deterministic.complete.compute",
        title="Complete a partial deterministic tree automaton",
        description=(
            "Fill every missing symbol and child-state transition with one "
            "appended nonfinal sink state. Preserve the source state order, "
            "return the source-to-completed state map and sink identity, and "
            "preserve the accepted tree language. A complete input is returned "
            "unchanged with no sink. State, transition-row, output-cell, and "
            "work bounds are checked before Cartesian expansion."
        ),
        request_type=TreeAutomatonCompletionRequest,
        result_type=TreeAutomatonCompletionResult,
        run=compute_tree_automaton_completion,
        tags=("tree-automata", "completion", "deterministic", "exact"),
        discovery_terms=(
            "complete partial tree automaton",
            "add sink transition",
            "total deterministic bottom-up automaton",
        ),
        examples=(
            OperationExample(
                name="complete_nullary_unary_automaton",
                description=(
                    "Complete a one-state partial automaton over a constant "
                    "and unary symbol by adding a nonfinal sink."
                ),
                input={
                    "automaton": {
                        "state_count": 1,
                        "arity": [0, 1],
                        "transitions": [
                            {
                                "symbol": 0,
                                "child_states": [],
                                "target_state": 0,
                            }
                        ],
                        "final_states": [0],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.complement.compute",
        title="Complement a complete deterministic tree automaton",
        description=(
            "Require a complete deterministic bottom-up automaton over its "
            "explicit finite ranked signature. Canonically relabel states, "
            "return the exact old/new maps, preserve the full transition "
            "table, and complement the final-state set. Incomplete or "
            "nondeterministic input is rejected; transition product and result "
            "cells are admitted before expansion."
        ),
        request_type=TreeAutomatonComplementRequest,
        result_type=TreeAutomatonComplementResult,
        run=compute_tree_automaton_complement,
        tags=("tree-automata", "complement", "complete-deterministic", "exact"),
        discovery_terms=("complement tree language", "tree automaton complement"),
        examples=(
            OperationExample(
                name="complement_complete_unary_automaton",
                description="Complement a complete automaton over a nullary and unary signature.",
                input={
                    "automaton": {
                        "state_count": 2,
                        "arity": [0, 1],
                        "transitions": [
                            {"symbol": 0, "child_states": [], "target_state": 1},
                            {"symbol": 1, "child_states": [0], "target_state": 0},
                            {"symbol": 1, "child_states": [1], "target_state": 1},
                        ],
                        "final_states": [0],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.deterministic.minimize.compute",
        title="Minimize a deterministic bottom-up tree automaton",
        description=(
            "Return the smallest deterministic quotient over the same ranked "
            "signature, restricted to states reachable by finite ground trees. "
            "States are merged exactly when every ground-tree context accepts "
            "from both or neither; undefined transitions stay undefined. "
            "The old-to-new map uses -1 for unreachable source states, and the "
            "refinement work bound is checked before partition expansion."
        ),
        request_type=TreeAutomatonMinimizeRequest,
        result_type=TreeAutomatonMinimizeResult,
        run=compute_tree_automaton_minimize,
        tags=("tree-automata", "minimization", "deterministic", "exact"),
        discovery_terms=(
            "minimize tree automaton",
            "minimal deterministic tree automaton",
        ),
        examples=(
            OperationExample(
                name="merge_equivalent_reachable_states",
                description="Merge two nonfinal states with identical unary behavior and remove an unreachable state.",
                input={
                    "automaton": {
                        "state_count": 4,
                        "arity": [0, 1],
                        "transitions": [
                            {"symbol": 0, "child_states": [], "target_state": 0},
                            {"symbol": 1, "child_states": [0], "target_state": 1},
                            {"symbol": 1, "child_states": [1], "target_state": 1},
                            {"symbol": 1, "child_states": [2], "target_state": 2},
                        ],
                        "final_states": [],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.run.compute",
        title="Run a bottom-up tree automaton on a ranked tree",
        description="Execute a nondeterministic bottom-up tree automaton on a ranked "
        "tree and return the set of reachable root states and whether the "
        "tree is accepted.",
        request_type=TreeRunRequest,
        result_type=TreeRunResult,
        run=compute_tree_run,
        tags=("tree-automata", "run", "exact"),
        examples=(
            OperationExample(
                name="simple_run",
                description="Run a tree automaton on f(a, a).",
                input=_RUN_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.accepted_tree_count.compute",
        title="Count accepted trees of a given size",
        description="Count the number of ranked trees of a given size accepted by a "
        "bottom-up nondeterministic tree automaton. On-the-fly subset-state "
        "dynamic programming counts each distinct tree once, even when it has "
        "multiple accepting runs; the validated request carries a conservative "
        "work bound.",
        request_type=AcceptedTreeCountRequest,
        result_type=AcceptedTreeCountResult,
        run=compute_accepted_tree_count,
        tags=("tree-automata", "counting", "exact"),
        examples=(
            OperationExample(
                name="count_size_1",
                description="Count accepted trees of size 1.",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                    "tree_size": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ranked_tree.positions.compute",
        title="List all positions in a ranked tree",
        description=(
            "Return every node address as a zero-based child-index path in "
            "root-first preorder. The complete tree is retained, and node, "
            "depth, traversal-work, and result-byte bounds are admitted before "
            "the position list is materialized."
        ),
        request_type=RankedTreePositionsRequest,
        result_type=RankedTreePositionsResult,
        run=compute_ranked_tree_positions,
        tags=("ranked-tree", "positions", "exact"),
        discovery_terms=("tree node positions", "ranked tree paths"),
        examples=(
            OperationExample(
                name="binary_tree_positions",
                description="List positions for a root with a leaf and a unary child.",
                input={
                    "tree": {
                        "symbol": 0,
                        "children": [
                            {"symbol": 1, "children": []},
                            {
                                "symbol": 2,
                                "children": [{"symbol": 3, "children": []}],
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ranked_tree.subtree.compute",
        title="Extract a ranked-tree subtree",
        description=(
            "Return the subtree at a zero-based child-index position, retaining "
            "the complete source tree and source position. Tree validation, "
            "traversal work, and the duplicated source-bound result size are "
            "bounded before the exact result is constructed."
        ),
        request_type=RankedTreeSubtreeRequest,
        result_type=RankedTreeSubtreeResult,
        run=compute_ranked_tree_subtree,
        tags=("ranked-tree", "subtree", "exact"),
        discovery_terms=(
            "subtree extraction",
            "tree node address",
            "ranked tree context",
        ),
        examples=(
            OperationExample(
                name="extract_left_child",
                description="Extract the left subtree at position [0].",
                input={
                    "tree": {
                        "symbol": 0,
                        "children": [
                            {"symbol": 1, "children": []},
                            {"symbol": 2, "children": []},
                        ],
                    },
                    "position": [0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.deterministic.state_algebra.compute",
        title="Construct the transition algebra of a deterministic tree automaton",
        description=(
            "Return the finite algebra on the complete automaton state set whose "
            "operation tree_symbol_j is the transition function for ranked symbol "
            "j. Carrier positions preserve every state, including unreachable "
            "states; symbol order and arity are retained in the algebra signature. "
            "The result represents transition evaluation and does not include the "
            "automaton's accepting-state subset. Carrier, arity, signature, and "
            "expanded table-cell bounds are checked before table construction."
        ),
        request_type=TreeAutomatonStateAlgebraRequest,
        result_type=FiniteAlgebra,
        run=compute_tree_automaton_state_algebra,
        tags=("tree-automata", "universal-algebra", "exact"),
        discovery_terms=(
            "tree automaton state algebra",
            "finite algebra of tree transitions",
            "evaluate ranked tree in transition algebra",
        ),
        examples=(
            OperationExample(
                name="ranked_symbol_operations",
                description=(
                    "For a complete deterministic input automaton—exactly one "
                    "transition for every ranked symbol and child-state tuple—"
                    "interpret the nullary and binary ranked symbols as "
                    "operations on the exact automaton state set."
                ),
                input={
                    "automaton": {
                        "state_count": 2,
                        "arity": [0, 2],
                        "transitions": [
                            {"symbol": 0, "child_states": [], "target_state": 1},
                            {"symbol": 1, "child_states": [0, 0], "target_state": 0},
                            {"symbol": 1, "child_states": [0, 1], "target_state": 1},
                            {"symbol": 1, "child_states": [1, 0], "target_state": 1},
                            {"symbol": 1, "child_states": [1, 1], "target_state": 0},
                        ],
                        "final_states": [1],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
