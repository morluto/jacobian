"""Canonical one-hole contexts for finite ranked trees."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_RUN_TREE_DEPTH,
    MAX_RUN_TREE_NODES,
    MAX_TA_ARITY,
    MAX_TA_SYMBOLS,
    Arity,
    CompleteDeterministicBottomUpTreeAutomaton,
    RankedTree,
    ranked_tree_node_count,
)


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tree_context.{code}", message)


class TreeContextFrame(StrictModel):
    """One ancestor on the path from a context root to its unique hole."""

    symbol: int = Field(
        ge=0, lt=MAX_TA_SYMBOLS, description="Ranked alphabet symbol at this ancestor."
    )
    hole_child: int = Field(
        ge=0,
        lt=MAX_TA_ARITY,
        description="Zero-based child position that contains the unique hole.",
    )
    siblings: tuple[RankedTree, ...] = Field(
        max_length=MAX_TA_ARITY - 1,
        description="All other children, with the hole child omitted and order preserved.",
    )


class FiniteTreeContext(StrictModel):
    """A ranked tree with one distinguished leaf hole, encoded by its spine.

    Frames are ordered root-to-hole. In each frame `siblings` lists all
    children other than the hole child in their original order. An empty
    frame tuple is the identity context consisting only of a hole.
    """

    arity: tuple[Arity, ...] = Field(
        max_length=MAX_TA_SYMBOLS,
        description="Exact ranked signature; entry i is the arity of symbol i.",
    )
    frames: tuple[TreeContextFrame, ...] = Field(
        max_length=MAX_RUN_TREE_DEPTH,
        description="Canonical root-to-hole path; empty frames represent the identity context.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_context(cls, value: object) -> object:  # noqa: C901
        """Reject oversized JSON trees before Pydantic builds nested values."""
        if not isinstance(value, dict):
            return value
        if set(value) != {"arity", "frames"}:
            raise _error("shape", "context must contain only arity and frames")
        raw_arity = value.get("arity")
        if not isinstance(raw_arity, (tuple, list)) or len(raw_arity) > MAX_TA_SYMBOLS:
            raise _error("signature", "context arity exceeds the supported bound")
        frames = value.get("frames")
        if not isinstance(frames, (tuple, list)):
            return value
        if len(frames) > MAX_RUN_TREE_DEPTH:
            raise _error("depth", "context spine exceeds the supported depth")
        nodes = len(frames)
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict) or set(frame) != {
                "symbol",
                "hole_child",
                "siblings",
            }:
                raise _error("shape", "context frame has invalid fields")
            siblings = frame.get("siblings")
            if not isinstance(siblings, (tuple, list)):
                return value
            if len(siblings) > MAX_TA_ARITY - 1:
                raise _error("arity", "context frame has too many siblings")
            for sibling in siblings:
                if not isinstance(sibling, dict) or set(sibling) != {
                    "symbol",
                    "children",
                }:
                    raise _error("shape", "context sibling has invalid fields")
                stack = [(sibling, frame_index + 2)]
                while stack:
                    node, depth = stack.pop()
                    children = node.get("children")
                    if not isinstance(children, (tuple, list)):
                        return value
                    nodes += 1
                    if nodes > MAX_RUN_TREE_NODES:
                        raise _error(
                            "node_bound",
                            "context size exceeds the supported node bound",
                        )
                    if depth > MAX_RUN_TREE_DEPTH:
                        raise _error(
                            "depth", "context exceeds the supported tree depth"
                        )
                    if len(children) > MAX_TA_ARITY:
                        raise _error("arity", "tree node has too many children")
                    if any(not isinstance(child, dict) for child in children):
                        raise _error("shape", "tree children must be objects")
                    stack.extend((child, depth + 1) for child in children)
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if any(
            type(rank) is not int or not 0 <= rank <= MAX_TA_ARITY
            for rank in self.arity
        ):
            raise _error("rank", "context arities must be in the supported range")
        if len(self.frames) > MAX_RUN_TREE_DEPTH:
            raise _error("depth", "context spine exceeds the supported depth")
        node_count = len(self.frames)
        for frame_index, frame in enumerate(self.frames):
            if frame.symbol >= len(self.arity):
                raise _error("symbol", "context symbol is outside its ranked alphabet")
            rank = self.arity[frame.symbol]
            if frame.hole_child >= rank or len(frame.siblings) != rank - 1:
                raise _error("frame_rank", "context frame does not match symbol arity")
            for sibling in frame.siblings:
                node_count += ranked_tree_node_count(sibling)
                if frame_index + 1 + _tree_depth(sibling) > MAX_RUN_TREE_DEPTH:
                    raise _error("depth", "context exceeds the supported tree depth")
            if node_count > MAX_RUN_TREE_NODES:
                raise _error(
                    "node_bound", "context size exceeds the supported node bound"
                )
            if any(
                not _tree_matches(self.arity, sibling) for sibling in frame.siblings
            ):
                raise _error(
                    "sibling_rank", "context sibling tree does not match its alphabet"
                )
        return self


def _tree_matches(arity: tuple[int, ...], tree: RankedTree) -> bool:
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.symbol >= len(arity) or len(node.children) != arity[node.symbol]:
            return False
        stack.extend(node.children)
    return True


def _context_size(context: FiniteTreeContext) -> int:
    return len(context.frames) + sum(
        ranked_tree_node_count(sibling)
        for frame in context.frames
        for sibling in frame.siblings
    )


def _plug_tree_context(context: FiniteTreeContext, tree: RankedTree) -> RankedTree:
    """Substitute a ground tree into the unique hole of a context."""
    if not _tree_matches(context.arity, tree):
        raise OperationDomainValidationError(
            location=("tree",),
            code="tree_context.plug.alphabet_mismatch",
            message="plugged tree must match the context ranked alphabet",
        )
    current = tree
    for frame in reversed(context.frames):
        children = iter(frame.siblings)
        rebuilt = tuple(
            current if index == frame.hole_child else next(children)
            for index in range(context.arity[frame.symbol])
        )
        current = RankedTree(symbol=frame.symbol, children=rebuilt)
    return current


def _tree_depth(tree: RankedTree) -> int:
    depth = 0
    stack = [(tree, 1)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        stack.extend((child, d + 1) for child in node.children)
    return depth


def _tree_context_state_map(
    automaton: CompleteDeterministicBottomUpTreeAutomaton,
    context: FiniteTreeContext,
) -> tuple[int, ...]:
    """Return q -> state(C[q]) by propagating each state up the context spine."""
    if automaton.arity != context.arity:
        raise OperationDomainValidationError(
            location=("context", "arity"),
            code="tree_context.state_map.alphabet_mismatch",
            message="automaton and context ranked alphabets must match",
        )
    table = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }
    sibling_states = [
        tuple(_tree_state(table, sibling) for sibling in frame.siblings)
        for frame in context.frames
    ]
    results = []
    for start in range(automaton.state_count):
        state = start
        for frame_index in range(len(context.frames) - 1, -1, -1):
            frame = context.frames[frame_index]
            row = iter(sibling_states[frame_index])
            child_states = tuple(
                state if i == frame.hole_child else next(row)
                for i in range(context.arity[frame.symbol])
            )
            state = table[(frame.symbol, child_states)]
        results.append(state)
    return tuple(results)


def _tree_state(table: dict[tuple[int, tuple[int, ...]], int], tree: RankedTree) -> int:
    states: dict[int, int] = {}
    stack = [(tree, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children)
        else:
            states[id(node)] = table[
                (node.symbol, tuple(states[id(c)] for c in node.children))
            ]
    return states[id(tree)]


__all__ = ["FiniteTreeContext", "TreeContextFrame"]
