"""Domain-owned first-order term rewriting kernels."""

from __future__ import annotations

from typing import Literal

from jacobian.math.logic.term_rewriting.values import (
    MAX_CRITICAL_PAIR_CANDIDATES,
    MAX_CRITICAL_PAIR_RESULT_BYTES,
    MAX_CRITICAL_PAIR_RESULT_NODES,
    MAX_CRITICAL_PAIR_RULES,
    MAX_TERM_DEPTH,
    CriticalOverlapCandidate,
    CriticalPair,
    CriticalPairProfile,
    RankedSignature,
    RewriteApplication,
    RewriteRule,
    Term,
    _variable_symbols,
)

__all__ = [
    "apply_substitution",
    "critical_pairs",
    "match",
    "normal_form",
    "rewrite_steps",
    "selected_rewrite_step",
    "term_at_position",
    "unify",
]

_RESULT_NODES_EXCEEDED = "critical-pair result nodes exceed the supported bound"
_RESULT_BYTES_EXCEEDED = "critical-pair result bytes exceed the supported bound"
_RESULT_DEPTH_EXCEEDED = "critical-pair result depth exceeds the transport-safe bound"
_CRITICAL_PAIR_PROFILE_FIXED_NODES = 384
_RESULT_BYTES_PER_NODE = 96
_VARIABLE_LABEL_BASE_WIDTH = 6
_RESULT_TERM_MAX_DEPTH = MAX_TERM_DEPTH - 1


def apply_substitution(term: Term, subst: dict[int, Term]) -> Term:
    """Apply a substitution to a term, replacing variables with their bindings."""
    if term.is_variable:
        return subst.get(term.symbol, term)
    rebuilt: dict[int, Term] = {}
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            rebuilt[id(current)] = subst.get(current.symbol, current)
            continue
        pending = [child for child in current.children if id(child) not in rebuilt]
        if pending:
            stack.append(current)
            stack.extend(pending)
        else:
            rebuilt[id(current)] = Term(
                is_variable=False,
                symbol=current.symbol,
                children=tuple(rebuilt[id(child)] for child in current.children),
            )
    return rebuilt[id(term)]


def match(pattern: Term, subject: Term) -> dict[int, Term] | None:
    """One-way matching: instantiate pattern variables to obtain the subject."""
    result: dict[int, Term] = {}
    stack = [(pattern, subject)]
    while stack:
        pattern_part, subject_part = stack.pop()
        if pattern_part.is_variable:
            bound = result.get(pattern_part.symbol)
            if bound is None:
                result[pattern_part.symbol] = subject_part
            elif not _structural_equal(bound, subject_part):
                return None
            continue
        if (
            subject_part.is_variable
            or pattern_part.symbol != subject_part.symbol
            or len(pattern_part.children) != len(subject_part.children)
        ):
            return None
        stack.extend(zip(pattern_part.children, subject_part.children, strict=True))
    return result


def _structural_equal(left: Term, right: Term) -> bool:
    """Compare two terms structurally without recursing on host frames."""
    pending = [(left, right)]
    while pending:
        left_part, right_part = pending.pop()
        if left_part is right_part:
            continue
        if (
            left_part.is_variable != right_part.is_variable
            or left_part.symbol != right_part.symbol
            or len(left_part.children) != len(right_part.children)
        ):
            return False
        pending.extend(zip(left_part.children, right_part.children, strict=True))
    return True


def _apply_recursive_substitution(term: Term, subst: dict[int, Term]) -> Term:
    resolved: dict[int, Term] = {}
    stack = [term]
    while stack:
        current = stack.pop()
        if id(current) in resolved:
            continue
        if current.is_variable:
            bound = subst.get(current.symbol)
            if bound is None:
                resolved[id(current)] = current
                continue
            target = bound
            while target.is_variable and target.symbol in subst:
                target = subst[target.symbol]
            if target.is_variable or id(target) in resolved:
                resolved[id(current)] = resolved.get(id(target), target)
                continue
            stack.append(current)
            stack.append(target)
            continue
        pending = [child for child in current.children if id(child) not in resolved]
        if pending:
            stack.append(current)
            stack.extend(pending)
        else:
            resolved[id(current)] = Term(
                is_variable=False,
                symbol=current.symbol,
                children=tuple(resolved[id(child)] for child in current.children),
            )
    return resolved[id(term)]


def unify(left: Term, right: Term) -> dict[int, Term] | None:
    """Unify two terms, returning an idempotent most-general unifier."""
    return _unify(left, right, None)


def _unify(
    left: Term, right: Term, budget: _MaterializationBudget | None
) -> dict[int, Term] | None:
    """Run the kernel unification, charging materialized terms against budget.

    The substitution stays idempotent, so ``_expanded_node_count`` predicts
    each substituted equation exactly; both equation expansions are charged
    from that prediction before either is materialized.
    """
    equations = [(left, right)]
    substitution: dict[int, Term] = {}
    substitution_sizes: dict[int, int] = {}
    while equations:
        equation_left, equation_right = equations.pop()
        if budget is not None:
            budget.charge_nodes(_expanded_node_count(equation_left, substitution_sizes))
            budget.charge_nodes(
                _expanded_node_count(equation_right, substitution_sizes)
            )
        equation_left = _apply_recursive_substitution(equation_left, substitution)
        equation_right = _apply_recursive_substitution(equation_right, substitution)
        if _structural_equal(equation_left, equation_right):
            continue
        if equation_right.is_variable:
            equation_left, equation_right = equation_right, equation_left
        if equation_left.is_variable:
            if equation_left.symbol in _variable_symbols(equation_right):
                return None
            if budget is not None:
                budget.charge(equation_right)
            binding = {equation_left.symbol: equation_right}
            binding_size = _term_node_count(equation_right)
            binding_sizes = {equation_left.symbol: binding_size}
            expanded_substitution: dict[int, Term] = {}
            expanded_sizes: dict[int, int] = {}
            for variable, bound in substitution.items():
                expanded_size = _expanded_node_count(bound, binding_sizes)
                if budget is not None:
                    budget.charge_nodes(expanded_size)
                expanded_sizes[variable] = expanded_size
                expanded_substitution[variable] = _apply_recursive_substitution(
                    bound, binding
                )
            substitution_sizes = expanded_sizes
            substitution_sizes[equation_left.symbol] = binding_size
            substitution = expanded_substitution
            substitution[equation_left.symbol] = equation_right
            continue
        if (
            equation_right.is_variable
            or equation_left.symbol != equation_right.symbol
            or len(equation_left.children) != len(equation_right.children)
        ):
            return None
        equations.extend(
            zip(equation_left.children, equation_right.children, strict=True)
        )
    return substitution


def _bounded_unify(left: Term, right: Term) -> dict[int, Term] | None:
    """Unify within the bounded public result envelope."""

    try:
        return _unify(
            left,
            right,
            _MaterializationBudget(MAX_CRITICAL_PAIR_RESULT_NODES),
        )
    except _ResultEnvelopeError as error:
        raise ValueError(
            "unification result nodes exceed the supported bound"
        ) from error


def term_at_position(term: Term, position: tuple[int, ...]) -> Term:
    """Return the subterm at a child-index path, raising for an invalid path."""
    current = term
    for child_index in position:
        if not 0 <= child_index < len(current.children):
            raise ValueError("rewrite position is outside the source term")
        current = current.children[child_index]
    return current


def _replace_at_position(
    term: Term, position: tuple[int, ...], replacement: Term
) -> Term:
    if not position:
        return replacement
    spine = [term]
    for child_index in position:
        parent = spine[-1]
        if not 0 <= child_index < len(parent.children):
            raise ValueError("rewrite position is outside the source term")
        spine.append(parent.children[child_index])
    current = replacement
    for depth in reversed(range(len(position))):
        parent = spine[depth]
        children = list(parent.children)
        children[position[depth]] = current
        current = Term(
            is_variable=False, symbol=parent.symbol, children=tuple(children)
        )
    return current


def _positions(term: Term, prefix: tuple[int, ...] = ()) -> tuple[tuple[int, ...], ...]:
    positions: list[tuple[int, ...]] = []
    stack: list[tuple[Term, tuple[int, ...]]] = [(term, prefix)]
    while stack:
        current, path = stack.pop()
        positions.append(path)
        for child_index in reversed(range(len(current.children))):
            stack.append((current.children[child_index], (*path, child_index)))
    return tuple(positions)


def _nonvariable_positions(
    term: Term, prefix: tuple[int, ...] = ()
) -> tuple[tuple[int, ...], ...]:
    """Return exactly the positions whose subterm is a function application."""
    positions: list[tuple[int, ...]] = []
    stack: list[tuple[Term, tuple[int, ...]]] = [(term, prefix)]
    while stack:
        current, path = stack.pop()
        if current.is_variable:
            continue
        positions.append(path)
        for child_index in reversed(range(len(current.children))):
            stack.append((current.children[child_index], (*path, child_index)))
    return tuple(positions)


def _nonvariable_position_count(term: Term) -> int:
    """Count the positions whose subterm is a function application.

    Admission counts candidates before any canonicalization work, so a
    source that exceeds the candidate bound is rejected without ever
    materializing the position paths of a deep term.
    """
    count = 0
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            continue
        count += 1
        stack.extend(current.children)
    return count


def _term_node_count(term: Term) -> int:
    count = 0
    stack = [term]
    while stack:
        current = stack.pop()
        count += 1
        stack.extend(current.children)
    return count


class _ResultEnvelopeError(Exception):
    """Raised when admitted materialization would leave the node envelope."""


class _MaterializationBudget:
    """Charge every materialized term against a remaining node allowance.

    Charges count materialization events, so they upper-bound both the
    unification work performed and the distinct nodes any produced binding
    contributes to a result. Growth is predicted and charged before the
    corresponding terms are constructed, so an exceeded allowance is always
    detected before its allocation.
    """

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def charge_nodes(self, count: int) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise _ResultEnvelopeError

    def charge(self, term: Term) -> None:
        self.charge_nodes(_term_node_count(term))


def _variable_label_bytes(rules: tuple[RewriteRule, ...]) -> int:
    """Serialized-byte overhead for variable labels wider than the baseline.

    Variable labels never change the mathematical work, so admission charges
    their serialized width; the label maximum itself is the interoperable
    JSON integer range, a transport boundary rather than a work bound. The
    per-node byte cost already absorbs the baseline label width; labels
    wider than ``_VARIABLE_LABEL_BASE_WIDTH`` digits appear in the echoed
    source rules and in the original-ID renaming keys of every candidate row
    and pair, so their count is upper-bounded by the source variable nodes
    times one echo plus two full ledgers.
    """
    widest = 0
    variable_nodes = 0
    for rule in rules:
        stack = [rule.lhs, rule.rhs]
        while stack:
            term = stack.pop()
            if term.is_variable:
                variable_nodes += 1
                width = len(str(term.symbol))
                if width > widest:
                    widest = width
            else:
                stack.extend(term.children)
    extra_width = max(0, widest - _VARIABLE_LABEL_BASE_WIDTH)
    return extra_width * variable_nodes * (1 + 2 * MAX_CRITICAL_PAIR_CANDIDATES)


def _expanded_node_count(term: Term, binding_sizes: dict[int, int]) -> int:
    """Node count of ``apply_substitution(term, ...)`` without materializing it."""
    count = 0
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            count += binding_sizes.get(current.symbol, 1)
        else:
            count += 1
            stack.extend(current.children)
    return count


def _term_depth(term: Term) -> int:
    """Longest root-to-leaf path measured in nodes, without recursion."""
    deepest = 0
    stack = [(term, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > deepest:
            deepest = depth
        stack.extend((child, depth + 1) for child in current.children)
    return deepest


def _expanded_max_depth(term: Term, binding_depths: dict[int, int]) -> int:
    """Depth of ``apply_substitution(term, ...)`` without materializing it."""
    depths: dict[int, int] = {}
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            depths[id(current)] = binding_depths.get(current.symbol, 1)
            continue
        pending = [child for child in current.children if id(child) not in depths]
        if pending:
            stack.append(current)
            stack.extend(pending)
        else:
            depths[id(current)] = (
                1
                if not current.children
                else 1 + max(depths[id(child)] for child in current.children)
            )
    return depths[id(term)]


def _retained_source_charge(rules: tuple[RewriteRule, ...]) -> int:
    """Node count charged once for echoing every retained source term."""
    return sum(
        _term_node_count(rule.lhs) + _term_node_count(rule.rhs) for rule in rules
    )


def _admit_critical_pair_result_envelope(rules: tuple[RewriteRule, ...]) -> int:
    """Charge every retained result object against the result envelope.

    The result serializes the source rules beside the overlap profile, so the
    retained source terms and the fixed profile scaffold are charged before
    any overlap work: every source term node is charged exactly once, and
    ``_CRITICAL_PAIR_PROFILE_FIXED_NODES`` covers the shape-fixed scaffolding
    of a full candidate ledger and pair family (row framing, positions,
    renaming entries, and substitution keys). A zero-candidate system
    therefore still pays for the rules its result echoes.

    Binding growth is dependency-driven: a chained idempotent MGU expands
    exponentially in its binding depth, so no product of rule-side caps bounds
    the materialized substitution or its reducts. Each candidate is therefore
    unified exactly as execution will, under the shared remaining node
    allowance and with every materialized term charged: the two renamed
    unification terms are charged before construction, and a successful
    unifier additionally pays for the renamed outer rule, its spliced left
    side, both renamed right sides, its retained bindings, and the exact
    expansions execution will materialize as reducts. Failed unifications
    keep their committed charges against the shared allowance.

    Transport adds one further result obligation: reducts serialize under
    ``profile.pairs`` four strict-JSON wrappers deep and pair substitution
    bindings five, and every serialized node costs an object level plus a
    ``children`` array level including the leaf's empty array, so any
    transported reduct or binding carries at most
    ``_RESULT_TERM_MAX_DEPTH`` nodes on a root-to-leaf path. Both depths
    are predicted from the unifier without materializing the reducts and
    reject typedly.
    Returns the total charged nodes and raises when the envelope is exceeded.
    """
    remaining = (
        MAX_CRITICAL_PAIR_RESULT_NODES
        - _CRITICAL_PAIR_PROFILE_FIXED_NODES
        - _retained_source_charge(rules)
    )
    if remaining < 0:
        raise ValueError(_RESULT_NODES_EXCEEDED)
    for outer_index, outer in enumerate(rules):
        for position in _nonvariable_positions(outer.lhs):
            for inner_index, inner in enumerate(rules):
                if outer_index == inner_index and not position:
                    continue
                (
                    renamed_overlap,
                    renamed_inner_lhs,
                    outer_renaming,
                    inner_renaming,
                ) = _overlap_unification_terms(outer, inner, position)
                budget = _MaterializationBudget(remaining)
                try:
                    budget.charge_nodes(_term_node_count(renamed_inner_lhs))
                    budget.charge_nodes(_term_node_count(renamed_overlap))
                    substitution = _unify(
                        renamed_inner_lhs,
                        renamed_overlap,
                        budget,
                    )
                except _ResultEnvelopeError:
                    raise ValueError(_RESULT_NODES_EXCEEDED) from None
                remaining = budget.remaining
                if substitution is None:
                    continue
                overlap_size = _term_node_count(renamed_overlap)
                spliced_size = (
                    _term_node_count(outer.lhs)
                    - overlap_size
                    + _term_node_count(inner.rhs)
                )
                try:
                    budget.charge_nodes(_term_node_count(outer.lhs))
                    budget.charge_nodes(spliced_size)
                    budget.charge_nodes(_term_node_count(inner.rhs))
                    budget.charge_nodes(_term_node_count(outer.rhs))
                except _ResultEnvelopeError:
                    raise ValueError(_RESULT_NODES_EXCEEDED) from None
                remaining = budget.remaining
                renamed_outer_lhs = _rename_variables(outer.lhs, outer_renaming)
                renamed_inner_rhs = _rename_variables(inner.rhs, inner_renaming)
                spliced = _replace_at_position(
                    renamed_outer_lhs, position, renamed_inner_rhs
                )
                renamed_outer_rhs = _rename_variables(outer.rhs, outer_renaming)
                binding_sizes = {
                    variable: _term_node_count(binding)
                    for variable, binding in substitution.items()
                }
                binding_depths = {
                    variable: _term_depth(binding)
                    for variable, binding in substitution.items()
                }
                for bound in substitution.values():
                    remaining -= _term_node_count(bound)
                remaining -= _expanded_node_count(spliced, binding_sizes)
                remaining -= _expanded_node_count(renamed_outer_rhs, binding_sizes)
                if remaining < 0:
                    raise ValueError(_RESULT_NODES_EXCEEDED)
                if max(binding_depths.values(), default=0) > (_RESULT_TERM_MAX_DEPTH):
                    raise ValueError(_RESULT_DEPTH_EXCEEDED)
                if (
                    _expanded_max_depth(spliced, binding_depths)
                    > _RESULT_TERM_MAX_DEPTH
                    or _expanded_max_depth(renamed_outer_rhs, binding_depths)
                    > _RESULT_TERM_MAX_DEPTH
                ):
                    raise ValueError(_RESULT_DEPTH_EXCEEDED)
    return MAX_CRITICAL_PAIR_RESULT_NODES - remaining


def _validate_critical_pair_source(
    signature: RankedSignature, rules: tuple[RewriteRule, ...]
) -> None:
    """Admit the complete overlap/replay envelope before enumerating it."""
    if len(rules) > MAX_CRITICAL_PAIR_RULES:
        raise ValueError("critical-pair rule count exceeds the supported bound")
    for rule in rules:
        signature.validate_term(rule.lhs)
        signature.validate_term(rule.rhs)
    if (
        _CRITICAL_PAIR_PROFILE_FIXED_NODES + _retained_source_charge(rules)
        > MAX_CRITICAL_PAIR_RESULT_NODES
    ):
        raise ValueError(_RESULT_NODES_EXCEEDED)
    canonical_rules = {_canonical_rule_key(rule) for rule in rules}
    if len(canonical_rules) != len(rules):
        raise ValueError("critical-pair rules must be duplicate-free up to renaming")
    candidates = len(rules) * sum(
        _nonvariable_position_count(rule.lhs) for rule in rules
    ) - len(rules)
    if candidates > MAX_CRITICAL_PAIR_CANDIDATES:
        raise ValueError("critical-pair overlap candidates exceed the supported bound")
    charged_nodes = _admit_critical_pair_result_envelope(rules)
    label_bytes = _variable_label_bytes(rules)
    if charged_nodes * _RESULT_BYTES_PER_NODE + label_bytes > (
        MAX_CRITICAL_PAIR_RESULT_BYTES
    ):
        raise ValueError(_RESULT_BYTES_EXCEEDED)


def _preorder_variables(term: Term) -> tuple[int, ...]:
    """Return each variable once, in structural preorder."""
    seen: set[int] = set()
    ordered: list[int] = []
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            if current.symbol not in seen:
                seen.add(current.symbol)
                ordered.append(current.symbol)
            continue
        stack.extend(reversed(current.children))
    return tuple(ordered)


def _rename_variables(term: Term, renaming: dict[int, int]) -> Term:
    if term.is_variable:
        return Term(is_variable=True, symbol=renaming[term.symbol])
    rebuilt: dict[int, Term] = {}
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            rebuilt[id(current)] = Term(
                is_variable=True, symbol=renaming[current.symbol]
            )
            continue
        pending = [child for child in current.children if id(child) not in rebuilt]
        if pending:
            stack.append(current)
            stack.extend(pending)
        else:
            rebuilt[id(current)] = Term(
                is_variable=False,
                symbol=current.symbol,
                children=tuple(rebuilt[id(child)] for child in current.children),
            )
    return rebuilt[id(term)]


def _canonical_rule_key(rule: RewriteRule) -> str:
    """Flat token stream identifying the rule up to variable renaming.

    Variables are renamed to their left-hand-side preorder indices and both
    sides are emitted as prefix-free ``V<index>`` / ``A<symbol>:<arity>``
    tokens, so two rules are duplicate-free up to renaming exactly when
    their keys are equal. The stream is built iteratively, which keeps deep
    sources away from serializer recursion limits.
    """
    renaming = {
        variable: index for index, variable in enumerate(_preorder_variables(rule.lhs))
    }
    parts: list[str] = []
    for side in (rule.lhs, rule.rhs):
        stack = [side]
        while stack:
            current = stack.pop()
            if current.is_variable:
                parts.append(f"V{renaming[current.symbol]};")
            else:
                parts.append(f"A{current.symbol}:{len(current.children)};")
                stack.extend(reversed(current.children))
    return "".join(parts)


def _renaming_for(variables: tuple[int, ...], offset: int) -> dict[int, int]:
    """Map each variable to ``offset`` plus its preorder index."""
    return {variable: offset + index for index, variable in enumerate(variables)}


def _renamed_rule(rule: RewriteRule, renaming: dict[int, int]) -> RewriteRule:
    return RewriteRule(
        lhs=_rename_variables(rule.lhs, renaming),
        rhs=_rename_variables(rule.rhs, renaming),
    )


def _overlap_unification_terms(
    outer: RewriteRule, inner: RewriteRule, position: tuple[int, ...]
) -> tuple[Term, Term, dict[int, int], dict[int, int]]:
    """Rename only the two terms one overlap unification inspects.

    Unification never inspects either rule's right side or the outer left
    side away from ``position``, so those copies are deferred until an
    overlap actually unifies; only the renamed inner left side and the
    renamed subterm at ``position`` are constructed here.
    """
    outer_variables = _preorder_variables(outer.lhs)
    outer_renaming = _renaming_for(outer_variables, 0)
    inner_renaming = _renaming_for(_preorder_variables(inner.lhs), len(outer_variables))
    return (
        _rename_variables(term_at_position(outer.lhs, position), outer_renaming),
        _rename_variables(inner.lhs, inner_renaming),
        outer_renaming,
        inner_renaming,
    )


def critical_pairs(
    signature: RankedSignature, rules: tuple[RewriteRule, ...]
) -> CriticalPairProfile:
    """Validate and enumerate all unifiable nonvariable overlaps."""

    _validate_critical_pair_source(signature, rules)
    return _critical_pairs(signature, rules)


def _critical_pairs(
    signature: RankedSignature, rules: tuple[RewriteRule, ...]
) -> CriticalPairProfile:
    """Enumerate all unifiable nonvariable overlaps of a finite TRS.

    Rows are ordered by outer rule, position, then inner rule.  The tautological
    root overlap of a rule with itself is excluded; all other source-indexed
    overlaps are retained, even if their displayed reducts coincide.
    """
    candidates: list[CriticalOverlapCandidate] = []
    pairs: list[CriticalPair] = []
    for outer_index, outer in enumerate(rules):
        for position in _nonvariable_positions(outer.lhs):
            for inner_index, inner in enumerate(rules):
                if outer_index == inner_index and not position:
                    continue
                (
                    renamed_overlap,
                    renamed_inner_lhs,
                    outer_renaming,
                    inner_renaming,
                ) = _overlap_unification_terms(outer, inner, position)
                substitution = unify(renamed_inner_lhs, renamed_overlap)
                candidate_index = len(candidates)
                candidates.append(
                    CriticalOverlapCandidate(
                        outer_rule_index=outer_index,
                        inner_rule_index=inner_index,
                        position=position,
                        outer_variable_renaming=outer_renaming,
                        inner_variable_renaming=inner_renaming,
                        unifiable=substitution is not None,
                    )
                )
                if substitution is None:
                    continue
                renamed_outer_rule = _renamed_rule(outer, outer_renaming)
                spliced = _replace_at_position(
                    renamed_outer_rule.lhs,
                    position,
                    _rename_variables(inner.rhs, inner_renaming),
                )
                inner_reduct = apply_substitution(spliced, substitution)
                outer_reduct = apply_substitution(renamed_outer_rule.rhs, substitution)
                pairs.append(
                    CriticalPair(
                        candidate_index=candidate_index,
                        outer_variable_renaming=outer_renaming,
                        inner_variable_renaming=inner_renaming,
                        substitution=substitution,
                        inner_reduct=inner_reduct,
                        outer_reduct=outer_reduct,
                    )
                )
    return CriticalPairProfile(candidates=tuple(candidates), pairs=tuple(pairs))


def selected_rewrite_step(
    term: Term,
    rules: tuple[RewriteRule, ...],
    position: tuple[int, ...],
    rule_index: int,
) -> RewriteApplication | None:
    """Apply exactly the declared rule at exactly the declared position."""
    if not 0 <= rule_index < len(rules):
        raise ValueError("rule_index is out of range")
    redex = term_at_position(term, position)
    substitution = match(rules[rule_index].lhs, redex)
    if substitution is None:
        return None
    replacement = apply_substitution(rules[rule_index].rhs, substitution)
    return RewriteApplication(
        position=position,
        rule_index=rule_index,
        substitution=substitution,
        term=_replace_at_position(term, position, replacement),
    )


def rewrite_steps(
    term: Term, rules: tuple[RewriteRule, ...]
) -> tuple[RewriteApplication, ...]:
    """Return every applicable one-step derivation, including its witness."""
    return tuple(
        application
        for position in _positions(term)
        for rule_index in range(len(rules))
        if (application := selected_rewrite_step(term, rules, position, rule_index))
        is not None
    )


def normal_form(
    term: Term, rules: tuple[RewriteRule, ...], max_steps: int = 1000
) -> tuple[
    Term,
    Literal["NORMAL_FORM", "STEP_LIMIT"],
    int,
    RewriteApplication | None,
]:
    """Run the explicit leftmost-outermost, rule-order strategy.

    Returns (term, status, steps, next_step). ``next_step`` is the open
    obligation when the declared step bound is exhausted.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    steps = 0
    current = term
    while steps < max_steps:
        applications = rewrite_steps(current, rules)
        if not applications:
            return (current, "NORMAL_FORM", steps, None)
        current = applications[0].term
        steps += 1
    applications = rewrite_steps(current, rules)
    if not applications:
        return (current, "NORMAL_FORM", steps, None)
    return (current, "STEP_LIMIT", steps, applications[0])
