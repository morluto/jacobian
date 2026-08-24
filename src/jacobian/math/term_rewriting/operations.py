"""Domain-owned first-order term rewriting kernels."""

from __future__ import annotations

from typing import Literal

from jacobian.math.term_rewriting.values import (
    MAX_CRITICAL_PAIR_CANDIDATES,
    MAX_CRITICAL_PAIR_RESULT_BYTES,
    MAX_CRITICAL_PAIR_RESULT_NODES,
    MAX_CRITICAL_PAIR_RULE_NODES,
    MAX_CRITICAL_PAIR_RULES,
    MAX_CRITICAL_PAIR_VARIABLE_ID,
    CriticalOverlapCandidate,
    CriticalPair,
    CriticalPairProfile,
    RankedSignature,
    RewriteApplication,
    RewriteRule,
    Term,
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


def _variables(term: Term) -> set[int]:
    if term.is_variable:
        return {term.symbol}
    result: set[int] = set()
    for child in term.children:
        result |= _variables(child)
    return result


def apply_substitution(term: Term, subst: dict[int, Term]) -> Term:
    """Apply a substitution to a term, replacing variables with their bindings."""
    if term.is_variable:
        if term.symbol in subst:
            return subst[term.symbol]
        return term
    new_children = tuple(apply_substitution(c, subst) for c in term.children)
    return Term(is_variable=False, symbol=term.symbol, children=new_children)


def match(pattern: Term, subject: Term) -> dict[int, Term] | None:
    """One-way matching: instantiate pattern variables to obtain the subject."""
    if pattern.is_variable:
        return {pattern.symbol: subject}
    if subject.is_variable:
        return None
    if pattern.symbol != subject.symbol:
        return None
    if len(pattern.children) != len(subject.children):
        return None
    result: dict[int, Term] = {}
    for p_child, s_child in zip(pattern.children, subject.children, strict=False):
        sub_result = match(p_child, s_child)
        if sub_result is None:
            return None
        for var, val in sub_result.items():
            if var in result and result[var] != val:
                return None
            result[var] = val
    return result


def _apply_recursive_substitution(term: Term, subst: dict[int, Term]) -> Term:
    if term.is_variable and term.symbol in subst:
        return _apply_recursive_substitution(subst[term.symbol], subst)
    if term.is_variable:
        return term
    return Term(
        is_variable=False,
        symbol=term.symbol,
        children=tuple(
            _apply_recursive_substitution(child, subst) for child in term.children
        ),
    )


def unify(left: Term, right: Term) -> dict[int, Term] | None:
    """Unify two terms, returning an idempotent most-general unifier."""
    return _unify(left, right, None)


def _unify(
    left: Term, right: Term, budget: _MaterializationBudget | None
) -> dict[int, Term] | None:
    """Run the kernel unification, charging materialized terms against budget."""
    equations = [(left, right)]
    substitution: dict[int, Term] = {}
    while equations:
        equation_left, equation_right = equations.pop()
        equation_left = _apply_recursive_substitution(equation_left, substitution)
        equation_right = _apply_recursive_substitution(equation_right, substitution)
        if budget is not None:
            budget.charge(equation_left)
            budget.charge(equation_right)
        if equation_left == equation_right:
            continue
        if equation_right.is_variable:
            equation_left, equation_right = equation_right, equation_left
        if equation_left.is_variable:
            if equation_left.symbol in _variables(equation_right):
                return None
            if budget is not None:
                budget.charge(equation_right)
            binding = {equation_left.symbol: equation_right}
            substitution = {
                variable: _apply_recursive_substitution(term, binding)
                for variable, term in substitution.items()
            }
            if budget is not None:
                for bound in substitution.values():
                    budget.charge(bound)
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
    child_index = position[0]
    children = list(term.children)
    children[child_index] = _replace_at_position(
        children[child_index], position[1:], replacement
    )
    return Term(is_variable=False, symbol=term.symbol, children=tuple(children))


def _positions(term: Term, prefix: tuple[int, ...] = ()) -> tuple[tuple[int, ...], ...]:
    return (
        prefix,
        *(
            position
            for child_index, child in enumerate(term.children)
            for position in _positions(child, (*prefix, child_index))
        ),
    )


def _nonvariable_positions(
    term: Term, prefix: tuple[int, ...] = ()
) -> tuple[tuple[int, ...], ...]:
    """Return exactly the positions whose subterm is a function application."""
    if term.is_variable:
        return ()
    return (
        prefix,
        *(
            position
            for child_index, child in enumerate(term.children)
            for position in _nonvariable_positions(child, (*prefix, child_index))
        ),
    )


def _term_node_count(term: Term) -> int:
    return 1 + sum(_term_node_count(child) for child in term.children)


class _ResultEnvelopeError(Exception):
    """Raised when admitted materialization would leave the node envelope."""


class _MaterializationBudget:
    """Charge every materialized term against a remaining node allowance.

    Charges count materialization events, so they upper-bound both the
    unification work performed and the distinct nodes any produced binding
    contributes to a result.
    """

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def charge(self, term: Term) -> None:
        self.remaining -= _term_node_count(term)
        if self.remaining < 0:
            raise _ResultEnvelopeError


def _require_bounded_variable_ids(term: Term) -> None:
    if term.is_variable and term.symbol > MAX_CRITICAL_PAIR_VARIABLE_ID:
        raise ValueError("critical-pair variable IDs exceed the supported bound")
    for child in term.children:
        _require_bounded_variable_ids(child)


def _expanded_node_count(term: Term, binding_sizes: dict[int, int]) -> int:
    """Node count of ``apply_substitution(term, ...)`` without materializing it."""
    if term.is_variable:
        return binding_sizes.get(term.symbol, 1)
    return 1 + sum(_expanded_node_count(child, binding_sizes) for child in term.children)


def _admit_critical_pair_result_envelope(rules: tuple[RewriteRule, ...]) -> int:
    """Charge every materialized overlap object against the result envelope.

    Binding growth is dependency-driven: a chained idempotent MGU expands
    exponentially in its binding depth, so no product of rule-side caps bounds
    the materialized substitution or its reducts. Each candidate is therefore
    standardized apart and unified exactly as execution will, under the shared
    remaining node allowance and with every materialized term charged; reduct
    sizes are then computed exactly from the substitution. Returns the total
    charged nodes and raises when the envelope is exceeded.
    """
    remaining = MAX_CRITICAL_PAIR_RESULT_NODES
    for outer_index, outer in enumerate(rules):
        for position in _nonvariable_positions(outer.lhs):
            for inner_index, inner in enumerate(rules):
                if outer_index == inner_index and not position:
                    continue
                (
                    standardized_outer,
                    standardized_inner,
                    _outer_renaming,
                    _inner_renaming,
                ) = _standardize_apart(outer, inner)
                budget = _MaterializationBudget(remaining)
                try:
                    substitution = _unify(
                        standardized_inner.lhs,
                        term_at_position(standardized_outer.lhs, position),
                        budget,
                    )
                except _ResultEnvelopeError:
                    raise ValueError(_RESULT_NODES_EXCEEDED) from None
                if substitution is None:
                    continue
                remaining = budget.remaining
                binding_sizes = {
                    variable: _term_node_count(binding)
                    for variable, binding in substitution.items()
                }
                remaining -= _expanded_node_count(
                    _replace_at_position(
                        standardized_outer.lhs, position, standardized_inner.rhs
                    ),
                    binding_sizes,
                )
                remaining -= _expanded_node_count(
                    standardized_outer.rhs, binding_sizes
                )
                if remaining < 0:
                    raise ValueError(_RESULT_NODES_EXCEEDED)
    return MAX_CRITICAL_PAIR_RESULT_NODES - remaining


def _validate_critical_pair_source(
    signature: RankedSignature, rules: tuple[RewriteRule, ...]
) -> None:
    """Admit the complete overlap/replay envelope before enumerating it."""
    if not 1 <= len(rules) <= MAX_CRITICAL_PAIR_RULES:
        raise ValueError("critical-pair rule count exceeds the supported bound")
    for rule in rules:
        signature.validate_term(rule.lhs)
        signature.validate_term(rule.rhs)
        _require_bounded_variable_ids(rule.lhs)
        _require_bounded_variable_ids(rule.rhs)
        if (
            _term_node_count(rule.lhs) > MAX_CRITICAL_PAIR_RULE_NODES
            or _term_node_count(rule.rhs) > MAX_CRITICAL_PAIR_RULE_NODES
        ):
            raise ValueError("critical-pair rule sides exceed the supported node bound")
    canonical_rules = {_canonical_rule(rule).model_dump_json() for rule in rules}
    if len(canonical_rules) != len(rules):
        raise ValueError("critical-pair rules must be duplicate-free up to renaming")
    candidates = len(rules) * sum(
        len(_nonvariable_positions(rule.lhs)) for rule in rules
    ) - len(rules)
    if candidates > MAX_CRITICAL_PAIR_CANDIDATES:
        raise ValueError("critical-pair overlap candidates exceed the supported bound")
    charged_nodes = _admit_critical_pair_result_envelope(rules)
    if charged_nodes * 96 > MAX_CRITICAL_PAIR_RESULT_BYTES:
        raise ValueError("critical-pair result bytes exceed the supported bound")


def _preorder_variables(term: Term) -> tuple[int, ...]:
    """Return each variable once, in structural preorder."""
    seen: set[int] = set()
    ordered: list[int] = []
    for position in _positions(term):
        subterm = term_at_position(term, position)
        if subterm.is_variable and subterm.symbol not in seen:
            seen.add(subterm.symbol)
            ordered.append(subterm.symbol)
    return tuple(ordered)


def _rename_variables(term: Term, renaming: dict[int, int]) -> Term:
    if term.is_variable:
        return Term(is_variable=True, symbol=renaming[term.symbol])
    return Term(
        symbol=term.symbol,
        children=tuple(_rename_variables(child, renaming) for child in term.children),
    )


def _canonical_rule(rule: RewriteRule) -> RewriteRule:
    variables = _preorder_variables(rule.lhs)
    renaming = {variable: index for index, variable in enumerate(variables)}
    return RewriteRule(
        lhs=_rename_variables(rule.lhs, renaming),
        rhs=_rename_variables(rule.rhs, renaming),
    )


def _standardize_apart(
    outer: RewriteRule, inner: RewriteRule
) -> tuple[RewriteRule, RewriteRule, dict[int, int], dict[int, int]]:
    """Canonically rename one ordered pair of rules to disjoint variables."""
    outer_variables = _preorder_variables(outer.lhs)
    outer_renaming = {variable: index for index, variable in enumerate(outer_variables)}
    inner_renaming = {
        variable: len(outer_variables) + index
        for index, variable in enumerate(_preorder_variables(inner.lhs))
    }
    return (
        RewriteRule(
            lhs=_rename_variables(outer.lhs, outer_renaming),
            rhs=_rename_variables(outer.rhs, outer_renaming),
        ),
        RewriteRule(
            lhs=_rename_variables(inner.lhs, inner_renaming),
            rhs=_rename_variables(inner.rhs, inner_renaming),
        ),
        outer_renaming,
        inner_renaming,
    )


def critical_pairs(
    signature: RankedSignature, rules: tuple[RewriteRule, ...]
) -> CriticalPairProfile:
    """Enumerate all unifiable nonvariable overlaps of a finite TRS.

    Rows are ordered by outer rule, position, then inner rule.  The tautological
    root overlap of a rule with itself is excluded; all other source-indexed
    overlaps are retained, even if their displayed reducts coincide.
    """
    _validate_critical_pair_source(signature, rules)
    candidates: list[CriticalOverlapCandidate] = []
    pairs: list[CriticalPair] = []
    for outer_index, outer in enumerate(rules):
        for position in _nonvariable_positions(outer.lhs):
            for inner_index, inner in enumerate(rules):
                if outer_index == inner_index and not position:
                    continue
                (
                    standardized_outer,
                    standardized_inner,
                    outer_renaming,
                    inner_renaming,
                ) = _standardize_apart(outer, inner)
                substitution = unify(
                    standardized_inner.lhs,
                    term_at_position(standardized_outer.lhs, position),
                )
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
                inner_reduct = apply_substitution(
                    _replace_at_position(
                        standardized_outer.lhs,
                        position,
                        standardized_inner.rhs,
                    ),
                    substitution,
                )
                outer_reduct = apply_substitution(standardized_outer.rhs, substitution)
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
