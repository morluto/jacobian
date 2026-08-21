"""Domain-owned first-order term rewriting kernels."""

from __future__ import annotations

from typing import Literal

from jacobian.math.term_rewriting.values import RewriteApplication, RewriteRule, Term

__all__ = [
    "apply_substitution",
    "match",
    "normal_form",
    "rewrite_steps",
    "selected_rewrite_step",
    "term_at_position",
    "unify",
]


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
    equations = [(left, right)]
    substitution: dict[int, Term] = {}
    while equations:
        equation_left, equation_right = equations.pop()
        equation_left = _apply_recursive_substitution(equation_left, substitution)
        equation_right = _apply_recursive_substitution(equation_right, substitution)
        if equation_left == equation_right:
            continue
        if equation_right.is_variable:
            equation_left, equation_right = equation_right, equation_left
        if equation_left.is_variable:
            if equation_left.symbol in _variables(equation_right):
                return None
            binding = {equation_left.symbol: equation_right}
            substitution = {
                variable: _apply_recursive_substitution(term, binding)
                for variable, term in substitution.items()
            }
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
