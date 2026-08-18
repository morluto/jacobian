"""Domain-owned first-order term rewriting kernels."""

from __future__ import annotations

from jacobian.math.term_rewriting.values import RewriteRule, Term

__all__ = [
    "apply_substitution",
    "match",
    "normal_form",
    "rewrite_step",
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
    """One-way matching: match a pattern (with variables) against a ground subject."""
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


def _unify(
    left: Term, right: Term, subst: dict[int, Term],
) -> dict[int, Term] | None:
    """Recursive unification with substitution accumulation."""
    left = _resolve(left, subst)
    right = _resolve(right, subst)
    if left.is_variable and right.is_variable and left.symbol == right.symbol:
        return subst
    if left.is_variable:
        if left.symbol in _variables(right):
            return None
        new_subst = dict(subst)
        new_subst[left.symbol] = right
        return new_subst
    if right.is_variable:
        if right.symbol in _variables(left):
            return None
        new_subst = dict(subst)
        new_subst[right.symbol] = left
        return new_subst
    if left.symbol != right.symbol or len(left.children) != len(right.children):
        return None
    new_subst = dict(subst)
    for l_child, r_child in zip(left.children, right.children, strict=False):
        result = _unify(l_child, r_child, new_subst)
        if result is None:
            return None
        new_subst = result
    return new_subst


def _resolve(term: Term, subst: dict[int, Term]) -> Term:
    if term.is_variable and term.symbol in subst:
        return _resolve(subst[term.symbol], subst)
    return term


def unify(left: Term, right: Term) -> dict[int, Term] | None:
    """Unify two terms, returning a substitution or None."""
    return _unify(left, right, {})


def _rewrite_at_root(term: Term, rules: list[RewriteRule]) -> Term | None:
    for rule in rules:
        subst = match(rule.lhs, term)
        if subst is not None:
            return apply_substitution(rule.rhs, subst)
    return None


def rewrite_step(term: Term, rules: list[RewriteRule]) -> tuple[bool, Term]:
    """Apply one rewrite step at the leftmost-outermost redex.

    Returns (rewritten, new_term).
    """
    result = _rewrite_at_root(term, rules)
    if result is not None:
        return (True, result)
    for i, child in enumerate(term.children):
        rewritten, new_child = rewrite_step(child, rules)
        if rewritten:
            new_children = list(term.children)
            new_children[i] = new_child
            return (True, Term(is_variable=False, symbol=term.symbol, children=tuple(new_children)))
    return (False, term)


def normal_form(
    term: Term, rules: list[RewriteRule], max_steps: int = 1000,
) -> tuple[Term, bool, int]:
    """Compute the normal form of a term.

    Returns (term, converged, steps).
    """
    steps = 0
    current = term
    while steps < max_steps:
        rewritten, new_term = rewrite_step(current, rules)
        if not rewritten:
            return (current, True, steps)
        current = new_term
        steps += 1
    return (current, False, steps)
