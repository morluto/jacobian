"""Tests for first-order term rewriting operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.term_rewriting.operations import (
    apply_substitution,
    match,
    normal_form,
    rewrite_step,
    unify,
)
from jacobian.math.term_rewriting.values import RewriteRule, Term


# Helpers
def _var(symbol: int) -> Term:
    return Term(is_variable=True, symbol=symbol)


def _app(symbol: int, *children: Term) -> Term:
    return Term(is_variable=False, symbol=symbol, children=tuple(children))


class TestSubstitution:
    def test_substitute_variable(self):
        term = _var(0)
        result = apply_substitution(term, {0: _app(1)})
        assert result == _app(1)

    def test_substitute_in_children(self):
        term = _app(0, _var(0), _var(1))
        result = apply_substitution(term, {0: _app(1)})
        assert result == _app(0, _app(1), _var(1))

    def test_substitute_no_change(self):
        term = _app(0, _var(0))
        result = apply_substitution(term, {})
        assert result == term


class TestMatching:
    def test_match_variable(self):
        result = match(_var(0), _app(1))
        assert result == {0: _app(1)}

    def test_match_function(self):
        pattern = _app(0, _var(0), _var(1))
        subject = _app(0, _app(1), _app(2))
        result = match(pattern, subject)
        assert result == {0: _app(1), 1: _app(2)}

    def test_match_symbol_mismatch(self):
        result = match(_app(0), _app(1))
        assert result is None

    def test_match_arity_mismatch(self):
        result = match(_app(0, _var(0)), _app(0, _var(0), _var(1)))
        assert result is None


class TestUnification:
    def test_unify_same_symbol(self):
        result = unify(_app(0, _var(0), _app(1)), _app(0, _app(2), _app(1)))
        assert result is not None
        assert result == {0: _app(2)}

    def test_unify_variables(self):
        result = unify(_var(0), _var(1))
        assert result is not None

    def test_unify_failure(self):
        result = unify(_app(0), _app(1))
        assert result is None

    def test_unify_occurs_check(self):
        result = unify(_var(0), _app(1, _var(0)))
        assert result is None


class TestRewriteStep:
    def test_rewrite_root(self):
        # Rule: f(x) -> g(x)
        rules = [RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0)))]
        term = _app(0, _app(2))
        rewritten, new_term = rewrite_step(term, rules)
        assert rewritten
        assert new_term == _app(1, _app(2))

    def test_rewrite_in_child(self):
        # Rule: f(x) -> g(x)
        rules = [RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0)))]
        term = _app(3, _app(0, _app(2)))
        rewritten, new_term = rewrite_step(term, rules)
        assert rewritten
        assert new_term == _app(3, _app(1, _app(2)))

    def test_no_rewrite(self):
        rules = [RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0)))]
        term = _app(2, _app(2))
        rewritten, _ = rewrite_step(term, rules)
        assert not rewritten


class TestNormalForm:
    def test_convergent(self):
        # Rule: f(x) -> x  (strips one f per step)
        rules = [RewriteRule(lhs=_app(0, _var(0)), rhs=_var(0))]
        term = _app(0, _app(0, _app(1)))
        result, converged, steps = normal_form(term, rules, max_steps=100)
        assert converged
        assert result == _app(1)
        assert steps == 2

    def test_non_convergent(self):
        # Rule: f(x) -> f(f(x))  (divergent)
        rules = [RewriteRule(lhs=_app(0, _var(0)), rhs=_app(0, _app(0, _var(0))))]
        term = _app(0, _app(1))
        _result, converged, steps = normal_form(term, rules, max_steps=10)
        assert not converged
        assert steps == 10


class TestValidation:
    def test_variable_with_children_rejected(self):
        with pytest.raises(ValidationError):
            Term(is_variable=True, symbol=0, children=(_var(1),))

    def test_lhs_must_be_function(self):
        with pytest.raises(ValidationError):
            RewriteRule(lhs=_var(0), rhs=_app(1))
