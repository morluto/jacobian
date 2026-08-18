"""Tests for first-order term rewriting operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.term_rewriting._models import (
    NormalFormRequest,
    RewriteStepRequest,
    UnificationRequest,
)
from jacobian.math.term_rewriting._operations import (
    compute_normal_form,
    compute_rewrite_step,
    compute_unification,
)
from jacobian.math.term_rewriting.operations import (
    apply_substitution,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
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

    def test_unifier_is_idempotent_and_independently_replays(self):
        left = _app(0, _var(0), _var(0))
        right = _app(0, _var(1), _app(2))
        result = unify(left, right)
        assert result == {0: _app(2), 1: _app(2)}
        assert apply_substitution(left, result) == apply_substitution(right, result)
        wire_result = compute_unification(UnificationRequest(left=left, right=right))
        assert wire_result.unified
        assert wire_result.substitution == result


class TestRewriteStep:
    def test_rewrite_root(self):
        # Rule: f(x) -> g(x)
        rules = (RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0))),)
        term = _app(0, _app(2))
        applications = rewrite_steps(term, rules)
        assert len(applications) == 1
        assert applications[0].position == ()
        assert applications[0].rule_index == 0
        assert applications[0].substitution == {0: _app(2)}
        assert applications[0].term == _app(1, _app(2))

    def test_rewrite_in_child(self):
        # Rule: f(x) -> g(x)
        rules = (RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0))),)
        term = _app(3, _app(0, _app(2)))
        applications = rewrite_steps(term, rules)
        assert len(applications) == 1
        assert applications[0].position == (0,)
        assert applications[0].term == _app(3, _app(1, _app(2)))

    def test_no_rewrite(self):
        rules = (RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0))),)
        term = _app(2, _app(2))
        assert rewrite_steps(term, rules) == ()

    def test_all_steps_preserve_rule_and_position_choices(self):
        rules = (
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0))),
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(2, _var(0))),
        )
        term = _app(0, _app(0, _app(3)))
        result = compute_rewrite_step(RewriteStepRequest(term=term, rules=rules))
        assert result.scope == "ALL_APPLICABLE_STEPS"
        assert tuple(
            (application.position, application.rule_index)
            for application in result.applications
        ) == (((), 0), ((), 1), ((0,), 0), ((0,), 1))
        assert tuple(application.term for application in result.applications) == (
            _app(1, _app(0, _app(3))),
            _app(2, _app(0, _app(3))),
            _app(0, _app(1, _app(3))),
            _app(0, _app(2, _app(3))),
        )

    def test_selected_step_applies_only_declared_choice(self):
        rules = (
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(0))),
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(2, _var(0))),
        )
        term = _app(0, _app(0, _app(3)))
        result = compute_rewrite_step(
            RewriteStepRequest(
                term=term,
                rules=rules,
                selection={"position": [0], "rule_index": 1},
            )
        )
        assert result.scope == "SELECTED_STEP"
        assert len(result.applications) == 1
        assert result.applications[0].term == _app(0, _app(2, _app(3)))

    def test_selected_inapplicable_rule_is_exact_negative(self):
        rules = (RewriteRule(lhs=_app(0), rhs=_app(1)),)
        assert selected_rewrite_step(_app(2), rules, (), 0) is None
        result = compute_rewrite_step(
            RewriteStepRequest(
                term=_app(2),
                rules=rules,
                selection={"position": [], "rule_index": 0},
            )
        )
        assert result.scope == "SELECTED_STEP"
        assert result.applications == ()


class TestNormalForm:
    def test_convergent(self):
        # Rule: f(x) -> x  (strips one f per step)
        rules = (RewriteRule(lhs=_app(0, _var(0)), rhs=_var(0)),)
        term = _app(0, _app(0, _app(1)))
        result, status, steps, next_step = normal_form(term, rules, max_steps=100)
        assert status == "NORMAL_FORM"
        assert result == _app(1)
        assert steps == 2
        assert next_step is None

    def test_non_convergent(self):
        # Rule: f(x) -> f(f(x))  (divergent)
        rules = (RewriteRule(lhs=_app(0, _var(0)), rhs=_app(0, _app(0, _var(0)))),)
        term = _app(0, _app(1))
        _result, status, steps, next_step = normal_form(term, rules, max_steps=10)
        assert status == "STEP_LIMIT"
        assert steps == 10
        assert next_step is not None

    def test_normal_form_reached_exactly_at_step_limit(self):
        rule = RewriteRule(lhs=_app(0, _var(0)), rhs=_var(0))
        result = compute_normal_form(
            NormalFormRequest(
                term=_app(0, _app(1)),
                rules=(rule,),
                strategy="LEFTMOST_OUTERMOST_RULE_ORDER",
                max_steps=1,
            )
        )
        assert result.status == "NORMAL_FORM"
        assert result.term == _app(1)
        assert result.steps == 1
        assert result.next_step is None


class TestValidation:
    def test_variable_with_children_rejected(self):
        with pytest.raises(ValidationError):
            Term(is_variable=True, symbol=0, children=(_var(1),))

    def test_lhs_must_be_function(self):
        with pytest.raises(ValidationError):
            RewriteRule(lhs=_var(0), rhs=_app(1))

    def test_rhs_variables_must_be_bound_by_lhs(self):
        with pytest.raises(ValidationError, match="RHS variables"):
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(1)))

    def test_selected_position_must_exist(self):
        with pytest.raises(ValidationError, match="outside the source term"):
            RewriteStepRequest(
                term=_app(0),
                rules=(RewriteRule(lhs=_app(0), rhs=_app(1)),),
                selection={"position": [0], "rule_index": 0},
            )
