"""Tests for first-order term rewriting operations."""

import tracemalloc

import pytest
from pydantic import ValidationError

from jacobian.math import term_rewriting
from jacobian.math.term_rewriting._models import (
    CriticalPairsRequest,
    CriticalPairsResult,
    NormalFormRequest,
    NormalFormResult,
    RewriteStepRequest,
    RewriteStepResult,
    SubstitutionRequest,
    SubstitutionResult,
    UnificationRequest,
    UnificationResult,
)
from jacobian.math.term_rewriting._operations import (
    compute_critical_pairs,
    compute_normal_form,
    compute_rewrite_step,
    compute_substitution,
    compute_unification,
)
from jacobian.math.term_rewriting._tools import TOOLS
from jacobian.math.term_rewriting.operations import (
    _MaterializationBudget,
    _nonvariable_positions,
    _replace_at_position,
    _ResultEnvelopeError,
    _standardize_apart,
    _term_node_count,
    _unify,
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)
from jacobian.math.term_rewriting.values import (
    MAX_CRITICAL_PAIR_CANDIDATES,
    MAX_CRITICAL_PAIR_RESULT_BYTES,
    MAX_CRITICAL_PAIR_RESULT_NODES,
    RankedSignature,
    RewriteRule,
    Term,
)


# Helpers
def _var(symbol: int) -> Term:
    return Term(is_variable=True, symbol=symbol)


def _app(symbol: int, *children: Term) -> Term:
    return Term(is_variable=False, symbol=symbol, children=tuple(children))


def _complete_tree(symbol: int, leaf: Term, branching: int, depth: int) -> Term:
    result = leaf
    for _ in range(depth):
        result = _app(symbol, *([result] * branching))
    return result


def _chained_overlap_witness(
    depth: int,
) -> tuple[RankedSignature, RewriteRule, RewriteRule]:
    """Signature and two root-overlapping rules with a chained idempotent MGU.

    Overlap slot ``k`` opposes one bare chain variable to a tripled next-chain
    variable, so unifying the left sides binds the first variable through
    ``depth`` nested substitutions while every rule side stays within the
    16-node bound (both depth-6 sides are exactly 16 nodes). Each right side
    repeats its chain variable 15 times, so each root orientation expands to
    ``1 + 15 * ((3 ** (depth + 1) - 1) // 2)`` reduct nodes.
    """
    left_slots: list[Term] = []
    right_slots: list[Term] = []
    for level in range(depth):
        if level % 2 == 0:
            left_slots.append(_var(100 + level))
            right_slots.append(_app(1, *([_var(101 + level)] * 3)))
        else:
            left_slots.append(_app(1, *([_var(101 + level)] * 3)))
            right_slots.append(_var(100 + level))
    signature = RankedSignature(arities=(depth, 3, 15))
    return (
        signature,
        RewriteRule(lhs=_app(0, *left_slots), rhs=_app(2, *([_var(100)] * 15))),
        RewriteRule(lhs=_app(0, *right_slots), rhs=_app(2, *([_var(101)] * 15))),
    )


def _failed_overlap_witness(
    depth: int,
) -> tuple[RankedSignature, RewriteRule, RewriteRule]:
    """Signature and two root-overlapping rules whose every overlap fails.

    The shared root carries a leading constant clash that the kernel examines
    only after every chained binding has been built, so each overlap charges
    its complete dependency-driven expansion before returning ``None``. Both
    depth-8 root orientations charge 36914 nodes, so each fits alone in a
    fresh envelope while two already exceed it together.
    """
    left_slots: list[Term] = [_app(2)]
    right_slots: list[Term] = [_app(3)]
    for level in range(depth):
        if level % 2 == 0:
            left_slots.append(_var(100 + level))
            right_slots.append(_app(1, *([_var(101 + level)] * 3)))
        else:
            left_slots.append(_app(1, *([_var(101 + level)] * 3)))
            right_slots.append(_var(100 + level))
    signature = RankedSignature(arities=(depth + 1, 3, 0, 0))
    return (
        signature,
        RewriteRule(lhs=_app(0, *left_slots), rhs=_var(100)),
        RewriteRule(lhs=_app(0, *right_slots), rhs=_var(101)),
    )


def _repeated_chain_overlap_witness(
    repeats: int,
) -> tuple[RankedSignature, RewriteRule, RewriteRule]:
    """Signature and two root-overlapping rules whose MGU chains a binding to
    4369 nodes and then meets a pending equation repeating it ``repeats`` times.

    Three arity-16 chain slots grow the first variable through nested
    substitutions exactly as :func:`_chained_overlap_witness` does; the leading
    slot then repeats that variable under one arity-16 application, so its
    pending-equation expansion is ``repeats * 4369 + 1`` nodes and must be
    predicted and charged before the substituted equation is constructed.
    """
    left_slots: list[Term] = []
    right_slots: list[Term] = []
    for level in range(3):
        if level % 2 == 0:
            left_slots.append(_var(100 + level))
            right_slots.append(_app(1, *([_var(101 + level)] * 16)))
        else:
            left_slots.append(_app(1, *([_var(101 + level)] * 16)))
            right_slots.append(_var(100 + level))
    signature = RankedSignature(arities=(4, 16))
    return (
        signature,
        RewriteRule(
            lhs=_app(0, _app(1, *([_var(100)] * repeats)), *left_slots),
            rhs=_var(100),
        ),
        RewriteRule(
            lhs=_app(0, _app(1, *([_var(300)] * repeats)), *right_slots),
            rhs=_var(300),
        ),
    )


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "term_rewriting.critical_pairs.compute",
        "term_rewriting.matching.compute",
        "term_rewriting.unification.compute",
        "term_rewriting.rewrite_step.compute",
    }


def test_substitution_and_normalization_remain_native() -> None:
    variable = term_rewriting.Term(is_variable=True, symbol=0)
    constant = term_rewriting.Term(symbol=1)
    assert term_rewriting.apply_substitution(variable, {0: constant}) == constant
    rule = term_rewriting.RewriteRule(lhs=_app(0, variable), rhs=variable)
    result, status, steps, next_step = term_rewriting.normal_form(
        _app(0, constant), (rule,), max_steps=1
    )
    assert (result, status, steps, next_step) == (
        constant,
        "NORMAL_FORM",
        1,
        None,
    )


def test_native_choice_and_bound_validation_is_explicit() -> None:
    rule = RewriteRule(lhs=_app(0), rhs=_app(1))
    with pytest.raises(ValueError, match="rule_index"):
        selected_rewrite_step(_app(0), (rule,), (), 1)
    with pytest.raises(ValueError, match="max_steps"):
        normal_form(_app(0), (rule,), max_steps=-1)


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

    def test_result_is_bound_to_the_source_substitution(self):
        result = compute_substitution(
            SubstitutionRequest(
                signature={"arities": [1, 0]},
                term=_app(0, _var(0)),
                substitution={"mapping": {0: _app(1)}},
            )
        )
        payload = result.model_dump()
        payload["result"] = _app(1).model_dump()
        with pytest.raises(ValidationError, match="not bound"):
            SubstitutionResult.model_validate(payload)


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
        wire_result = compute_unification(
            UnificationRequest(signature={"arities": [2, 0, 0]}, left=left, right=right)
        )
        assert wire_result.unified
        assert wire_result.substitution == result

    def test_result_rejects_terms_outside_its_ranked_signature(self):
        with pytest.raises(ValidationError, match="undeclared"):
            UnificationResult(
                signature={"arities": [0]},
                left=_app(1),
                right=_app(1),
                unified=True,
                substitution={},
            )


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
        result = compute_rewrite_step(
            RewriteStepRequest(
                signature={"arities": [1, 1, 1, 0]}, term=term, rules=rules
            )
        )
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
                signature={"arities": [1, 1, 1, 0]},
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
                signature={"arities": [0, 0, 0]},
                term=_app(2),
                rules=rules,
                selection={"position": [], "rule_index": 0},
            )
        )
        assert result.scope == "SELECTED_STEP"
        assert result.applications == ()

    def test_result_reestablishes_signature_membership(self):
        result = compute_rewrite_step(
            RewriteStepRequest(
                signature={"arities": [1, 0]},
                term=_app(0, _app(1)),
                rules=(RewriteRule(lhs=_app(0, _var(0)), rhs=_var(0)),),
            )
        )
        payload = result.model_dump()
        payload["signature"] = {"arities": [0, 0]}
        with pytest.raises(ValidationError, match="child count"):
            RewriteStepResult.model_validate(payload)


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
                signature={"arities": [1, 0]},
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

        payload = result.model_dump()
        payload["signature"] = {"arities": [0, 0]}
        with pytest.raises(ValidationError, match="child count"):
            NormalFormResult.model_validate(payload)


class TestCriticalPairs:
    def test_empty_rewrite_system_yields_the_empty_profile(self):
        # An empty finite TRS has an unambiguously empty overlap ledger and
        # pair family, so admission must accept it without work.
        signature = term_rewriting.RankedSignature(arities=(1,))
        profile = critical_pairs(signature, ())
        assert profile.candidates == ()
        assert profile.pairs == ()

        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1]}, rules=())
        )
        assert result.rules == ()
        assert result.profile.candidates == ()
        assert result.profile.pairs == ()
        assert (
            CriticalPairsResult.model_validate_json(result.model_dump_json()) == result
        )

    def test_nested_overlap_records_both_peak_reducts(self):
        # f(g(x)) -> x overlaps g(y) -> y at the nested g-position.
        rules = (
            RewriteRule(lhs=_app(0, _app(1, _var(7))), rhs=_var(7)),
            RewriteRule(lhs=_app(1, _var(99)), rhs=_var(99)),
        )
        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1, 1]}, rules=rules)
        )
        assert len(result.profile.candidates) == 4
        candidate = result.profile.candidates[2]
        assert (
            candidate.outer_rule_index,
            candidate.inner_rule_index,
            candidate.position,
            candidate.unifiable,
        ) == (
            0,
            1,
            (0,),
            True,
        )
        assert candidate.outer_variable_renaming == {7: 0}
        assert candidate.inner_variable_renaming == {99: 1}
        assert len(result.profile.pairs) == 1
        pair = result.profile.pairs[0]
        assert pair.candidate_index == 2
        assert pair.outer_variable_renaming == {7: 0}
        assert pair.inner_variable_renaming == {99: 1}
        assert pair.substitution == {0: _var(1)}
        assert pair.inner_reduct == _app(0, _var(1))
        assert pair.outer_reduct == _var(1)
        assert (
            CriticalPairsResult.model_validate_json(result.model_dump_json()) == result
        )

    def test_standardize_apart_makes_variable_labels_irrelevant(self):
        first = (
            RewriteRule(lhs=_app(0, _app(1, _var(7))), rhs=_var(7)),
            RewriteRule(lhs=_app(1, _var(99)), rhs=_var(99)),
        )
        renamed = (
            RewriteRule(lhs=_app(0, _app(1, _var(1000))), rhs=_var(1000)),
            RewriteRule(lhs=_app(1, _var(4)), rhs=_var(4)),
        )
        signature = term_rewriting.RankedSignature(arities=(1, 1))
        first_profile = critical_pairs(signature, first)
        renamed_profile = critical_pairs(signature, renamed)
        assert tuple(
            (candidate.outer_rule_index, candidate.inner_rule_index, candidate.position)
            for candidate in first_profile.candidates
        ) == tuple(
            (candidate.outer_rule_index, candidate.inner_rule_index, candidate.position)
            for candidate in renamed_profile.candidates
        )
        assert tuple(
            (
                pair.candidate_index,
                pair.substitution,
                pair.inner_reduct,
                pair.outer_reduct,
            )
            for pair in first_profile.pairs
        ) == tuple(
            (
                pair.candidate_index,
                pair.substitution,
                pair.inner_reduct,
                pair.outer_reduct,
            )
            for pair in renamed_profile.pairs
        )

    def test_self_root_overlap_is_tautological_and_excluded(self):
        rule = RewriteRule(lhs=_app(0, _var(0)), rhs=_var(0))
        assert (
            critical_pairs(term_rewriting.RankedSignature(arities=(1,)), (rule,)).pairs
            == ()
        )

    def test_nonunifiable_overlap_produces_no_pair(self):
        rules = (
            RewriteRule(lhs=_app(0, _app(1)), rhs=_app(2)),
            RewriteRule(lhs=_app(2), rhs=_app(1)),
        )
        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1, 0, 0]}, rules=rules)
        )
        assert result.profile.pairs == ()
        assert len(result.profile.candidates) == 4
        assert all(not candidate.unifiable for candidate in result.profile.candidates)

    def test_schema_describes_critical_pair_admission_limits(self):
        rules = CriticalPairsRequest.model_json_schema()["properties"]["rules"]
        assert rules["maxItems"] == 8
        assert rules["x-jacobian-bounds"] == {
            "max_overlap_candidates": 32,
            "max_result_bytes": 4 * 1024 * 1024,
            "max_result_nodes": 42_752,
            "max_variable_id": 999_999,
        }

    def test_retained_rules_pay_for_the_result_envelope_without_overlaps(self):
        # f(x) -> R excludes its single tautological root overlap, so the
        # overlap ledger is empty and only the retained rules carry the
        # envelope cost of the result.
        signature = term_rewriting.RankedSignature(arities=(1, 16))
        rule = RewriteRule(
            lhs=_app(0, _var(0)),
            rhs=_complete_tree(1, _var(0), 16, 4),
        )
        assert _term_node_count(rule.rhs) > MAX_CRITICAL_PAIR_RESULT_NODES
        assert len(_nonvariable_positions(rule.lhs)) == 1
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=(rule,))
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, (rule,))

    def test_large_retained_rules_admit_within_the_shared_envelope(self):
        signature = term_rewriting.RankedSignature(arities=(1, 15))
        rule = RewriteRule(
            lhs=_app(0, _var(0)),
            rhs=_complete_tree(1, _var(0), 15, 3),
        )
        assert _term_node_count(rule.rhs) < MAX_CRITICAL_PAIR_RESULT_NODES
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=(rule,))
        )
        assert result.profile.candidates == ()
        assert result.profile.pairs == ()
        assert (
            CriticalPairsResult.model_validate_json(result.model_dump_json()) == result
        )

    def test_retained_source_bound_precedes_duplicate_detection(self):
        # Canonicalizing a bushy rule allocates a full copy plus its JSON
        # serialization, so the retained-source bound must reject an oversized
        # system before the duplicate check ever reconstructs one.
        signature = term_rewriting.RankedSignature(arities=(1, 16))
        rule = RewriteRule(
            lhs=_app(0, _var(0)),
            rhs=_complete_tree(1, _var(0), 16, 4),
        )
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=(rule, rule))
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, (rule, rule))

    def test_result_replays_exact_critical_pair_family(self):
        rules = (
            RewriteRule(lhs=_app(0, _app(1, _var(0))), rhs=_var(0)),
            RewriteRule(lhs=_app(1, _var(1)), rhs=_var(1)),
        )
        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1, 1]}, rules=rules)
        )
        payload = result.model_dump()
        payload["profile"]["pairs"][0]["outer_reduct"] = _app(0, _var(1)).model_dump()
        with pytest.raises(ValidationError, match="do not replay"):
            CriticalPairsResult.model_validate(payload)

    def test_duplicate_rules_are_rejected_before_trivial_root_pairs(self):
        first = RewriteRule(lhs=_app(0, _var(7)), rhs=_var(7))
        alpha_equivalent = RewriteRule(lhs=_app(0, _var(12)), rhs=_var(12))
        with pytest.raises(ValidationError, match="duplicate-free"):
            CriticalPairsRequest(
                signature={"arities": [1]}, rules=(first, alpha_equivalent)
            )

    def test_candidate_work_is_preflight_bounded(self):
        def unary_chain(variable: int, length: int) -> Term:
            result = _var(variable)
            for _ in range(length):
                result = _app(0, result)
            return result

        rules = tuple(
            RewriteRule(lhs=_app(0, _var(index)), rhs=unary_chain(index, index))
            for index in range(7)
        )
        with pytest.raises(ValidationError, match="overlap candidates"):
            CriticalPairsRequest(signature={"arities": [1]}, rules=rules)

    def test_exact_candidate_boundary_has_bounded_complete_output(self):
        def unary_chain(variable: int, length: int) -> Term:
            result = _var(variable)
            for _ in range(length):
                result = _app(0, result)
            return result

        rules = (
            RewriteRule(lhs=unary_chain(0, 3), rhs=_var(0)),
            RewriteRule(lhs=unary_chain(1, 2), rhs=_var(1)),
            RewriteRule(lhs=unary_chain(2, 2), rhs=unary_chain(2, 1)),
            RewriteRule(lhs=unary_chain(3, 2), rhs=unary_chain(3, 2)),
        )
        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1]}, rules=rules)
        )
        assert len(result.profile.candidates) == 32
        assert len(result.profile.pairs) == 32
        assert len(result.model_dump_json()) <= MAX_CRITICAL_PAIR_RESULT_BYTES

    def test_long_rules_reach_the_result_sensitive_preflight(self):
        # f^16(x) -> x has a 17-node left side, yet its complete overlap
        # family is 15 non-root self-overlaps with a small exact profile.
        lhs = _var(0)
        for _ in range(16):
            lhs = _app(0, lhs)
        assert _term_node_count(lhs) == 17
        rule = RewriteRule(lhs=lhs, rhs=_var(0))
        result = compute_critical_pairs(
            CriticalPairsRequest(signature={"arities": [1]}, rules=(rule,))
        )
        assert len(result.profile.candidates) == 15
        assert len(result.profile.pairs) == 15
        assert all(candidate.unifiable for candidate in result.profile.candidates)
        assert all(pair.substitution for pair in result.profile.pairs)
        assert (
            CriticalPairsResult.model_validate_json(result.model_dump_json()) == result
        )

    def test_oversized_rule_work_stays_derived_bound_bounded(self):
        lhs = _var(0)
        for _ in range(64):
            lhs = _app(0, lhs)
        with pytest.raises(ValidationError, match="overlap candidates"):
            CriticalPairsRequest(
                signature={"arities": [1]},
                rules=(RewriteRule(lhs=lhs, rhs=_var(0)),),
            )

    def test_failed_overlap_charges_commit_to_the_shared_budget(self):
        signature, outer_rule, inner_rule = _failed_overlap_witness(8)
        rules = (outer_rule, inner_rule)
        candidates = len(rules) * sum(
            len(_nonvariable_positions(rule.lhs)) for rule in rules
        ) - len(rules)
        assert candidates <= MAX_CRITICAL_PAIR_CANDIDATES
        for outer_index, outer in enumerate(rules):
            for position in _nonvariable_positions(outer.lhs):
                for inner_index, inner in enumerate(rules):
                    if outer_index == inner_index and not position:
                        continue
                    standardized_outer, standardized_inner, _, _ = _standardize_apart(
                        outer, inner
                    )
                    budget = _MaterializationBudget(MAX_CRITICAL_PAIR_RESULT_NODES)
                    substitution = _unify(
                        standardized_inner.lhs,
                        term_at_position(standardized_outer.lhs, position),
                        budget,
                    )
                    assert substitution is None
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, rules)
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=rules)

    def test_failed_overlap_family_within_budget_still_admits(self):
        signature, outer_rule, inner_rule = _failed_overlap_witness(7)
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=(outer_rule, inner_rule))
        )
        assert len(result.profile.candidates) == 20
        assert result.profile.pairs == ()
        assert all(not candidate.unifiable for candidate in result.profile.candidates)

    def test_native_operation_enforces_the_same_signature_and_work_bounds(self):
        signature = term_rewriting.RankedSignature(arities=(1,))
        rule = RewriteRule(lhs=_app(0, _var(1_000_000)), rhs=_var(1_000_000))
        with pytest.raises(ValueError, match="variable IDs"):
            critical_pairs(signature, (rule,))

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_chained_binding_family_admits_and_replays_within_envelope(
        self, depth: int
    ):
        signature, outer_rule, inner_rule = _chained_overlap_witness(depth)
        rules = (outer_rule, inner_rule)
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=rules)
        )
        materialized_nodes = sum(
            _term_node_count(term)
            for pair in result.profile.pairs
            for term in (
                *pair.substitution.values(),
                pair.inner_reduct,
                pair.outer_reduct,
            )
        )
        assert materialized_nodes <= MAX_CRITICAL_PAIR_RESULT_NODES
        assert CriticalPairsResult.model_validate_json(result.model_dump_json()) == (
            result
        )
        for pair in result.profile.pairs:
            candidate = result.profile.candidates[pair.candidate_index]
            standardized_outer, standardized_inner, _, _ = _standardize_apart(
                rules[candidate.outer_rule_index], rules[candidate.inner_rule_index]
            )
            substitution = unify(
                standardized_inner.lhs,
                term_at_position(standardized_outer.lhs, candidate.position),
            )
            assert substitution is not None
            assert pair.substitution == substitution
            spliced = _replace_at_position(
                standardized_outer.lhs, candidate.position, standardized_inner.rhs
            )
            assert pair.inner_reduct == apply_substitution(spliced, substitution)
            assert pair.outer_reduct == apply_substitution(
                standardized_outer.rhs, substitution
            )

    def test_dependency_chained_mgu_expansion_is_preflight_bounded(self):
        signature, outer_rule, inner_rule = _chained_overlap_witness(6)
        rules = (outer_rule, inner_rule)
        assert all(
            _term_node_count(side) == 16
            for rule in rules
            for side in (rule.lhs, rule.rhs)
        )
        candidates = len(rules) * sum(
            len(_nonvariable_positions(rule.lhs)) for rule in rules
        ) - len(rules)
        assert 0 < candidates <= MAX_CRITICAL_PAIR_CANDIDATES
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=rules)
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, rules)

    def test_deep_chained_binding_growth_is_preflight_bounded(self):
        signature, outer_rule, inner_rule = _chained_overlap_witness(9)
        rules = (outer_rule, inner_rule)
        candidates = len(rules) * sum(
            len(_nonvariable_positions(rule.lhs)) for rule in rules
        ) - len(rules)
        assert 0 < candidates <= MAX_CRITICAL_PAIR_CANDIDATES
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=rules)
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, rules)

    def test_repeated_binding_growth_is_preflight_bounded(self):
        # The chained MGU binds the first variable to a 4369-node term and the
        # leading slot then repeats it sixteen times, so substituting that one
        # pending equation expands to 69905 nodes. Prediction must charge the
        # expansion before the substituted equation is constructed.
        signature, outer_rule, inner_rule = _repeated_chain_overlap_witness(16)
        rules = (outer_rule, inner_rule)
        candidates = len(rules) * sum(
            len(_nonvariable_positions(rule.lhs)) for rule in rules
        ) - len(rules)
        assert 0 < candidates <= MAX_CRITICAL_PAIR_CANDIDATES
        with pytest.raises(ValidationError, match="result nodes"):
            CriticalPairsRequest(signature=signature, rules=rules)
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, rules)

    def test_budgeted_unification_charges_exactly_the_materialized_sizes(self):
        for depth in range(1, 6):
            _, outer_rule, inner_rule = _chained_overlap_witness(depth)
            standardized_outer, standardized_inner, _, _ = _standardize_apart(
                outer_rule, inner_rule
            )
            inner_lhs = standardized_inner.lhs
            overlap = term_at_position(standardized_outer.lhs, ())
            expected = unify(inner_lhs, overlap)
            assert expected is not None
            budget = _MaterializationBudget(MAX_CRITICAL_PAIR_RESULT_NODES)
            assert _unify(inner_lhs, overlap, budget) == expected
            spent = MAX_CRITICAL_PAIR_RESULT_NODES - budget.remaining
            tight = _MaterializationBudget(spent)
            assert _unify(inner_lhs, overlap, tight) == expected
            with pytest.raises(_ResultEnvelopeError):
                _unify(inner_lhs, overlap, _MaterializationBudget(spent - 1))

    def test_budgeted_unification_prepends_equation_growth_exactly(self):
        _, outer_rule, inner_rule = _repeated_chain_overlap_witness(2)
        standardized_outer, standardized_inner, _, _ = _standardize_apart(
            outer_rule, inner_rule
        )
        inner_lhs = standardized_inner.lhs
        overlap = term_at_position(standardized_outer.lhs, ())
        expected = unify(inner_lhs, overlap)
        assert expected is not None
        budget = _MaterializationBudget(MAX_CRITICAL_PAIR_RESULT_NODES)
        assert _unify(inner_lhs, overlap, budget) == expected
        spent = MAX_CRITICAL_PAIR_RESULT_NODES - budget.remaining
        tight = _MaterializationBudget(spent)
        assert _unify(inner_lhs, overlap, tight) == expected
        with pytest.raises(_ResultEnvelopeError):
            _unify(inner_lhs, overlap, _MaterializationBudget(spent - 1))

    def test_repeated_bound_variable_expansion_is_charged_before_materialization(
        self,
    ):
        # Twelve alternating overlap slots build a dependency-chain MGU whose
        # top stored binding grows to an 8191-node term while both rule sides
        # stay small. The last processed slot opposes a sixteen-fold
        # repetition of that bound variable to a bare variable, so predicting
        # both equation expansions must reject the overlap before its
        # ~131k-node substituted equation is ever constructed.
        inner_cascade: list[Term] = []
        outer_cascade: list[Term] = []
        for level in range(6):
            inner_symbol = 101 + 2 * level
            outer_symbol = 102 + 2 * level
            next_outer_symbol = 104 + 2 * level
            # Bare outer variables alternate with doubled inner symbols, so
            # each later binding eagerly expands every earlier stored one.
            inner_cascade.append(_app(1, *([_var(inner_symbol)] * 2)))
            outer_cascade.append(_var(outer_symbol))
            inner_cascade.append(_var(inner_symbol))
            outer_cascade.append(_app(1, *([_var(next_outer_symbol)] * 2)))
        signature = RankedSignature(arities=(16, 2))
        inner_rule = RewriteRule(
            lhs=_app(
                0,
                _var(126),
                *reversed(inner_cascade),
                *[_var(index) for index in (120, 121, 122)],
            ),
            rhs=_var(101),
        )
        outer_rule = RewriteRule(
            lhs=_app(
                0,
                _app(0, *([_var(102)] * 16)),
                *reversed(outer_cascade),
                *[_var(index) for index in (123, 124, 125)],
            ),
            rhs=_var(102),
        )
        rules = (outer_rule, inner_rule)
        candidates = len(rules) * sum(
            len(_nonvariable_positions(rule.lhs)) for rule in rules
        ) - len(rules)
        assert 0 < candidates <= MAX_CRITICAL_PAIR_CANDIDATES

        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="result nodes"):
                critical_pairs(signature, rules)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The admitted dependency chain itself costs ~33k charged nodes and
        # keeps ~9.5k envelope nodes free; the rejected sixteen-fold
        # repetition of the 8191-node top binding must never exist, so the
        # traced peak stays far below the tens of MB an uncharged
        # materialization would reach.
        assert peak < 16 * 1024 * 1024

    def test_bushy_retained_rhs_is_rejected_before_canonical_copy(self):
        # One rule whose schema-valid RHS is a complete arity-16 tree of ~70k
        # nodes: the retained-source node bound must reject it during request
        # validation before any canonical copy or JSON serialization exists.
        signature = RankedSignature(arities=(1, 16))
        rule = RewriteRule(
            lhs=_app(0, _var(0)),
            rhs=_complete_tree(1, _var(0), 16, 4),
        )
        assert _term_node_count(rule.rhs) > MAX_CRITICAL_PAIR_RESULT_NODES
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="result nodes"):
                critical_pairs(signature, (rule,))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < MAX_CRITICAL_PAIR_RESULT_BYTES // 8


class TestValidation:
    def test_public_terms_must_use_one_ranked_signature(self):
        with pytest.raises(ValidationError, match="ranked signature"):
            UnificationRequest(
                signature={"arities": [1]},
                left=_app(0, _var(0)),
                right=_app(0, _var(0), _var(1)),
            )

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
                signature={"arities": [0, 0]},
                term=_app(0),
                rules=(RewriteRule(lhs=_app(0), rhs=_app(1)),),
                selection={"position": [0], "rule_index": 0},
            )
