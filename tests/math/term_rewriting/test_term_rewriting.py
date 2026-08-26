"""Tests for first-order term rewriting operations."""

import tracemalloc
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math import term_rewriting
from jacobian.math.term_rewriting import _kernel as operations_module
from jacobian.math.term_rewriting._kernel import (
    _bounded_unify,
    _MaterializationBudget,
    _nonvariable_positions,
    _positions,
    _replace_at_position,
    _ResultEnvelopeError,
    _standardize_apart,
    _term_depth,
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
    verify_critical_pairs_result,
    verify_substitution_result,
)
from jacobian.math.term_rewriting._tools import TOOLS
from jacobian.math.term_rewriting.values import (
    MAX_CRITICAL_PAIR_CANDIDATES,
    MAX_CRITICAL_PAIR_RESULT_BYTES,
    MAX_CRITICAL_PAIR_RESULT_NODES,
    MAX_TERM_DEPTH,
    MAX_VARIABLE_LABEL,
    RankedSignature,
    RewriteRule,
    Term,
)


@contextmanager
def _validation_error(code: str):
    """Assert a structured owner-local Pydantic validation code."""

    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == code


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


def _chain_unary(symbol: int, length: int, leaf: Term) -> Term:
    result = leaf
    for _ in range(length):
        result = _app(symbol, result)
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


def _refreshed_binding_chain_terms() -> tuple[Term, Term]:
    """Unification terms whose stored binding is refreshed by a later bind.

    Processing pops ``w = f(x)`` first (storing ``w = f(x)``), then binds
    ``x`` to a 4095-node ground tree, which eagerly expands the stored ``w``
    binding to ``f(T)`` and must refresh its cached size. The final equation
    repeats ``w`` sixteen times under one application on both sides, so its
    precharge is ``2 * (1 + 16 * |f(T)|)`` only when the cache was refreshed;
    a stale three-node size undercharges it while the substituted equation
    materializes 65537 nodes per side.
    """
    tree = _complete_tree(3, _app(4), 2, 11)
    left = _app(
        0,
        _app(2, *([_var(52)] * 16)),
        _var(51),
        _var(52),
    )
    right = _app(
        0,
        _app(2, *([_var(52)] * 16)),
        tree,
        _app(1, _var(51)),
    )
    return left, right


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
        assert not verify_substitution_result(
            SubstitutionResult.model_validate(payload)
        )


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
        with _validation_error("term_rewriting.undeclared_symbol"):
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
        with _validation_error("term_rewriting.signature_arity"):
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
        with _validation_error("term_rewriting.signature_arity"):
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

    def test_nested_overlap_with_constant_reducts(self):
        # Nullary reducts are valid overlap results and have depth one.
        rules = (
            RewriteRule(lhs=_app(0, _app(1, _var(7))), rhs=_app(2)),
            RewriteRule(lhs=_app(1, _var(99)), rhs=_app(3)),
        )
        signature = RankedSignature(arities=(1, 1, 0, 0))
        native = critical_pairs(signature, rules)
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=rules)
        )

        assert len(native.pairs) == 1
        pair = native.pairs[0]
        assert pair.inner_reduct == _app(0, _app(3))
        assert pair.outer_reduct == _app(2)
        assert result.profile == native
        assert CriticalPairsResult.model_validate_json(result.model_dump_json()) == (
            result
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
            "max_result_term_depth": MAX_TERM_DEPTH - 1,
            "max_term_depth": MAX_TERM_DEPTH,
            "max_variable_label": MAX_VARIABLE_LABEL,
        }
        symbol = CriticalPairsRequest.model_json_schema()["$defs"]["Term"][
            "properties"
        ]["symbol"]
        assert symbol["maximum"] == MAX_VARIABLE_LABEL

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
        with _validation_error("term_rewriting.critical_pair_source"):
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
        with _validation_error("term_rewriting.critical_pair_source"):
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
        assert not verify_critical_pairs_result(
            CriticalPairsResult.model_validate(payload)
        )

    def test_duplicate_rules_are_rejected_before_trivial_root_pairs(self):
        first = RewriteRule(lhs=_app(0, _var(7)), rhs=_var(7))
        alpha_equivalent = RewriteRule(lhs=_app(0, _var(12)), rhs=_var(12))
        with _validation_error("term_rewriting.critical_pair_source"):
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
        with _validation_error("term_rewriting.critical_pair_source"):
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
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(
                signature={"arities": [1]},
                rules=(RewriteRule(lhs=lhs, rhs=_var(0)),),
            )

    def test_deep_unary_source_rejects_by_candidates_without_path_expansion(self):
        # A 20000-node unary left side fits the retained-source envelope and
        # exceeds the candidate bound by orders of magnitude, but duplicate
        # detection and candidate counting must stay linear: admission
        # returns the typed candidate rejection without materializing the
        # quadratic position-path expansion (hundreds of MB of path tuples
        # when every prefix was retained and rewalked).
        def unary_chain(function_nodes: int) -> Term:
            term = _var(0)
            for _ in range(function_nodes):
                term = _app(0, term)
            return term

        lhs = unary_chain(20_000)
        assert _term_node_count(lhs) == 20_001
        signature = RankedSignature(arities=(1,))
        rule = RewriteRule(lhs=lhs, rhs=_var(0))
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="overlap candidates"):
                critical_pairs(signature, (rule,))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 8 * 1024 * 1024
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=(rule,))

    def test_root_overlap_composed_reduct_depth_is_admission_bounded(self):
        # Two individually transport-safe rules overlap at the root: the
        # unifier splices g^14(y) into h^30(x), so the outer reduct is a
        # 45-node chain even though every source path carries at most 31
        # nodes. Admission must predict that composition from the unifier
        # and reject typedly instead of failing result canonicalization.
        def unary(symbol: int, leaf: Term, length: int) -> Term:
            term = leaf
            for _ in range(length):
                term = _app(symbol, term)
            return term

        rules = (
            RewriteRule(lhs=_app(0, _var(0)), rhs=unary(2, _var(0), 30)),
            RewriteRule(lhs=_app(0, unary(1, _var(1), 14)), rhs=_var(1)),
        )
        signature = RankedSignature(arities=(1, 1, 1))
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=rules)
        with pytest.raises(ValueError, match="result depth"):
            critical_pairs(signature, rules)

    def test_boundary_composed_reduct_admits_and_transports(self):
        # h^16 after g^13 composes exactly MAX_TERM_DEPTH - 1 nodes on its
        # only path, the deepest reduct a root overlap can serialize under
        # profile.pairs once each node's children array is counted, so this
        # family admits, replays exactly, and encodes within strict JSON
        # transport.
        def unary(symbol: int, leaf: Term, length: int) -> Term:
            term = leaf
            for _ in range(length):
                term = _app(symbol, term)
            return term

        rules = (
            RewriteRule(lhs=_app(0, _var(0)), rhs=unary(2, _var(0), 16)),
            RewriteRule(lhs=_app(0, unary(1, _var(1), 13)), rhs=_var(1)),
        )
        result = compute_critical_pairs(
            CriticalPairsRequest(
                signature=RankedSignature(arities=(1, 1, 1)), rules=rules
            )
        )
        deepest = max(
            result.profile.pairs, key=lambda pair: _term_depth(pair.outer_reduct)
        )
        assert _term_depth(deepest.outer_reduct) == MAX_TERM_DEPTH - 1
        assert _term_depth(deepest.inner_reduct) == 1
        assert len(result.model_dump_json()) <= MAX_CRITICAL_PAIR_RESULT_BYTES
        assert encode_strict_json(result.model_dump(mode="json"))
        assert (
            CriticalPairsResult.model_validate_json(result.model_dump_json()) == result
        )

    def test_one_deeper_composed_reduct_is_rejected_typed(self):
        # The same overlap shape with one more h node composes a 31-node
        # reduct, so depth alone - not candidates or nodes - must trigger
        # the typed rejection.
        def unary(symbol: int, leaf: Term, length: int) -> Term:
            term = leaf
            for _ in range(length):
                term = _app(symbol, term)
            return term

        rules = (
            RewriteRule(lhs=_app(0, _var(0)), rhs=unary(2, _var(0), 17)),
            RewriteRule(lhs=_app(0, unary(1, _var(1), 13)), rhs=_var(1)),
        )
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(
                signature=RankedSignature(arities=(1, 1, 1)), rules=rules
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
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=rules)

    def test_failed_overlap_family_within_budget_still_admits(self):
        signature, outer_rule, inner_rule = _failed_overlap_witness(7)
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=(outer_rule, inner_rule))
        )
        assert len(result.profile.candidates) == 20
        assert result.profile.pairs == ()
        assert all(not candidate.unifiable for candidate in result.profile.candidates)

    def test_wide_variable_labels_admit_by_serialized_size(self):
        # Variable labels never change the mathematical work: a single rule
        # whose only self-root overlap is excluded must admit regardless of
        # its label magnitudes, because its echoed result is far below both
        # output budgets.
        signature = term_rewriting.RankedSignature(arities=(1,))
        rule = RewriteRule(lhs=_app(0, _var(1_000_000)), rhs=_var(1_000_000))
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=(rule,))
        )
        assert result.profile.candidates == ()
        assert result.profile.pairs == ()
        assert CriticalPairsResult.model_validate_json(result.model_dump_json()) == (
            result
        )

    def test_label_serialization_width_is_charged_against_the_byte_bound(self):
        # Six bush rules repeat their wide-label leaves across the echoed
        # source. Labels stay inside the interoperable integer maximum, so
        # sixteen-digit labels push the serialized result past the byte
        # bound while a baseline-width family of the same shape still admits.
        def wide_bush(label: int) -> tuple[RewriteRule, ...]:
            def bush() -> Term:
                inner = _app(6, *([_var(label)] * 16))
                middle = _app(6, *([inner] * 16))
                return _app(6, *([middle] * 16))

            return tuple(
                RewriteRule(lhs=_app(root, _var(label)), rhs=bush())
                for root in range(6)
            )

        signature = term_rewriting.RankedSignature(arities=tuple([1] * 6 + [16]))
        result = compute_critical_pairs(
            CriticalPairsRequest(signature=signature, rules=wide_bush(123_456))
        )
        assert len(result.profile.candidates) == 30
        assert all(not candidate.unifiable for candidate in result.profile.candidates)
        assert result.profile.pairs == ()
        assert CriticalPairsResult.model_validate_json(result.model_dump_json()) == (
            result
        )
        widest = MAX_VARIABLE_LABEL
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=wide_bush(widest))
        with pytest.raises(ValueError, match="result bytes"):
            critical_pairs(signature, wide_bush(widest))

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
        with _validation_error("term_rewriting.critical_pair_source"):
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
        with _validation_error("term_rewriting.critical_pair_source"):
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
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=rules)
        with pytest.raises(ValueError, match="result nodes"):
            critical_pairs(signature, rules)

    def test_refreshed_binding_sizes_charge_the_repeated_equation_exactly(self):
        # The stored w binding is eagerly rewritten when x is bound, so its
        # cached size must be refreshed at the same moment: the final
        # sixteen-fold repetition of w is precharged from |f(T)|, and any
        # smaller allowance must reject before that equation materializes.
        left, right = _refreshed_binding_chain_terms()
        tree = _complete_tree(3, _app(4), 2, 11)
        binding = _app(1, tree)
        expected_spend = (
            # root equation predictions before any binding exists
            _term_node_count(left)
            + _term_node_count(right)
            # w = f(x): prediction plus storing the binding
            + (_term_node_count(_app(1, _var(51))) + 1)
            + _term_node_count(_app(1, _var(51)))
            # x = T: prediction plus storing the binding
            + (1 + _term_node_count(tree))
            + _term_node_count(tree)
            # eagerly expanding the stored w binding onto f(T)
            + _term_node_count(binding)
            # the repeated-w equations precharge from the REFRESHED size
            + 2 * (1 + 16 * _term_node_count(binding))
        )
        budget = _MaterializationBudget(expected_spend)
        assert _unify(left, right, budget) == {52: binding, 51: tree}
        with pytest.raises(_ResultEnvelopeError):
            _unify(left, right, _MaterializationBudget(expected_spend - 1))

    def test_stale_binding_sizes_neither_admit_nor_materialize_the_chain(self):
        # End-to-end form of the refreshed-size obligation: processing the
        # overlap equations stores W = A(y) first, then binds y = C(z),
        # which eagerly expands the stored W binding from 17 to 273 nodes.
        # Six subsequent equations each repeat W sixteen times, so their
        # honest precharges are 6 * 4369 nodes against a refreshed cache;
        # the pair work afterwards exceeds what remains and the family must
        # reject typedly. A cache left at the stale 17-node size predicts
        # only 6 * 273 there, admits the system, and then materializes the
        # uncharged expansions during replay.
        signature = RankedSignature(arities=(8, 16))
        inner_rule = RewriteRule(
            lhs=_app(
                0,
                *[_app(1, *([_var(50)] * 16)) for _ in range(6)],
                _var(52),
                _app(1, *([_var(52)] * 16)),
            ),
            rhs=_var(52),
        )
        outer_rule = RewriteRule(
            lhs=_app(
                0,
                *[_app(1, *([_var(51)] * 16)) for _ in range(6)],
                _app(1, *([_var(50)] * 16)),
                _var(51),
            ),
            rhs=_var(51),
        )
        rules = (inner_rule, outer_rule)
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
        assert peak < 8 * 1024 * 1024
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=rules)

    def test_failed_overlaps_rename_only_the_terms_unification_inspects(
        self, monkeypatch
    ):
        # Standardize-apart work is bounded by what unification actually
        # reads: a failed overlap renames only the inner left side and the
        # overlap subterm, never either right side, and duplicate detection
        # compares flat canonical keys without renaming anything.
        renamed_nodes = []
        original_rename = operations_module._rename_variables

        def counting_rename(term, renaming):
            renamed = original_rename(term, renaming)
            renamed_nodes.append(_term_node_count(renamed))
            return renamed

        monkeypatch.setattr(operations_module, "_rename_variables", counting_rename)
        big_rhs = _complete_tree(2, _var(0), 16, 2)
        outer = RewriteRule(lhs=_app(0, _var(0)), rhs=big_rhs)
        inner = RewriteRule(lhs=_app(1, _var(1)), rhs=_var(1))
        signature = RankedSignature(arities=(1, 1, 16))
        profile = critical_pairs(signature, (outer, inner))
        assert all(not candidate.unifiable for candidate in profile.candidates)
        assert profile.pairs == ()
        expected_renamed_nodes = (
            2 * 2 * (_term_node_count(outer.lhs) + _term_node_count(inner.lhs))
        )
        assert sum(renamed_nodes) == expected_renamed_nodes

    def test_standardize_apart_copies_are_charged_against_the_shared_envelope(self):
        # Three rules whose six root overlaps all unify carry near-envelope
        # right sides; replaying them materializes renamed rules, splices,
        # and both right sides per pair, so the shared envelope must reject
        # once those copy charges accumulate even though every reduct fits
        # alone.
        signature = RankedSignature(arities=(1, 16))
        big_rhs = _complete_tree(1, _var(7), 16, 3)
        small_rhs = _complete_tree(1, _var(9), 16, 2)
        rules = (
            RewriteRule(lhs=_app(0, _var(7)), rhs=big_rhs),
            RewriteRule(lhs=_app(0, _var(8)), rhs=_var(8)),
            RewriteRule(lhs=_app(0, _var(9)), rhs=small_rhs),
        )
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
        assert peak < 8 * 1024 * 1024
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest(signature=signature, rules=rules)

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


class TestDeepTermTraversal:
    def test_deep_rule_source_returns_a_profile_not_a_recursion_error(self):
        # A schema-valid 1200-deep unary right side stays far below the node
        # envelope and its single rule excludes its tautological root
        # overlap, so kernel traversal must stay iterative and return the
        # empty profile instead of raising RecursionError. The wire contract
        # separately rejects the same source with a typed depth error,
        # because strict JSON transport cannot carry a chain that deep.
        def deep_rule() -> RewriteRule:
            rhs = _var(0)
            for _ in range(1200):
                rhs = _app(0, rhs)
            return RewriteRule(lhs=_app(0, _var(0)), rhs=rhs)

        rule = deep_rule()
        assert _term_node_count(rule.rhs) == 1201
        profile = critical_pairs(RankedSignature(arities=(1,)), (rule,))
        assert profile.candidates == ()
        assert profile.pairs == ()
        with _validation_error("term_rewriting.transport_depth"):
            CriticalPairsRequest(signature={"arities": [1]}, rules=(rule,))

    def test_transport_depth_boundary_rejects_one_deeper_node_typed(self):
        # One more unary node exceeds MAX_TERM_DEPTH; direct model
        # construction fails with the typed depth error instead of relying
        # on the shared JSON transport limit to reject the payload later.
        def unary_chain(function_nodes: int) -> Term:
            term = _var(0)
            for _ in range(function_nodes):
                term = _app(0, term)
            return term

        lhs = unary_chain(MAX_TERM_DEPTH)
        with _validation_error("term_rewriting.critical_pair_source"):
            CriticalPairsRequest.model_validate(
                {
                    "signature": {"arities": [1]},
                    "rules": [
                        {
                            "lhs": lhs.model_dump(),
                            "rhs": {"is_variable": True, "symbol": 0},
                        }
                    ],
                }
            )
        request = CriticalPairsRequest(
            signature={"arities": [1]},
            rules=(RewriteRule(lhs=unary_chain(MAX_TERM_DEPTH - 1), rhs=_var(0)),),
        )
        assert len(request.rules) == 1

    def test_substitution_labels_share_the_interoperable_bound(self):
        # Substitution keys travel as JSON object keys, which bypasses the
        # integer-range check on values, so the model owns the same label
        # bound for them.
        def substitution_payload(key: int) -> dict:
            return {
                "signature": {"arities": [0]},
                "term": {"is_variable": False, "symbol": 0},
                "substitution": {"mapping": {key: {"is_variable": True, "symbol": 0}}},
            }

        admitted = SubstitutionRequest.model_validate(
            substitution_payload(MAX_VARIABLE_LABEL)
        )
        assert MAX_VARIABLE_LABEL in admitted.substitution.mapping
        with _validation_error("term_rewriting.substitution_labels"):
            SubstitutionRequest.model_validate(
                substitution_payload(MAX_VARIABLE_LABEL + 1)
            )

    def test_composed_substitution_depth_is_transport_bounded(self):
        # Each side of a substitution passes the 31-node checks separately,
        # but composing f^30(x) with x -> f^30(c) yields a 61-node chain, so
        # admission must preflight the composed result and reject typedly
        # instead of failing canonicalization after the operation ran.
        def unary_chain(length: int, leaf: Term) -> Term:
            term = leaf
            for _ in range(length):
                term = _app(0, term)
            return term

        signature = {"arities": [1, 0]}
        with _validation_error("term_rewriting.transport_depth"):
            SubstitutionRequest(
                signature=signature,
                term=unary_chain(30, _var(0)),
                substitution={"mapping": {0: unary_chain(30, _app(1))}},
            )
        boundary = SubstitutionRequest(
            signature=signature,
            term=unary_chain(15, _var(0)),
            substitution={"mapping": {0: unary_chain(15, _app(1))}},
        )
        result = compute_substitution(boundary)
        assert _term_depth(result.result) == 2 * MAX_TERM_DEPTH - 31
        assert SubstitutionResult.model_validate_json(result.model_dump_json()) == (
            result
        )

    def test_native_substitution_result_rejects_untransportable_composition(self):
        def unary_chain(length: int, leaf: Term) -> Term:
            term = leaf
            for _ in range(length):
                term = _app(0, term)
            return term

        term = unary_chain(30, _var(0))
        mapping = {0: unary_chain(30, _app(1))}
        with _validation_error("term_rewriting.transport_depth"):
            SubstitutionResult(
                signature={"arities": [1, 0]},
                term=term,
                substitution={"mapping": mapping},
                result=apply_substitution(term, mapping),
            )

    def test_spliced_rewrite_step_depth_is_transport_bounded(self):
        # f(g^15(x)) with rule g(y) -> f^30(y): every input path stays within
        # 31 nodes, but the rewritten term splices the expanded right side
        # under the f-prefix and reaches 46 nodes, so both enumeration modes
        # must reject typedly at admission.
        rules = (RewriteRule(lhs=_app(1, _var(3)), rhs=_chain_unary(0, 30, _var(3))),)
        term = _app(0, _chain_unary(1, 15, _var(0)))
        with _validation_error("term_rewriting.transport_depth"):
            RewriteStepRequest(
                signature={"arities": [1, 1]},
                term=term,
                rules=rules,
                selection={"position": [0], "rule_index": 0},
            )
        with _validation_error("term_rewriting.transport_depth"):
            RewriteStepRequest(signature={"arities": [1, 1]}, term=term, rules=rules)

    def test_boundary_spliced_rewrite_step_admits_and_transports(self):
        # The same shape with f^14 instead of f^30 composes a 30-node term,
        # one node inside the transport bound, so the step admits and its
        # exact application replays.
        rules = (RewriteRule(lhs=_app(1, _var(3)), rhs=_chain_unary(0, 14, _var(3))),)
        result = compute_rewrite_step(
            RewriteStepRequest(
                signature={"arities": [1, 1]},
                term=_app(0, _chain_unary(1, 15, _var(0))),
                rules=rules,
                selection={"position": [0], "rule_index": 0},
            )
        )
        assert len(result.applications) == 1
        assert _term_depth(result.applications[0].term) == 2 * MAX_TERM_DEPTH - 32
        assert RewriteStepResult.model_validate_json(result.model_dump_json()) == (
            result
        )

    def test_normal_form_run_depth_is_transport_bounded(self):
        # One step of g(y) -> f^30(y) under f(g^15(x)) pushes the run's term
        # to 46 nodes, so admission must simulate the declared strategy and
        # reject typedly before the operation runs.
        rules = (RewriteRule(lhs=_app(1, _var(3)), rhs=_chain_unary(0, 30, _var(3))),)
        with _validation_error("term_rewriting.transport_depth"):
            NormalFormRequest(
                signature={"arities": [1, 1]},
                term=_app(0, _chain_unary(1, 15, _var(0))),
                rules=rules,
                strategy="LEFTMOST_OUTERMOST_RULE_ORDER",
                max_steps=1,
            )

    def test_normal_form_step_limit_within_depth_admits_and_transports(self):
        # A self-looping rule keeps every intermediate term at the source
        # depth, so the bounded prefix admits with its open next step even
        # though the strategy exhausts its step budget.
        rules = (RewriteRule(lhs=_app(1, _var(3)), rhs=_app(1, _var(3))),)
        result = compute_normal_form(
            NormalFormRequest(
                signature={"arities": [1, 1]},
                term=_app(0, _chain_unary(1, 15, _var(0))),
                rules=rules,
                strategy="LEFTMOST_OUTERMOST_RULE_ORDER",
                max_steps=3,
            )
        )
        assert result.status == "STEP_LIMIT"
        assert result.steps == 3
        assert result.next_step is not None
        assert _term_depth(result.term) == MAX_TERM_DEPTH - 14
        assert NormalFormResult.model_validate_json(result.model_dump_json()) == (
            result
        )

    def test_composed_mgu_binding_depth_is_transport_bounded(self):
        # Unifying f(x, y) with f(u^16(y), u^16(c)) keeps every input path
        # inside 31 nodes, but the idempotent MGU composes x -> u^32(c),
        # whose 33-node path strict JSON transport cannot carry, so
        # admission must reject the request before execution instead of
        # letting result canonicalization fail afterwards.
        left = _app(0, _var(0), _var(1))
        right = _app(0, _chain_unary(1, 16, _var(1)), _chain_unary(1, 16, _app(2)))
        assert _term_depth(left) <= MAX_TERM_DEPTH
        assert _term_depth(right) <= MAX_TERM_DEPTH
        mgu = unify(left, right)
        assert mgu is not None
        assert _term_depth(mgu[1]) == MAX_TERM_DEPTH - 14
        assert _term_depth(mgu[0]) == 2 * MAX_TERM_DEPTH - 29
        with _validation_error("term_rewriting.transport_depth"):
            UnificationRequest(signature={"arities": [2, 1, 0]}, left=left, right=right)
        payload = {
            "signature": {"arities": [2, 1, 0]},
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
        }
        encoded = encode_strict_json(payload)
        with _validation_error("term_rewriting.transport_depth"):
            UnificationRequest.model_validate_json(encoded)
        with _validation_error("term_rewriting.transport_depth"):
            UnificationResult(
                signature={"arities": [2, 1, 0]},
                left=left,
                right=right,
                unified=True,
                substitution=mgu,
            )

    def test_boundary_composed_mgu_admits_and_transports(self):
        # With u^15 chains every composed binding stays exactly within the
        # envelope: x binds to u^30(c), whose 31-node path equals
        # MAX_TERM_DEPTH, so the request computes a typed idempotent MGU
        # that replays and round-trips.
        left = _app(0, _var(0), _var(1))
        right = _app(0, _chain_unary(1, 15, _var(1)), _chain_unary(1, 15, _app(2)))
        result = compute_unification(
            UnificationRequest(signature={"arities": [2, 1, 0]}, left=left, right=right)
        )
        assert result.unified
        assert result.substitution[1] == _chain_unary(1, 15, _app(2))
        assert _term_depth(result.substitution[1]) == MAX_TERM_DEPTH - 15
        assert _term_depth(result.substitution[0]) == MAX_TERM_DEPTH
        assert apply_substitution(left, result.substitution) == apply_substitution(
            right, result.substitution
        )
        assert UnificationResult.model_validate_json(result.model_dump_json()) == (
            result
        )

    def test_dependency_chained_mgu_is_rejected_before_materialization(self):
        # Six equations x_i = F(x_{i+1}, ..., x_{i+1}) followed by x_6 = c
        # have shallow, small inputs but expand x_0 to 17,895,697 nodes.
        # The bounded kernel must charge the growth before constructing it.
        signature = RankedSignature(arities=(7, 16, 0))
        left = _app(0, *[_var(index) for index in range(7)])
        right = _app(
            0,
            *[_app(1, *([_var(index + 1)] * 16)) for index in range(6)],
            _app(2),
        )

        with pytest.raises(ValueError, match="result nodes"):
            _bounded_unify(left, right)
        with _validation_error("term_rewriting.unification_bound"):
            UnificationRequest(signature=signature, left=left, right=right)

    def test_deep_unification_and_matching_stay_typed(self):
        def chain(length: int) -> Term:
            term = _var(7)
            for _ in range(length - 1):
                term = _app(0, term)
            return term

        left = chain(1500)
        assert unify(left, chain(1500)) == {}
        assert unify(left, _app(1)) is None
        assert match(chain(400), chain(400)) == {7: _var(7)}
        assert match(_app(1), chain(1500)) is None
        substituted = apply_substitution(chain(1500), {7: _app(2)})
        assert _term_node_count(substituted) == 1500

    def test_deep_replacement_and_positions_remain_exact(self):
        subject = _var(3)
        for _ in range(1500):
            subject = _app(0, subject)
        replacement = _app(9)
        spliced = _replace_at_position(subject, (0,) * 1500, replacement)
        assert term_at_position(spliced, (0,) * 1500) == replacement
        assert len(_positions(subject)) == 1501
        assert len(_nonvariable_positions(subject)) == 1500


class TestValidation:
    def test_public_terms_must_use_one_ranked_signature(self):
        with _validation_error("term_rewriting.signature_arity"):
            UnificationRequest(
                signature={"arities": [1]},
                left=_app(0, _var(0)),
                right=_app(0, _var(0), _var(1)),
            )

    def test_variable_with_children_rejected(self):
        with _validation_error("term_rewriting.variable_children"):
            Term(is_variable=True, symbol=0, children=(_var(1),))

    def test_lhs_must_be_function(self):
        with _validation_error("term_rewriting.lhs_variable"):
            RewriteRule(lhs=_var(0), rhs=_app(1))

    def test_rhs_variables_must_be_bound_by_lhs(self):
        with _validation_error("term_rewriting.rhs_variables"):
            RewriteRule(lhs=_app(0, _var(0)), rhs=_app(1, _var(1)))

    def test_selected_position_must_exist(self):
        with _validation_error("term_rewriting.selection_position"):
            RewriteStepRequest(
                signature={"arities": [0, 0]},
                term=_app(0),
                rules=(RewriteRule(lhs=_app(0), rhs=_app(1)),),
                selection={"position": [0], "rule_index": 0},
            )
