import importlib.util
import json
import sys
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/agent-workflow-v1/triplewise-empty-extremal-audit/tests/verifier.py"
)


def module():
    task_tests = str(VERIFIER.parent)
    previous_support = sys.modules.pop("verifier_support", None)
    sys.path.insert(0, task_tests)
    try:
        spec = importlib.util.spec_from_file_location("triplewise_verifier", VERIFIER)
        assert spec and spec.loader
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(task_tests)
        sys.modules.pop("verifier_support", None)
        if previous_support is not None:
            sys.modules["verifier_support"] = previous_support


def test_valid_alternative_matching():
    family = [[], *[[i] for i in range(7)], [0, 2], [1, 4], [3, 6]]
    assert module().valid_family(7, family)


def test_source_bound_cannot_be_attained():
    family = [[], *[[i] for i in range(7)], [0, 1], [2, 3], [4, 5]]
    assert len(family) == 11
    assert len(family) < 2 * 7


def _prepared_submission(tmp_path, mutate):
    task, app, logs = support._prepare_case(
        tmp_path, "triplewise-empty-extremal-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    mutate(submission, app)
    support._write_json(submission_path, submission)
    return support._run_verifier(task, app, logs)


def test_non_integer_family_elements_are_rejected():
    assert not module().valid_family(7, [[], [float("nan")]])
    assert not module().valid_family(7, [[], [1.0]])
    assert not module().valid_family(7, [[], [True]])


def test_extra_duplicate_probe_is_rejected(tmp_path):
    def mutate(submission, _app):
        submission["result"]["constructions"].append(
            submission["result"]["constructions"][0]
        )

    reward = _prepared_submission(tmp_path, mutate)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_nested_extra_fields_are_rejected(tmp_path):
    def mutate(submission, _app):
        submission["result"]["extra"] = "undeclared"

    reward = _prepared_submission(tmp_path, mutate)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_protocol_failures_do_not_hide_mathematical_correctness(tmp_path):
    def mutate(submission, _app):
        submission["limitations"] = []

    reward = _prepared_submission(tmp_path, mutate)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_negated_scope_is_rejected(tmp_path):
    def mutate(submission, _app):
        submission["scope"] = (
            "does not cover distinct subsets with triplewise empty intersection"
        )

    reward = _prepared_submission(tmp_path, mutate)
    assert reward["correctness"] == 1.0
    assert reward["scope_accuracy"] == 0.0
    assert reward["reward"] == 0.0


def test_unrelated_digest_bound_evidence_is_rejected(tmp_path):
    def mutate(submission, app):
        evidence = app / "evidence" / "answer.txt"
        evidence.write_text("This unrelated text is nonempty but proves nothing.\n")
        submission["evidence"][0]["sha256"] = support._digest(evidence)

    reward = _prepared_submission(tmp_path, mutate)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0
