from __future__ import annotations

import copy
import itertools
import json
import pathlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import (
    _MappedPathFactory,
    run_verifier_in_child,
)
from jsonschema import Draft202012Validator

TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/lean-guard-scope-assurance"
)


class InputCase(TypedDict):
    id: str
    operation: str
    expression_location: str
    result_type: NotRequired[str]
    numerator: NotRequired[str]
    divisor: NotRequired[str]
    guard: NotRequired[str | None]
    guard_position: NotRequired[str]
    declaration_kind: NotRequired[str]
    ofnat_zero_equals_one: NotRequired[bool]
    lt_is_universal: NotRequired[bool]


class CaseResult(TypedDict):
    id: str
    findings: list[str]
    reason: str


class Result(TypedDict):
    cases: list[CaseResult]


class Submission(TypedDict):
    result: Result


LoaderCase = tuple[Path, Path, Callable[[], dict[str, Any] | None]]


def _input_cases() -> list[InputCase]:
    return cast(
        list[InputCase],
        json.loads((TASK_PATH / "environment/input.json").read_text())["cases"],
    )


def _derive(case: InputCase) -> CaseResult:
    """Use public semantic facts, without hidden answers or verifier helpers."""
    if case["expression_location"] == "proof_term":
        # The public statement-analysis scope excludes theorem proof terms.
        assert case["declaration_kind"] == "theorem"
        findings: list[str] = []
        reason = "PROOF_TERMS_EXCLUDED_FROM_STATEMENT_ANALYSIS"
    elif case.get("ofnat_zero_equals_one") is True:
        # In this custom type, the literal divisor one is zero. A universal
        # less-than relation also holds at zero and cannot establish nonzero.
        assert case["result_type"] == "AllZero" and case["divisor"] == "one"
        findings = ["DIVISION_BY_ZERO"]
        reason = (
            "ARBITRARY_LT_DOES_NOT_ESTABLISH_NONZERO"
            if case.get("lt_is_universal") is True
            else "LITERAL_ONE_IS_DEFINITIONALLY_ZERO_IN_CUSTOM_TYPE"
        )
    elif case.get("guard") == "b_ne_zero":
        assert case["result_type"] == "Nat" and case["divisor"] == "b"
        assert case["guard_position"] == "after_expression_binder"
        # Full proof-state scope establishes b != 0, but 3 / 2 still truncates.
        divisor = 2
        assert divisor != 0 and divmod(3, divisor) == (1, 1)
        findings = ["INTEGER_DIVISION_TRUNCATION"]
        reason = "FULL_SCOPE_GUARD_BUT_NAT_DIVISION_MAY_TRUNCATE"
    else:
        assert case["result_type"] == "Nat" and case["numerator"] == "zero"
        assert case["guard"] is None and case["divisor"] == "x"
        # Zero is divisible by every positive divisor; the unguarded x can
        # nevertheless be zero. Truncation and zero-divisor risk are separate.
        assert all(divmod(0, divisor) == (0, 0) for divisor in range(1, 8))
        findings = ["DIVISION_BY_ZERO"]
        reason = "ZERO_NUMERATOR_IS_EXACT_BUT_DIVISOR_IS_UNGUARDED"
    return {"id": case["id"], "findings": findings, "reason": reason}


def _submission() -> Submission:
    return {"result": {"cases": [_derive(case) for case in _input_cases()]}}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _schema() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((TASK_PATH / "environment/submission_schema.json").read_text()),
    )


@pytest.fixture
def loader_case(tmp_path: Path) -> LoaderCase:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copyfile(TASK_PATH / "environment/input.json", app / "input.json")
    # Import the real task-local loader with its normal default paths mapped
    # just as in the isolated child harness. Do not replace its validation.
    mapper = _MappedPathFactory(
        {"/app": app, "/tests": TASK_PATH / "tests", "/logs/verifier": logs}
    )
    with pytest.MonkeyPatch.context() as mounts:
        mounts.setattr(pathlib, "Path", mapper)
        support = load_source_module(
            "_lean_guard_case_support", TASK_PATH / "tests/verifier_support.py"
        )
    return app, logs, cast(Callable[[], dict[str, Any] | None], support.load_submission)


def _assert_boundary(
    loader_case: LoaderCase, submission: object, *, accepted: bool
) -> None:
    app, _, load_submission = loader_case
    assert Draft202012Validator(_schema()).is_valid(submission) is accepted
    _write_json(app / "submission.json", submission)
    loaded = load_submission()
    assert (loaded is not None) is accepted
    if accepted:
        assert loaded == submission


def _assert_reward(loader_case: LoaderCase, expected: float) -> None:
    app, logs, _ = loader_case
    outcome = run_verifier_in_child(task=TASK_PATH, app=app, logs=logs)
    assert outcome.details["correctness"] == expected
    assert outcome.reward == expected
    assert json.loads((logs / "reward.json").read_text()) == {"reward": expected}
    assert (
        json.loads((logs / "reward-details.json").read_text())["correctness"]
        == expected
    )


def test_generated_public_contract_is_current() -> None:
    assert not check(TASK_PATH / "tests/public_contract.json", TASK_PATH)
    Draft202012Validator.check_schema(_schema())


def _assert_public_case_alignment(cases: list[InputCase]) -> None:
    identifiers = [case["id"] for case in cases]
    assert len(identifiers) == len(set(identifiers))
    case_schema = _schema()["properties"]["result"]["properties"]["cases"]
    prefixes = case_schema["prefixItems"]
    assert len(identifiers) == case_schema["minItems"] == case_schema["maxItems"]
    assert len(prefixes) == len(identifiers)
    assert [
        prefix["allOf"][1]["properties"]["id"]["const"] for prefix in prefixes
    ] == identifiers


def test_schema_positions_match_unique_public_input_ids() -> None:
    _assert_public_case_alignment(_input_cases())


@pytest.mark.parametrize(
    "drift", ["addition", "removal", "rename", "reorder", "duplicate"]
)
def test_public_domain_guard_detects_source_drift(drift: str) -> None:
    cases = _input_cases()
    if drift == "addition":
        added = copy.deepcopy(cases[0])
        added["id"] = "new-public-case"
        cases.append(added)
    elif drift == "removal":
        cases.pop()
    elif drift == "rename":
        cases[0]["id"] = "renamed-public-case"
    elif drift == "reorder":
        cases.reverse()
    else:
        cases[-1]["id"] = cases[0]["id"]
    with pytest.raises(AssertionError):
        _assert_public_case_alignment(cases)


def test_only_public_input_order_passes_all_permutations(
    loader_case: LoaderCase,
) -> None:
    canonical = _submission()
    rows = canonical["result"]["cases"]
    accepted = 0
    for order in itertools.permutations(range(len(rows))):
        submission: Submission = {"result": {"cases": [rows[index] for index in order]}}
        expected = order == tuple(range(len(rows)))
        _assert_boundary(loader_case, submission, accepted=expected)
        accepted += expected
    assert len(list(itertools.permutations(rows))) == 120
    assert accepted == 1


def test_each_position_requires_its_public_id(loader_case: LoaderCase) -> None:
    identifiers = [case["id"] for case in _input_cases()]
    for position, identifier in itertools.product(range(5), [*identifiers, "unknown"]):
        submission = _submission()
        submission["result"]["cases"][position]["id"] = identifier
        _assert_boundary(
            loader_case, submission, accepted=identifier == identifiers[position]
        )


@pytest.mark.parametrize("position", range(5))
def test_every_prefix_retains_full_row_constraints(
    loader_case: LoaderCase, position: int
) -> None:
    canonical = _submission()
    invalid_rows: list[object] = [None, [], True, 0, "row"]
    row = cast(dict[str, object], canonical["result"]["cases"][position])
    for field in ("id", "findings", "reason"):
        invalid_rows.append({key: value for key, value in row.items() if key != field})
    deltas: tuple[dict[str, object], ...] = (
        {"id": 0},
        {"id": True},
        {"id": None},
        {"id": []},
        {"extra": 1},
        {"findings": "DIVISION_BY_ZERO"},
        {"findings": ["UNKNOWN"]},
        {"findings": [1]},
        {"findings": ["DIVISION_BY_ZERO", "DIVISION_BY_ZERO"]},
        {"reason": 1},
        {"reason": "UNKNOWN"},
    )
    for delta in deltas:
        invalid_rows.append({**row, **delta})
    for invalid_row in invalid_rows:
        submission = copy.deepcopy(canonical)
        cast(list[object], submission["result"]["cases"])[position] = invalid_row
        _assert_boundary(loader_case, submission, accepted=False)


def test_positions_do_not_publish_findings_or_reasons(loader_case: LoaderCase) -> None:
    # Every allowed classification remains a semantic claim for the verifier,
    # including false claims, rather than an answer encoded in a positional schema.
    public = json.loads((TASK_PATH / "environment/input.json").read_text())
    findings: list[str] = public["allowed_findings"]
    reasons = {row["reason"] for row in _submission()["result"]["cases"]}
    for position in range(5):
        for width in range(len(findings) + 1):
            for selected in itertools.combinations(findings, width):
                submission = _submission()
                submission["result"]["cases"][position]["findings"] = list(selected)
                _assert_boundary(loader_case, submission, accepted=True)
        for reason in reasons:
            submission = _submission()
            submission["result"]["cases"][position]["reason"] = reason
            _assert_boundary(loader_case, submission, accepted=True)


@pytest.mark.parametrize(
    "defect",
    ["unknown", "duplicate", "id-swap", "reorder", "missing", "extra", "missing-id"],
)
def test_child_rejects_identity_defects(loader_case: LoaderCase, defect: str) -> None:
    submission = _submission()
    rows = submission["result"]["cases"]
    if defect == "unknown":
        rows[0]["id"] = "unknown"
    elif defect == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif defect == "id-swap":
        rows[0]["id"], rows[1]["id"] = rows[1]["id"], rows[0]["id"]
    elif defect == "reorder":
        rows.reverse()
    elif defect == "missing":
        rows.pop()
    elif defect == "extra":
        rows.append(copy.deepcopy(rows[0]))
    else:
        del cast(dict[str, object], rows[0])["id"]
    _assert_boundary(loader_case, submission, accepted=False)
    _assert_reward(loader_case, 0.0)


def test_child_replays_independent_semantic_findings(loader_case: LoaderCase) -> None:
    _assert_boundary(loader_case, _submission(), accepted=True)
    _assert_reward(loader_case, 1.0)


@pytest.mark.parametrize(
    ("position", "findings"),
    [
        (0, []),  # Full-scope nonzero guard does not make Nat division exact.
        (1, []),  # Literal one is zero in the public custom type.
        (2, []),  # A universal less-than relation cannot rule out zero.
        (3, ["DIVISION_BY_ZERO"]),  # The proof term is excluded from analysis.
        (4, ["INTEGER_DIVISION_TRUNCATION"]),  # A zero numerator is exact.
    ],
)
def test_child_rejects_false_findings_with_correct_ids(
    loader_case: LoaderCase, position: int, findings: list[str]
) -> None:
    submission = _submission()
    submission["result"]["cases"][position]["findings"] = findings
    _assert_boundary(loader_case, submission, accepted=True)
    _assert_reward(loader_case, 0.0)


def test_child_rejects_false_reason_with_correct_ids(loader_case: LoaderCase) -> None:
    submission = _submission()
    submission["result"]["cases"][0]["reason"] = (
        "PROOF_TERMS_EXCLUDED_FROM_STATEMENT_ANALYSIS"
    )
    _assert_boundary(loader_case, submission, accepted=True)
    _assert_reward(loader_case, 0.0)


@pytest.mark.parametrize(
    "defect", ["id", "missing", "extra", "reorder", "semantic-fact"]
)
def test_public_input_drift_fails_binding(loader_case: LoaderCase, defect: str) -> None:
    _assert_boundary(loader_case, _submission(), accepted=True)
    app, _, load_submission = loader_case
    public = json.loads((app / "input.json").read_text())
    rows = public["cases"]
    if defect == "id":
        rows[0]["id"] = "changed-id"
    elif defect == "missing":
        rows.pop()
    elif defect == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "reorder":
        rows.reverse()
    else:
        rows[1]["ofnat_zero_equals_one"] = False
    _write_json(app / "input.json", public)
    assert load_submission() is None
    _assert_reward(loader_case, 0.0)
