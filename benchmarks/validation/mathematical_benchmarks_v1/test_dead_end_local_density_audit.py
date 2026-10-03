from __future__ import annotations

import copy
import itertools
import json
import math
import pathlib
import shutil
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import _MappedPathFactory
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier
from jsonschema import Draft202012Validator

TASK = "dead-end-local-density-audit"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/dead-end-local-density-audit"
)


class InputCase(TypedDict):
    case_id: str
    p: int
    b: int
    digits: list[int]


class CaseResult(TypedDict):
    case_id: str
    branch: str
    forbidden_residues: list[int]
    valid_count: int
    density_numerator: int
    density_denominator: int


class Result(TypedDict):
    cases: list[CaseResult]


class Submission(TypedDict):
    result: Result


def _input_cases() -> list[InputCase]:
    return cast(
        list[InputCase],
        json.loads((TASK_PATH / "environment" / "input.json").read_text())["cases"],
    )


def _derive(case: InputCase) -> CaseResult:
    """Solve the public linear congruences by gcd and modular inverse."""
    p, b = case["p"], case["b"]
    modulus = p * p
    divisor = math.gcd(b, modulus)
    reduced_modulus = modulus // divisor
    forbidden = {0}
    for digit in case["digits"]:
        if digit % divisor == 0:
            base = (
                -digit // divisor * pow(b // divisor, -1, reduced_modulus)
            ) % reduced_modulus
            forbidden.update(base + k * reduced_modulus for k in range(divisor))
    count = modulus - len(forbidden)
    density = Fraction(count, modulus)
    return {
        "case_id": case["case_id"],
        "branch": (
            "INVERTIBLE"
            if divisor == 1
            else "SINGLY_DIVISIBLE"
            if divisor == p
            else "SQUARE_DIVISIBLE"
        ),
        "forbidden_residues": sorted(forbidden),
        "valid_count": count,
        "density_numerator": density.numerator,
        "density_denominator": density.denominator,
    }


def _submission() -> Submission:
    return {"result": {"cases": [_derive(case) for case in _input_cases()]}}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    _rewrite(app, _submission())
    return TASK_PATH, app, logs


def _rewrite(app: Path, submission: object) -> None:
    _write_json(app / "submission.json", submission)


def _schema() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((TASK_PATH / "environment" / "submission_schema.json").read_text()),
    )


@pytest.fixture
def loader_case(tmp_path: Path) -> tuple[Path, Callable[[], dict[str, Any] | None]]:
    task, app, logs = _case(tmp_path)
    # Bind the actual task-local loader's default paths at import time, just as
    # the child harness does. No schema compiler or validator is substituted.
    mapper = _MappedPathFactory(
        {"/app": app, "/tests": task / "tests", "/logs/verifier": logs}
    )
    with pytest.MonkeyPatch.context() as mounts:
        mounts.setattr(pathlib, "Path", mapper)
        support = load_source_module(
            "_density_case_support", task / "tests" / "verifier_support.py"
        )
    return app, cast(Callable[[], dict[str, Any] | None], support.load_submission)


def _assert_reward(task: Path, app: Path, logs: Path, expected: float) -> None:
    outcome = _run_verifier(task, app, logs)
    assert outcome.details["correctness"] == expected
    assert outcome.reward == expected
    assert json.loads((logs / "reward.json").read_text()) == {"reward": expected}


def test_public_identity_assignments(
    loader_case: tuple[Path, Callable[[], dict[str, Any] | None]],
) -> None:
    app, load_submission = loader_case
    identifiers = [case["case_id"] for case in _input_cases()]
    validator = Draft202012Validator(_schema())
    accepted = 0
    assignments = 0
    for assigned in itertools.product([*identifiers, "unknown-case"], repeat=4):
        submission = _submission()
        for row, identifier in zip(
            submission["result"]["cases"], assigned, strict=True
        ):
            row["case_id"] = identifier
        expected = set(assigned) == set(identifiers)
        assert validator.is_valid(submission) is expected, assigned
        _rewrite(app, submission)
        loaded = load_submission()
        assert (loaded is not None) is expected, assigned
        if expected:
            assert loaded == submission
            accepted += 1
        assignments += 1
    assert assignments == 625
    assert accepted == 24


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_accepts_case_reordering(tmp_path: Path, order: tuple[int, ...]) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    rows = submission["result"]["cases"]
    submission["result"]["cases"] = [rows[index] for index in order]
    assert Draft202012Validator(_schema()).is_valid(submission)
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 1.0)


@pytest.mark.parametrize("pair", list(itertools.combinations(range(4), 2)))
def test_id_only_swaps_are_mathematically_false(
    tmp_path: Path, pair: tuple[int, int]
) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    left, right = (submission["result"]["cases"][index] for index in pair)
    left["case_id"], right["case_id"] = right["case_id"], left["case_id"]
    assert Draft202012Validator(_schema()).is_valid(submission)
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 0.0)


def test_rejects_boolean_integer_fields(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    submission["result"]["cases"][1]["density_numerator"] = True
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 0.0)


def test_rejects_duplicate_case_id(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    submission["result"]["cases"][1]["case_id"] = submission["result"]["cases"][0][
        "case_id"
    ]
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 0.0)


@pytest.mark.parametrize(
    "defect", ["unknown", "missing-row", "extra-row", "missing-id"]
)
def test_rejects_identity_defects(tmp_path: Path, defect: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    rows = submission["result"]["cases"]
    if defect == "unknown":
        rows[0]["case_id"] = "unknown-case"
    elif defect == "missing-row":
        rows.pop()
    elif defect == "extra-row":
        rows.append(copy.deepcopy(rows[0]))
    else:
        del cast(dict[str, object], rows[0])["case_id"]
    assert not Draft202012Validator(_schema()).is_valid(submission)
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 0.0)


@pytest.mark.parametrize(
    "defect",
    [
        "branch",
        "forbidden_residues",
        "valid_count",
        "density_numerator",
        "density_denominator",
        "unreduced-density",
        "unsorted-residues",
    ],
)
def test_rejects_wrong_math_with_complete_identity(tmp_path: Path, defect: str) -> None:
    task, app, logs = _case(tmp_path)
    submission = _submission()
    row = submission["result"]["cases"][0]
    if defect == "branch":
        row["branch"] = "SQUARE_DIVISIBLE"
    elif defect == "forbidden_residues":
        row["forbidden_residues"].remove(0)
    elif defect == "valid_count":
        row["valid_count"] += 1
    elif defect == "density_numerator":
        row["density_numerator"] += 1
    elif defect == "density_denominator":
        row["density_denominator"] += 1
    elif defect == "unreduced-density":
        row["density_numerator"] *= 2
        row["density_denominator"] *= 2
    else:
        row["forbidden_residues"].reverse()
    assert Draft202012Validator(_schema()).is_valid(submission)
    _rewrite(app, submission)
    _assert_reward(task, app, logs, 0.0)


def _assert_public_domain(schema: dict[str, Any], cases: Sequence[InputCase]) -> None:
    identifiers = [case["case_id"] for case in cases]
    declared = schema["$defs"]["case"]["properties"]["case_id"]["enum"]
    rows = schema["properties"]["result"]["properties"]["cases"]
    assert len(identifiers) == len(set(identifiers))
    assert len(declared) == len(set(declared)) == len(identifiers)
    assert set(declared) == set(identifiers)
    assert rows["minItems"] == rows["maxItems"] == len(identifiers)
    clauses = rows["allOf"]
    assert len(clauses) == len(identifiers)
    assert {
        clause["contains"]["properties"]["case_id"]["const"] for clause in clauses
    } == set(identifiers)
    for clause in clauses:
        assert clause["minContains"] == clause["maxContains"] == 1
        assert clause["contains"]["type"] == "object"
        assert clause["contains"]["required"] == ["case_id"]


def test_public_identity_domain_and_cardinality_match_input() -> None:
    cases = _input_cases()
    _assert_public_domain(_schema(), cases)
    _assert_public_domain(_schema(), list(reversed(cases)))


@pytest.mark.parametrize("drift", ["added", "removed", "renamed", "duplicate"])
def test_public_input_identity_drift_is_detected(drift: str) -> None:
    cases = _input_cases()
    if drift == "added":
        cases.append({**cases[0], "case_id": "new-public-case"})
    elif drift == "removed":
        cases.pop()
    elif drift == "renamed":
        cases[0]["case_id"] = "renamed-public-case"
    else:
        cases[1]["case_id"] = cases[0]["case_id"]
    with pytest.raises(AssertionError):
        _assert_public_domain(_schema(), cases)


def test_generated_public_artifacts_match_contract() -> None:
    assert check(TASK_PATH / "tests" / "public_contract.json", TASK_PATH) == []


@pytest.mark.parametrize(
    "artifact", ["instruction.md", "environment/submission_schema.json"]
)
def test_generated_public_artifact_drift_is_detected(
    tmp_path: Path, artifact: str
) -> None:
    task = tmp_path / TASK
    (task / "environment").mkdir(parents=True)
    for relative in ("instruction.md", "environment/submission_schema.json"):
        shutil.copyfile(TASK_PATH / relative, task / relative)
    path = task / artifact
    if artifact == "instruction.md":
        path.write_text(
            path.read_text().replace("each mathematical result", "each result")
        )
    else:
        schema = _schema()
        schema["properties"]["result"]["properties"]["cases"].pop("allOf")
        _write_json(path, schema)
    errors = check(TASK_PATH / "tests" / "public_contract.json", task)
    assert len(errors) == 1
    assert "content drift" in errors[0]
