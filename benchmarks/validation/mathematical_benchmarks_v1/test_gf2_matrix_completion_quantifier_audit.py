import copy
import json
import shutil
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling.public_contract import (
    check,
    load_contract,
    render_submission_schema,
)
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import VerifierOutput
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier
from jsonschema import Draft202012Validator

TASK = "gf2-matrix-completion-quantifier-audit"
TASK_ROOT = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
PUBLIC_INPUT = json.loads((TASK_ROOT / "environment/input.json").read_text())
DIMENSIONS = tuple(
    range(PUBLIC_INPUT["dimension_bounds"][0], PUBLIC_INPUT["dimension_bounds"][1] + 1)
)
MATRIX_FIELDS = ("pattern", "low_rank_completion", "full_rank_completion")
PUBLIC_SCHEMA = json.loads(
    (TASK_ROOT / "environment/submission_schema.json").read_text()
)
PUBLIC_VALIDATOR = Draft202012Validator(PUBLIC_SCHEMA)


def _construction(n: int) -> dict[str, Any]:
    # I_n + E_01 is asymmetric, band-supported, and invertible upper triangular.
    # It has n+1 forced ones. The all-ones completion has rank exactly one.
    pattern = [[int(i == j or (i, j) == (0, 1)) for j in range(n)] for i in range(n)]
    return {
        "result": {
            "dimension": n,
            "defects": copy.deepcopy(
                PUBLIC_SCHEMA["$defs"]["result"]["properties"]["defects"]["const"]
            ),
            "pattern": pattern,
            "low_rank_completion": [[1] * n for _ in range(n)],
            "full_rank_completion": copy.deepcopy(pattern),
        }
    }


@pytest.fixture
def runtime_support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    support = load_source_module(
        "_gf2_verifier_support", TASK_ROOT / "tests/verifier_support.py"
    )
    # Redirect only the task contract path; exercise the real runtime loader.
    monkeypatch.setattr(
        support._load_public_contract,
        "__defaults__",
        (TASK_ROOT / "tests/public_contract.json",),
    )
    return support


def _assert_protocol_rejected(
    tmp_path: Path, runtime_support: ModuleType, submission: dict[str, Any]
) -> None:
    assert not PUBLIC_VALIDATOR.is_valid(submission)
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission))
    assert runtime_support.load_submission(path, require_input_binding=False) is None


def _verify(tmp_path: Path, submission: dict[str, Any]) -> VerifierOutput:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_ROOT / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(TASK_ROOT, app, logs)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_public_construction_passes_at_every_dimension(
    tmp_path: Path, runtime_support: ModuleType, n: int
) -> None:
    submission = _construction(n)
    PUBLIC_VALIDATOR.validate(submission)
    assert runtime_support.submission_matches_public_schema(submission)
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission))
    assert (
        runtime_support.load_submission(path, require_input_binding=False) == submission
    )
    assert _verify(tmp_path, submission).reward == 1.0


@pytest.mark.parametrize("n,field", tuple(product(DIMENSIONS, MATRIX_FIELDS)))
def test_each_matrix_shape_must_match_dimension(
    tmp_path: Path, runtime_support: ModuleType, n: int, field: str
) -> None:
    # All 7 dimensions x 3 matrices x 48 wrong public row/column pairs = 1,008.
    for rows, cols in product(DIMENSIONS, repeat=2):
        if (rows, cols) == (n, n):
            continue
        submission = _construction(n)
        submission["result"][field] = [[0] * cols for _ in range(rows)]
        _assert_protocol_rejected(tmp_path, runtime_support, submission)


def test_reported_mixed_matrix_shapes_are_rejected_before_replay(
    tmp_path: Path, runtime_support: ModuleType
) -> None:
    submission = _construction(8)
    for field, (rows, cols) in zip(
        MATRIX_FIELDS, ((14, 9), (8, 14), (12, 8)), strict=True
    ):
        submission["result"][field] = [[0] * cols for _ in range(rows)]
    _assert_protocol_rejected(tmp_path, runtime_support, submission)
    assert not runtime_support.submission_matches_public_schema(submission)
    assert _verify(tmp_path, submission).reward == 0


@pytest.mark.parametrize(
    "n,field,change", tuple(product(DIMENSIONS, MATRIX_FIELDS, (-1, 1)))
)
def test_ragged_rows_are_rejected(
    tmp_path: Path, runtime_support: ModuleType, n: int, field: str, change: int
) -> None:
    submission = _construction(n)
    submission["result"][field][-1] = [0] * (n + change)
    _assert_protocol_rejected(tmp_path, runtime_support, submission)


@pytest.mark.parametrize("field", MATRIX_FIELDS)
@pytest.mark.parametrize("entry", (True, False, -1, 2, 0.5, "1", None))
def test_invalid_matrix_entries_are_rejected(
    tmp_path: Path, runtime_support: ModuleType, field: str, entry: object
) -> None:
    submission = _construction(8)
    submission["result"][field][0][0] = entry
    _assert_protocol_rejected(tmp_path, runtime_support, submission)
    assert _verify(tmp_path, submission).reward == 0


@pytest.mark.parametrize("dimension", (7, 15, True, 8.5, "8", None))
def test_invalid_dimension_is_rejected(
    tmp_path: Path, runtime_support: ModuleType, dimension: object
) -> None:
    submission = _construction(8)
    submission["result"]["dimension"] = dimension
    _assert_protocol_rejected(tmp_path, runtime_support, submission)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_rank_corruption_fails(tmp_path: Path, n: int) -> None:
    submission = _construction(n)
    submission["result"]["low_rank_completion"] = submission["result"][
        "full_rank_completion"
    ]
    PUBLIC_VALIDATOR.validate(submission)
    assert _verify(tmp_path, submission).reward == 0


@pytest.mark.parametrize("field", ("dimension", "defects", *MATRIX_FIELDS))
def test_missing_result_field_is_rejected(
    tmp_path: Path, runtime_support: ModuleType, field: str
) -> None:
    submission = _construction(8)
    del submission["result"][field]
    _assert_protocol_rejected(tmp_path, runtime_support, submission)


@pytest.mark.parametrize("location", ("envelope", "result"))
def test_unknown_fields_are_rejected(
    tmp_path: Path, runtime_support: ModuleType, location: str
) -> None:
    submission = _construction(8)
    target = submission if location == "envelope" else submission["result"]
    target["unknown"] = 0
    _assert_protocol_rejected(tmp_path, runtime_support, submission)


def test_extra_witness_key_is_rejected(
    tmp_path: Path, runtime_support: ModuleType
) -> None:
    submission = _construction(8)
    submission["witness"] = []
    _assert_protocol_rejected(tmp_path, runtime_support, submission)
    assert _verify(tmp_path, submission).reward == 0


def test_public_size_domain_and_generated_artifacts_agree() -> None:
    assert tuple(range(8, 15)) == DIMENSIONS
    accepted = []
    for n in range(1, 22):
        if PUBLIC_VALIDATOR.is_valid(_construction(n)):
            accepted.append(n)
    assert tuple(accepted) == DIMENSIONS
    contract_path = TASK_ROOT / "tests/public_contract.json"
    assert check(contract_path, TASK_ROOT) == []
    assert (
        json.loads(render_submission_schema(load_contract(contract_path)))
        == PUBLIC_SCHEMA
    )
