from __future__ import annotations

import importlib.util
import json
import shutil
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling import public_contract
from benchmarks.validation._verifier_child import VerifierOutput, run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures
from jsonschema import Draft202012Validator

TASK = "subset-incidence-determinant"
TASK_PATH = _fixtures._task(TASK)
SAMPLE_SIZES = range(1, 9)


def _submission(n: int = 5, *, reverse_within_rank: bool = False) -> dict[str, Any]:
    """Build an incidence witness without consulting a solution fixture.

    The alternating sum over nonempty subsets of A intersect B is one when
    the intersection is nonempty and zero otherwise, giving Z D Z^T. Ordering
    by cardinality makes Z unit lower triangular; ties can be in either order.
    There are 2^(k-1)-1 nonempty even subsets, so det(D) is 1 only for k=1.
    """
    source = json.loads((TASK_PATH / "environment/input.json").read_text())
    order = sorted(
        range(1, 2**n),
        key=lambda mask: (mask.bit_count(), -mask if reverse_within_rank else mask),
    )
    return {
        "result": {
            "sample_n": n,
            "mask_order": order,
            "diagonal_weights": [1 if mask.bit_count() % 2 else -1 for mask in order],
            "trace": [
                {
                    "n": k,
                    "even_nonempty_count": 2 ** (k - 1) - 1,
                    "determinant": 1 if k == 1 else -1,
                }
                for k in range(1, source["trace_max_n"] + 1)
            ],
            "general_even_count": {
                "base": 2,
                "exponent_offset": -1,
                "constant_offset": -1,
            },
            "general_determinant": {"n_equals_1": 1, "otherwise": -1},
        }
    }


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "subset_incidence_support", TASK_PATH / "tests/verifier_support.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Redirect only the default contract path; retain the actual parser and schema.
    monkeypatch.setattr(
        module,
        "_load_public_contract",
        partial(module._load_public_contract, TASK_PATH / "tests/public_contract.json"),
    )
    return module


def _assert_public_boundary(
    tmp_path: Path,
    support: ModuleType,
    submission: dict[str, Any],
    *,
    accepted: bool,
) -> None:
    schema = json.loads((TASK_PATH / "environment/submission_schema.json").read_text())
    assert Draft202012Validator(schema).is_valid(submission) is accepted
    path = tmp_path / "submission.json"
    _fixtures._write_json(path, submission)
    loaded = support.load_submission(path, require_input_binding=False)
    assert loaded == (submission if accepted else None)


def _verify(tmp_path: Path, submission: dict[str, Any]) -> VerifierOutput:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment/input.json", app / "input.json")
    _fixtures._write_json(app / "submission.json", submission)
    return run_verifier_in_child(task=TASK_PATH, app=app, logs=logs)


def _assert_rejected(tmp_path: Path, submission: dict[str, Any]) -> None:
    rejected = _verify(tmp_path, submission)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_public_contract_exposes_exact_sample_shapes() -> None:
    contract_path = TASK_PATH / "tests/public_contract.json"
    assert public_contract.check(contract_path, TASK_PATH) == []
    contract = json.loads(contract_path.read_text())
    source = json.loads((TASK_PATH / "environment/input.json").read_text())
    result = contract["submission_result"]
    branches = result["oneOf"]
    assert sorted(
        branch["properties"]["sample_n"]["const"] for branch in branches
    ) == list(SAMPLE_SIZES)
    for branch in branches:
        properties = branch["properties"]
        size = 2 ** properties["sample_n"]["const"] - 1
        for field in ("mask_order", "diagonal_weights"):
            assert properties[field]["minItems"] == size
            assert properties[field]["maxItems"] == size
        assert properties["mask_order"]["items"] == {
            "type": "integer",
            "minimum": 1,
            "maximum": size,
        }
    assert result["properties"]["sample_n"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 8,
    }
    assert result["properties"]["trace"]["minItems"] == source["trace_max_n"]
    assert result["properties"]["trace"]["maxItems"] == source["trace_max_n"]


@pytest.mark.parametrize("n", SAMPLE_SIZES)
@pytest.mark.parametrize("reverse_within_rank", [False, True])
def test_accepts_every_sample_shape(
    tmp_path: Path, support: ModuleType, n: int, reverse_within_rank: bool
) -> None:
    submission = _submission(n, reverse_within_rank=reverse_within_rank)
    _assert_public_boundary(tmp_path, support, submission, accepted=True)
    accepted = _verify(tmp_path, submission)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


@pytest.mark.parametrize("sample_n", SAMPLE_SIZES)
@pytest.mark.parametrize("payload_n", SAMPLE_SIZES)
@pytest.mark.parametrize("field", ["mask_order", "diagonal_weights"])
def test_collection_size_must_match_sample_n(
    tmp_path: Path, support: ModuleType, sample_n: int, payload_n: int, field: str
) -> None:
    submission = _submission(sample_n)
    submission["result"][field] = _submission(payload_n)["result"][field]
    _assert_public_boundary(
        tmp_path, support, submission, accepted=sample_n == payload_n
    )


@pytest.mark.parametrize("n", SAMPLE_SIZES)
@pytest.mark.parametrize("field", ["mask_order", "diagonal_weights"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_rejects_adjacent_collection_lengths(
    tmp_path: Path, support: ModuleType, n: int, field: str, delta: int
) -> None:
    submission = _submission(n)
    values = submission["result"][field]
    if delta < 0:
        values.pop()
    else:
        values.append(2**n if field == "mask_order" else 1)
    _assert_public_boundary(tmp_path, support, submission, accepted=False)


@pytest.mark.parametrize("n", SAMPLE_SIZES)
@pytest.mark.parametrize("invalid_mask", ["zero", "above_universe", "huge", "boolean"])
def test_rejects_masks_outside_sample_universe(
    tmp_path: Path, support: ModuleType, n: int, invalid_mask: str
) -> None:
    submission = _submission(n)
    submission["result"]["mask_order"][0] = {
        "zero": 0,
        "above_universe": 2**n,
        "huge": 10**100,
        "boolean": True,
    }[invalid_mask]
    _assert_public_boundary(tmp_path, support, submission, accepted=False)


@pytest.mark.parametrize("sample_n", [0, 9, 10**100, True])
def test_rejects_unsupported_sample_n(
    tmp_path: Path, support: ModuleType, sample_n: object
) -> None:
    submission = _submission(1)
    submission["result"]["sample_n"] = sample_n
    _assert_public_boundary(tmp_path, support, submission, accepted=False)
    _assert_rejected(tmp_path, submission)


@pytest.mark.parametrize("length", [7, 9])
def test_rejects_wrong_trace_length(
    tmp_path: Path, support: ModuleType, length: int
) -> None:
    submission = _submission(1)
    trace = submission["result"]["trace"]
    if length < len(trace):
        trace.pop()
    else:
        trace.append({"n": 9, "even_nonempty_count": 255, "determinant": -1})
    _assert_public_boundary(tmp_path, support, submission, accepted=False)
    _assert_rejected(tmp_path, submission)


@pytest.mark.parametrize("field", ["mask_order", "diagonal_weights"])
def test_child_rejects_mismatched_shape(tmp_path: Path, field: str) -> None:
    submission = _submission(2)
    submission["result"][field] = _submission(3)["result"][field]
    _assert_rejected(tmp_path, submission)


def test_rejects_duplicate_masks(tmp_path: Path, support: ModuleType) -> None:
    submission = _submission(2)
    submission["result"]["mask_order"][1] = submission["result"]["mask_order"][0]
    _assert_public_boundary(tmp_path, support, submission, accepted=False)
    _assert_rejected(tmp_path, submission)


def test_rejects_boolean_diagonal_weights(tmp_path: Path) -> None:
    """Boolean ``true`` must not spoof integer 1 in diagonal_weights."""
    submission = _submission()
    submission["result"]["diagonal_weights"] = [
        True if w == 1 else w for w in submission["result"]["diagonal_weights"]
    ]
    _assert_rejected(tmp_path, submission)


def test_rejects_boolean_mask_order(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["mask_order"][0] = True
    _assert_rejected(tmp_path, submission)


def test_rejects_boolean_trace_fields(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["trace"][0]["n"] = True
    _assert_rejected(tmp_path, submission)


def test_rejects_boolean_sample_n(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["sample_n"] = True
    _assert_rejected(tmp_path, submission)


def test_accepts_reversed_same_cardinality_mask_order(tmp_path: Path) -> None:
    submission = _submission(reverse_within_rank=True)
    submission["result"]["trace"].reverse()
    accepted = _verify(tmp_path, submission)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_accepts_typed_general_formulas(tmp_path: Path) -> None:
    accepted = _verify(tmp_path, _submission())
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


@pytest.mark.parametrize("mutation", ["weights", "order", "even_count", "determinant"])
def test_schema_valid_mathematical_errors_still_fail(
    tmp_path: Path, support: ModuleType, mutation: str
) -> None:
    submission = _submission(2)
    result = submission["result"]
    if mutation == "weights":
        result["diagonal_weights"][0] = -1
    elif mutation == "order":
        result["mask_order"].reverse()
        result["diagonal_weights"].reverse()
    elif mutation == "even_count":
        result["general_even_count"]["base"] = 3
    else:
        result["general_determinant"]["otherwise"] = 1
    _assert_public_boundary(tmp_path, support, submission, accepted=True)
    _assert_rejected(tmp_path, submission)


def test_rejects_wrong_general_determinant_formula(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["general_determinant"]["otherwise"] = 1
    _assert_rejected(tmp_path, submission)
