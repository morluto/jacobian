from __future__ import annotations

import json
import re
import shutil
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier
from jsonschema import Draft202012Validator

TASK = "squarefree-class-independence-audit"
TASK_PATH = _fixtures._task(TASK)
PUBLIC_INPUT = json.loads((TASK_PATH / "environment" / "input.json").read_text())
BOUNDS = PUBLIC_INPUT["certificate_bounds"]
MODULI = range(BOUNDS["minimum_modulus"], BOUNDS["maximum_modulus"] + 1)
PAIR_COUNT_MATCH = re.search(r"exactly (\d+) ordered pairs", PUBLIC_INPUT["problem"])
assert PAIR_COUNT_MATCH is not None
PAIR_COUNT = int(PAIR_COUNT_MATCH[1])
SCHEMA = json.loads((TASK_PATH / "environment" / "submission_schema.json").read_text())


def _public_submission(modulus: int) -> dict[str, Any]:
    return {
        "result": {
            "classification": {
                "class_key": "SQUAREFREE_KERNEL",
                "product_square_iff": "KERNELS_EQUAL",
                "pair_count_formula": "SUM_OF_SQUARED_CLASS_SIZES",
                "independent_selection": "ONE_ELEMENT_PER_DISTINCT_CLASS",
            },
            "modular_obstruction": {
                "modulus": modulus,
                "quadratic_residues": sorted(
                    {value**2 % modulus for value in range(modulus)}
                ),
                "target_residue": PAIR_COUNT % modulus,
            },
        }
    }


def _has_obstruction(modulus: int) -> bool:
    # Derive certificates from public input, using pair sums rather than the
    # verifier's product of three residue lists. Zero permits shorter sums.
    residues = {value**2 % modulus for value in range(modulus)}
    pair_sums = {(left + right) % modulus for left in residues for right in residues}
    return all(
        (PAIR_COUNT - residue) % modulus not in pair_sums for residue in residues
    )


VALID_MODULI = tuple(modulus for modulus in MODULI if _has_obstruction(modulus))


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_source_module(
        "squarefree_public_bounds_support", TASK_PATH / "tests" / "verifier_support.py"
    )
    monkeypatch.setattr(
        module,
        "_load_public_contract",
        partial(
            module._load_public_contract, TASK_PATH / "tests" / "public_contract.json"
        ),
    )
    return module


def _assert_structure(
    submission: dict[str, Any], tmp_path: Path, support: ModuleType, *, accepted: bool
) -> None:
    assert Draft202012Validator(SCHEMA).is_valid(submission) is accepted
    path = tmp_path / "submission.json"
    _fixtures._write_json(path, submission)
    loaded = support.load_submission(path, require_input_binding=False)
    assert loaded == (submission if accepted else None)


def _assert_replay(
    submission: dict[str, Any], tmp_path: Path, *, reward: float
) -> None:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    _fixtures._write_json(app / "submission.json", submission)
    output = run_verifier_in_child(task=TASK_PATH, app=app, logs=logs)
    assert output.reward == reward
    assert output.details["correctness"] == reward
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}


def test_bounds_follow_public_domain_and_generated_contract_is_current() -> None:
    assert check(TASK_PATH / "tests" / "public_contract.json", TASK_PATH) == []
    obstruction = SCHEMA["properties"]["result"]["properties"]["modular_obstruction"][
        "properties"
    ]
    assert obstruction["modulus"] == {
        "type": "integer",
        "minimum": BOUNDS["minimum_modulus"],
        "maximum": BOUNDS["maximum_modulus"],
    }
    scalar = {"type": "integer", "minimum": 0, "maximum": BOUNDS["maximum_modulus"] - 1}
    assert obstruction["quadratic_residues"] == {
        "type": "array",
        "items": scalar,
        "minItems": 2,
        "maxItems": BOUNDS["maximum_modulus"],
        "uniqueItems": True,
    }
    assert obstruction["target_residue"] == scalar
    assert len(VALID_MODULI) > 1
    assert len(VALID_MODULI) < len(MODULI)


@pytest.mark.parametrize("modulus", MODULI)
def test_every_public_modulus_keeps_structural_candidates(
    tmp_path: Path, support: ModuleType, modulus: int
) -> None:
    submission = _public_submission(modulus)
    _assert_structure(submission, tmp_path, support, accepted=True)
    _assert_replay(submission, tmp_path, reward=float(_has_obstruction(modulus)))

    # The public bounds do not encode the complete set for this modulus or
    # the successful target. These deliberately wrong claims remain structural.
    submission["result"]["modular_obstruction"]["quadratic_residues"] = [0, 63]
    submission["result"]["modular_obstruction"]["target_residue"] = 63
    _assert_structure(submission, tmp_path, support, accepted=True)


@pytest.mark.parametrize("modulus", VALID_MODULI)
def test_independently_derived_alternate_certificates_accept_reordered_residues(
    tmp_path: Path, support: ModuleType, modulus: int
) -> None:
    submission = _public_submission(modulus)
    submission["result"]["modular_obstruction"]["quadratic_residues"].reverse()
    _assert_structure(submission, tmp_path, support, accepted=True)
    _assert_replay(submission, tmp_path, reward=1.0)


@pytest.mark.parametrize(
    ("field", "value", "violation"),
    [
        ("quadratic_residues", list(range(64)), None),
        ("quadratic_residues", list(range(65)), "maxItems"),
        ("quadratic_residues", list(range(1000)), "maxItems"),
        ("quadratic_residues", [0, 63], None),
        ("quadratic_residues", [0, 64], "maximum"),
        ("quadratic_residues", [0, 10**99], "maximum"),
        ("quadratic_residues", [0], "minItems"),
        ("quadratic_residues", [0, 0], "uniqueItems"),
        ("quadratic_residues", [-1, 0], "minimum"),
        ("quadratic_residues", [0, True], "type"),
        ("quadratic_residues", [0, 1.5], "type"),
        ("quadratic_residues", [0, "1"], "type"),
        ("target_residue", 63, None),
        ("target_residue", 64, "maximum"),
        ("target_residue", 10**99, "maximum"),
        ("target_residue", -1, "minimum"),
        ("target_residue", True, "type"),
        ("target_residue", 1.5, "type"),
        ("target_residue", "1", "type"),
        ("modulus", 1, "minimum"),
        ("modulus", 65, "maximum"),
    ],
)
def test_public_boundaries_and_malformed_claims_fail_closed(
    tmp_path: Path,
    support: ModuleType,
    field: str,
    value: object,
    violation: str | None,
) -> None:
    submission = _public_submission(BOUNDS["maximum_modulus"])
    submission["result"]["modular_obstruction"][field] = value
    _assert_structure(submission, tmp_path, support, accepted=violation is None)
    if violation is not None:
        assert violation in {
            error.validator
            for error in Draft202012Validator(SCHEMA).iter_errors(submission)
        }
    _assert_replay(submission, tmp_path, reward=0.0)


@pytest.mark.parametrize("claim", ["missing-residue", "extra-residue", "wrong-target"])
def test_structural_but_false_mathematical_claims_are_rejected(
    tmp_path: Path, support: ModuleType, claim: str
) -> None:
    submission = _public_submission(max(VALID_MODULI))
    obstruction = submission["result"]["modular_obstruction"]
    if claim == "missing-residue":
        obstruction["quadratic_residues"].pop()
    elif claim == "extra-residue":
        obstruction["quadratic_residues"].append(
            next(
                value
                for value in range(obstruction["modulus"])
                if value not in obstruction["quadratic_residues"]
            )
        )
    else:
        obstruction["target_residue"] = (
            obstruction["target_residue"] + 1
        ) % obstruction["modulus"]
    _assert_structure(submission, tmp_path, support, accepted=True)
    _assert_replay(submission, tmp_path, reward=0.0)


def test_accepts_permuted_quadratic_residues(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "permuted")
    submission = json.loads((app / "submission.json").read_text())
    residues = submission["result"]["modular_obstruction"]["quadratic_residues"]
    submission["result"]["modular_obstruction"]["quadratic_residues"] = list(
        reversed(residues)
    )
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
