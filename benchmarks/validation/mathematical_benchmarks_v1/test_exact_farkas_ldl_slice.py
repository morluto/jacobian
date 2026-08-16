import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures
from jsonschema import Draft202012Validator

TASK = "exact-farkas-ldl-slice"
TASK_DIR = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK


def test_result_witness_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def _rational(value: int = 1) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _ldl_certificate() -> dict[str, object]:
    return {
        "l": [
            [_rational(1 if row == column else 0) for column in range(4)]
            for row in range(4)
        ],
        "d": [_rational() for _ in range(4)],
    }


def _sylvester_certificate() -> dict[str, object]:
    return {"leading_principal_determinants": [_rational() for _ in range(4)]}


@pytest.mark.parametrize(
    ("proof_mode", "certificate_factory", "is_valid"),
    [
        ("LDL", _ldl_certificate, True),
        ("LDL", _sylvester_certificate, False),
        ("SYLVESTER", _ldl_certificate, False),
        ("SYLVESTER", _sylvester_certificate, True),
    ],
)
def test_schema_couples_proof_mode_to_certificate_shape(
    proof_mode: str,
    certificate_factory: Callable[[], dict[str, object]],
    is_valid: bool,
) -> None:
    schema = json.loads((TASK_DIR / "environment/submission_schema.json").read_text())
    submission = json.loads((TASK_DIR / "solution/submission.json").read_text())
    candidate = copy.deepcopy(submission)
    candidate["result"]["proof_mode"] = proof_mode
    candidate["result"]["positive_definite_certificate"] = certificate_factory()

    assert Draft202012Validator(schema).is_valid(candidate) is is_valid
