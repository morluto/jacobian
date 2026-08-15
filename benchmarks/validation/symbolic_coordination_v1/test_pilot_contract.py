from __future__ import annotations

import ast
import importlib.util
import json
import sys
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest
from benchmarks.tooling.command_runner import ToolCommandStatus, run_operator_command
from benchmarks.validation.symbolic_coordination_v1 import support
from jsonschema.validators import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "benchmarks/datasets/symbolic-coordination-v1"
MANIFEST = json.loads((DATASET / "pilot-manifest.json").read_text())
TASK_IDS = [case["task_id"] for case in MANIFEST["cases"]]


def canonical_case(tmp_path: Path, task_id: str):
    task, app, logs = support.prepare(tmp_path, task_id)
    return task, app, logs, json.loads((app / "submission.json").read_text())


def _parse_poly(coord: dict) -> dict[tuple[int, ...], Fraction]:
    """Collapse a polynomial coordinate into a nonzero term map keyed by exponents."""

    result: dict[tuple[int, ...], Fraction] = {}
    for term in coord["terms"]:
        coefficient = Fraction(
            int(term["coefficient"]["num"]), int(term["coefficient"]["den"])
        )
        exponents = tuple(term["exponents"])
        result[exponents] = result.get(exponents, Fraction(0)) + coefficient
    return {k: v for k, v in result.items() if v != 0}


def _evaluate(coords: list[dict[tuple[int, ...], Fraction]], point: tuple) -> tuple:
    """Evaluate the parsed polynomial coordinates at a rational point."""

    values = []
    for coord in coords:
        total = Fraction(0)
        for exponents, coefficient in coord.items():
            monomial = coefficient
            for value, exp in zip(point, exponents, strict=True):
                monomial *= value**exp
            total += monomial
        values.append(total)
    return tuple(values)


def _find_collision_in_grid(
    forward_map: dict, record: dict
) -> tuple[tuple, tuple, tuple]:
    """Search the declared grid for two distinct points sharing a common image."""

    coords = [_parse_poly(c) for c in forward_map["coordinates"]]
    values = tuple(
        Fraction(v) for v in range(record["min_numerator"], record["max_numerator"] + 1)
    )
    points = list(product(values, repeat=len(forward_map["variables"])))
    images: dict[tuple, tuple] = {}
    for point in points:
        image = _evaluate(coords, point)
        if image in images:
            return images[image], point, image
        images[image] = point
    raise AssertionError("no collision found in the declared grid")


def _point_json(point: tuple) -> list[dict[str, str]]:
    """Serialize a rational point as the certificate's coordinate list."""

    return [{"num": str(c.numerator), "den": str(c.denominator)} for c in point]


def test_generator_is_deterministic() -> None:
    completed = run_operator_command(
        "uv",
        (
            "run",
            "--locked",
            "python",
            str(DATASET / "generate.py"),
            "--check",
        ),
        cwd=ROOT,
        timeout_seconds=60.0,
    )
    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode(errors="replace")
    assert b"26 generated cases are current" in completed.stdout


def test_manifest_has_the_frozen_pilot_family_balance() -> None:
    assert MANIFEST["case_count"] == 26
    assert len(TASK_IDS) == len(set(TASK_IDS)) == 26
    assert Counter(case["family"] for case in MANIFEST["cases"]) == {
        "valid-two-sided-inverse": 5,
        "perturbed-near-miss": 4,
        "one-direction-only-evidence": 3,
        "constant-nonzero-jacobian": 4,
        "bounded-collision-scope": 6,
        "semantic-equivalence": 4,
    }


def test_verifier_is_clean_room_and_backend_independent() -> None:
    text = (DATASET / "verifier_template.py").read_text()
    imports = {
        *(
            alias.name.split(".", 1)[0]
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Import)
            for alias in node.names
        ),
        *(
            node.module.split(".", 1)[0]
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ),
    }
    assert not imports & {"jacobian", "sympy", "generate"}
    assert all(
        (DATASET / task_id / "tests/verifier.py").read_text() == text
        for task_id in TASK_IDS
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_every_hidden_oracle_solution_receives_full_reward(
    tmp_path: Path, task_id: str
) -> None:
    task, app, logs = support.prepare(tmp_path, task_id)
    result = support.run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.details == {
        "claim_binding": 1.0,
        "input_binding": 1.0,
        "mathematics": 1.0,
        "protocol": 1.0,
        "aggregate_reward": 1.0,
    }


@pytest.mark.parametrize(
    ("task_id", "first", "second", "image"),
    [
        (
            "symbolic-coordination-collision-found-01",
            [1, 2],
            [2, 1],
            [3, 2],
        ),
        (
            "symbolic-coordination-collision-found-02",
            [-1, 0],
            [1, 0],
            [1, 0],
        ),
    ],
)
def test_alternate_exact_collision_witnesses_are_accepted(
    tmp_path: Path,
    task_id: str,
    first: list[int],
    second: list[int],
    image: list[int],
) -> None:
    task, app, logs, submission = canonical_case(tmp_path, task_id)

    def point(values: list[int]) -> list[dict[str, str]]:
        return [{"num": str(value), "den": "1"} for value in values]

    certificate = submission["result"]["certificate"]
    certificate["first_point"] = point(first)
    certificate["second_point"] = point(second)
    certificate["common_image"] = point(image)
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.details["mathematics"] == 1.0


def test_malformed_certificate_is_rejected_without_crashing(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    submission["result"]["certificate"] = None
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("payload", [None, "{not-json"])
def test_empty_or_malformed_submission_fails_closed(
    tmp_path: Path, payload: str | None
) -> None:
    task, app, logs, _submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    if payload is None:
        (app / "submission.json").unlink()
    else:
        (app / "submission.json").write_text(payload)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


def test_unknown_submission_field_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    submission["unexpected"] = "field"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


def test_schema_invalid_submission_preserves_independent_diagnostics(
    tmp_path: Path,
) -> None:
    """A protocol-only schema error must not erase valid mathematical replay."""
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    submission["unexpected"] = "field"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.details["claim_binding"] == 1.0
    assert result.reward == 0.0


def test_one_sided_inverse_certificate_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-one-direction-01"
    )
    certificate = submission["result"]["certificate"]
    certificate["checked_directions"] = ["INVERSE_AFTER_FORWARD"]
    certificate.pop("forward_after_inverse_residuals")
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_forged_second_composition_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-02"
    )
    submission["result"]["certificate"]["forward_after_inverse_residuals"][0] = {
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0]}]
    }
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_near_miss_cannot_substitute_zero_residuals(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-near-miss-01"
    )
    certificate = submission["result"]["certificate"]
    certificate["inverse_after_forward_residuals"] = [{"terms": []}, {"terms": []}]
    certificate["forward_after_inverse_residuals"] = [{"terms": []}, {"terms": []}]
    submission["result"]["verdict"] = "VALID_TWO_SIDED_INVERSE"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_stale_subject_binding_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-03"
    )
    submission["result"]["bindings"]["subject_sha256"] = "sha256:" + "0" * 64
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["claim_binding"] == 0.0
    assert result.reward == 0.0


def test_substituted_workspace_input_is_rejected_separately(tmp_path: Path) -> None:
    task, app, logs, _submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-04"
    )
    data = json.loads((app / "input.json").read_text())
    data["claim_id"] += "-substituted"
    support.write_json(app / "input.json", data)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_legacy_generic_claim_field_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-grid-exhausted-01"
    )
    submission["claimed_assurance"] = "VERIFIED"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "task_id",
    [
        "symbolic-coordination-search-timeout-01",
        "symbolic-coordination-search-incomplete-01",
    ],
)
def test_incomplete_search_cannot_be_promoted_to_grid_exhaustion(
    tmp_path: Path, task_id: str
) -> None:
    task, app, logs, submission = canonical_case(tmp_path, task_id)
    record = json.loads((app / "input.json").read_text())["search_record"]
    submission["result"]["verdict"] = "NO_COLLISION_IN_DECLARED_GRID"
    submission["result"]["certificate"] = {
        "kind": "BOUNDED_GRID_EXHAUSTION_REPLAY",
        "grid": {
            "min_numerator": record["min_numerator"],
            "max_numerator": record["max_numerator"],
            "max_denominator": record["max_denominator"],
        },
        "examined_point_count": record["grid_point_count"],
        "global_consequence": "NOT_ESTABLISHED",
    }
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_keller_certificate_cannot_claim_global_invertibility(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-keller-only-02"
    )
    submission["result"]["certificate"]["global_invertibility"] = "PROVED"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_forged_collision_witness_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-collision-found-01"
    )
    submission["result"]["certificate"]["common_image"][0]["num"] = "99"
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("malformed", [True, float("nan"), {}])
def test_malformed_nested_exponents_fail_closed(
    tmp_path: Path, malformed: object
) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    submission["result"]["certificate"]["inverse_map"]["coordinates"][0]["terms"][0][
        "exponents"
    ][0] = malformed
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


def test_unreduced_rational_coefficient_is_accepted(tmp_path: Path) -> None:
    """Thread 5: schema-valid unreduced rationals must not be rejected."""
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-04"
    )
    inv_map = submission["result"]["certificate"]["inverse_map"]
    for coord in inv_map["coordinates"]:
        for term in coord["terms"]:
            term["coefficient"] = {
                "num": str(int(term["coefficient"]["num"]) * 2),
                "den": str(int(term["coefficient"]["den"]) * 2),
            }
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.reward == 1.0


def test_non_canonical_polynomial_is_accepted(tmp_path: Path) -> None:
    """Thread 1: schema-valid non-canonical polynomial encodings (duplicate
    exponents, zero terms, non-descending order) must be compared semantically."""
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-semantic-equivalence-01"
    )
    inv_map = submission["result"]["certificate"]["inverse_map"]
    # Insert a duplicate exponent that cancels and a zero term, plus reorder.
    coord0 = inv_map["coordinates"][0]
    original_terms = coord0["terms"]
    coord0["terms"] = [
        {"coefficient": {"num": "0", "den": "1"}, "exponents": [0, 0]},
        original_terms[1],
        {"coefficient": {"num": "5", "den": "1"}, "exponents": [1, 0]},
        {"coefficient": {"num": "-5", "den": "1"}, "exponents": [1, 0]},
        original_terms[0],
    ]
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.reward == 1.0


@pytest.mark.parametrize("field", ["unexpected"])
def test_malformed_enum_field_does_not_crash(tmp_path: Path, field: str) -> None:
    """Thread 4: unhashable enum values must produce a deterministic zero,
    not a TypeError crash."""
    task, app, logs, submission = canonical_case(
        tmp_path, "symbolic-coordination-valid-inverse-01"
    )
    submission[field] = ["not", "a", "string"]
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["protocol"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "task_id",
    [
        "symbolic-coordination-search-timeout-01",
        "symbolic-coordination-search-incomplete-01",
    ],
)
def test_collision_witness_in_timeout_case_is_accepted(
    tmp_path: Path, task_id: str
) -> None:
    """An exact collision witness is sufficient even after a non-terminal
    bounded search record."""
    task, app, logs, submission = canonical_case(tmp_path, task_id)
    data = json.loads((app / "input.json").read_text())
    record = data["search_record"]
    grid = {
        "min_numerator": record["min_numerator"],
        "max_numerator": record["max_numerator"],
        "max_denominator": record["max_denominator"],
    }
    first, second, image = _find_collision_in_grid(data["forward_map"], record)
    submission["result"]["verdict"] = "COLLISION_FOUND"
    submission["result"]["certificate"] = {
        "kind": "COLLISION_WITNESS_REPLAY",
        "grid": grid,
        "first_point": _point_json(first),
        "second_point": _point_json(second),
        "common_image": _point_json(image),
        "global_consequence": "MAP_NOT_INJECTIVE_OVER_QQ",
    }
    support.write_json(app / "submission.json", submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.reward == 1.0


def test_semantic_verdict_is_documented_without_a_generic_conclusion() -> None:
    task_dir = DATASET / "symbolic-coordination-grid-exhausted-01"
    instruction = (task_dir / "instruction.md").read_text()
    assert "semantic field" in instruction
    assert "generic conclusion" in instruction


def test_result_only_contract_is_documented() -> None:
    instruction = (
        DATASET / "symbolic-coordination-valid-inverse-01/instruction.md"
    ).read_text()
    assert "terminal certificate in the `result`" in instruction
    assert '"witness"' not in instruction


def _load_generate():
    spec = importlib.util.spec_from_file_location(
        "symbolic_coordination_generate", DATASET / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    return module


_CERTIFICATE_REPRESENTATIVES = {
    "TWO_SIDED_COMPOSITION_REPLAY": "symbolic-coordination-valid-inverse-01",
    "KELLER_DETERMINANT_REPLAY": "symbolic-coordination-keller-only-01",
    "COLLISION_WITNESS_REPLAY": "symbolic-coordination-collision-found-01",
    "BOUNDED_GRID_EXHAUSTION_REPLAY": "symbolic-coordination-grid-exhausted-01",
    "SEARCH_NONCONCLUSION": "symbolic-coordination-search-timeout-01",
}


def test_generated_schemas_accept_only_family_licensed_certificates() -> None:
    generate = _load_generate()
    samples = {
        kind: json.loads((DATASET / task_id / "solution/submission.json").read_text())[
            "result"
        ]
        for kind, task_id in _CERTIFICATE_REPRESENTATIVES.items()
    }
    assert set(samples) == set(generate.ALL_CERTIFICATE_KINDS)
    for raw in generate.cases():
        data = generate.bind_case(raw)
        schema = generate.submission_schema_parts(data)
        licensed = set(generate.licensed_certificate_kinds(data))
        oracle = json.loads(
            (DATASET / str(data["case_id"]) / "solution/submission.json").read_text()
        )
        Draft202012Validator(schema).validate(oracle)
        for kind, sample in samples.items():
            mutated = deepcopy(oracle)
            mutated["result"]["certificate"] = deepcopy(sample["certificate"])
            mutated["result"]["verdict"] = sample["verdict"]
            if kind == "SEARCH_NONCONCLUSION" and kind in licensed:
                record = data["search_record"]
                assert isinstance(record, dict)
                mutated["result"]["certificate"]["stop_reason"] = record["stop_reason"]
            errors = list(Draft202012Validator(schema).iter_errors(mutated))
            if kind in licensed:
                assert errors == [], (data["case_id"], kind)
            else:
                assert errors, (data["case_id"], kind)


def test_inverse_schema_rejects_keller_certificate_and_verdict() -> None:
    generate = _load_generate()
    data = generate.bind_case(
        next(
            case
            for case in generate.cases()
            if case["slug"] == "symbolic-coordination-valid-inverse-01"
        )
    )
    schema = generate.submission_schema_parts(data)
    oracle = json.loads(
        (
            DATASET
            / "symbolic-coordination-valid-inverse-01"
            / "solution/submission.json"
        ).read_text()
    )
    keller = json.loads(
        (
            DATASET
            / "symbolic-coordination-keller-only-01"
            / "solution/submission.json"
        ).read_text()
    )["result"]
    mixed = deepcopy(oracle)
    mixed["result"]["certificate"] = keller["certificate"]
    assert list(Draft202012Validator(schema).iter_errors(mixed))
    mixed_verdict = deepcopy(oracle)
    mixed_verdict["result"]["verdict"] = "KELLER_CONDITION_ONLY"
    assert list(Draft202012Validator(schema).iter_errors(mixed_verdict))
