from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import support

ROOT = Path(__file__).parents[2]
DATASETS = ROOT / "benchmarks" / "datasets"
PROVIDER_TASKS = ("cddlib", "cgal", "gudhi", "lean-repl", "nauty", "regina")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _solution_case(tmp_path: Path, dataset: str, task_name: str):
    task = DATASETS / dataset / task_name
    app = tmp_path / task_name / "app"
    logs = tmp_path / task_name / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    shutil.copy2(task / "solution" / "submission.json", app / "submission.json")
    shutil.copy2(task / "solution" / "answer.txt", app / "evidence" / "answer.txt")
    return task, app, logs


@pytest.mark.parametrize("task_name", PROVIDER_TASKS)
def test_provider_verifier_images_include_bound_frozen_input(task_name: str) -> None:
    task = DATASETS / "provider-feasibility-v1" / task_name
    assert (task / "tests" / "input.json").read_bytes() == (
        task / "environment" / "input.json"
    ).read_bytes()
    assert (
        "COPY expected.json input.json " in (task / "tests" / "Dockerfile").read_text()
    )


@pytest.mark.parametrize("task_name", PROVIDER_TASKS)
def test_provider_separate_verifier_publishes_bound_input_artifact(
    task_name: str,
) -> None:
    """Separate verifier mode only receives declared artifacts.

    ``load_submission`` requires ``/app/input.json`` to match the frozen
    verifier input, so provider tasks must publish that path or Oracle reward
    collapses to zero after a successful spike.
    """

    task = DATASETS / "provider-feasibility-v1" / task_name
    text = (task / "task.toml").read_text(encoding="utf-8")
    assert 'environment_mode = "separate"' in text
    assert '"/app/input.json"' in text
    assert text.index('"/app/input.json"') < text.index('"/app/submission.json"')


def test_provider_feasibility_task_descriptions_are_provider_specific() -> None:
    """Provider-feasibility task descriptions must identify their own provider.

    A copy/paste from cddlib left five tasks advertising a cddlib H/V
    polyhedra contract they do not exercise.  Each task's ``description`` and
    ``keywords`` must reference its own ``required_provider`` so catalog
    consumers can distinguish the tasks without reading their implementation.
    """

    import re
    import tomllib

    def _normalize_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower().replace("-", " ")))

    for task_name in PROVIDER_TASKS:
        task = DATASETS / "provider-feasibility-v1" / task_name
        cfg = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        provider = cfg["metadata"]["required_provider"]
        description = cfg["task"]["description"]
        keywords = cfg["task"]["keywords"]
        description_tokens = _normalize_tokens(description)
        keyword_tokens = set().union(*(_normalize_tokens(k) for k in keywords))
        provider_tokens = _normalize_tokens(provider)
        # Every provider's description must mention every token of its own
        # provider name (e.g. "lean" and "repl" for "lean-repl").
        missing_in_description = provider_tokens - description_tokens
        assert not missing_in_description, (
            f"{task_name}: description does not mention required_provider "
            f"{provider!r} tokens {sorted(missing_in_description)}: "
            f"{description!r}"
        )
        missing_in_keywords = provider_tokens - keyword_tokens
        assert not missing_in_keywords, (
            f"{task_name}: keywords do not include required_provider "
            f"{provider!r} tokens {sorted(missing_in_keywords)}: {keywords!r}"
        )
        # No task may advertise another provider's full name in its
        # description or keywords.  Check the contiguous normalized form so
        # "cgal" is not flagged by the word "c" in an unrelated description.
        description_norm = " ".join(
            description.lower().replace("-", " ").replace("_", " ").split()
        )
        keywords_norm = " ".join(
            " ".join(k.lower().replace("-", " ").split()) for k in keywords
        )
        for other in PROVIDER_TASKS:
            other_norm = " ".join(other.lower().replace("-", " ").split())
            if other_norm == " ".join(provider.lower().replace("-", " ").split()):
                continue
            if other_norm in description_norm:
                raise AssertionError(
                    f"{task_name}: description references unrelated provider "
                    f"{other!r}: {description!r}"
                )
            if other_norm in keywords_norm:
                raise AssertionError(
                    f"{task_name}: keywords reference unrelated provider "
                    f"{other!r}: {keywords!r}"
                )


def test_reliability_recomputes_input_and_rejects_coerced_state_count(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    expected_path = task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_probability"] = {"num": "0", "den": "1"}
    copied_task = tmp_path / "reliability-task"
    shutil.copytree(task, copied_task)
    _write_json(copied_task / "tests" / "expected.json", expected)

    accepted = support._run_verifier(copied_task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["states"] = "8"
    _write_json(submission_path, submission)
    rejected = support._run_verifier(copied_task, app, logs)
    assert rejected["reward"] == 0.0


def test_reliability_accepts_equivalent_unreduced_probability(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = {"num": "10", "den": "16"}
    _write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == pytest.approx(1.0)
    assert accepted["reward"] == pytest.approx(1.0)


def test_reliability_rejects_oversized_fraction_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "reliability-triangle-fair"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["probability"] = {"num": "9" * 5000, "den": "1"}
    _write_json(submission_path, submission)

    rejected = run_verifier_in_child(task=task, app=app, logs=logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
    assert (logs / "reward.json").is_file()


def test_symmetry_recomputes_orbits_and_rejects_nested_endpoint_bypass(
    tmp_path: Path,
) -> None:
    task, app, logs = _solution_case(
        tmp_path, "public-reproductions-v1", "symmetry-colored-reflection"
    )
    copied_task = tmp_path / "symmetry-task"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_edge_orbits"] = []
    _write_json(expected_path, expected)

    accepted = support._run_verifier(copied_task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["edge_orbits"] = [[[["a"], ["b"]], [["b"], ["c"]]]]
    _write_json(submission_path, submission)
    rejected = support._run_verifier(copied_task, app, logs)
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize(
    ("task_name", "field", "invalid"),
    (
        ("smith-rank-deficient", "rank", "1"),
        ("lean-transition", "goal_count", "2"),
    ),
)
def test_public_reproductions_reject_schema_invalid_integer_coercion(
    tmp_path: Path, task_name: str, field: str, invalid: str
) -> None:
    task, app, logs = _solution_case(tmp_path, "public-reproductions-v1", task_name)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"][field] = invalid
    _write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def _lean_case(tmp_path: Path, tasks: list[dict]):
    task = DATASETS / "provider-feasibility-v1" / "lean-repl"
    app = tmp_path / "lean-repl" / "app"
    logs = tmp_path / "lean-repl" / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    report = {
        "protocol": "leanprover-community/repl",
        "task_count": 2,
        "completed_count": 2,
        "parameter_error_count": 0,
        "return_code": 0,
        "tasks": tasks,
        "elapsed_seconds": 0.042,
        "stderr": "",
        "limitations": [
            "completed tactic states cannot be replayed into the originating command",
        ],
    }
    report_path = app / "evidence" / "provider-report.json"
    _write_json(report_path, report)
    expected = json.loads((task / "tests" / "expected.json").read_text())
    submission = {
        "task_id": expected["task_id"],
        "conclusion": "FEASIBLE",
        "result": {
            "provider": expected["provider"],
            "contract": expected["contract"],
            "status": "COMPLETED",
            "pin_sha256": expected["pin_sha256"],
        },
        "claimed_assurance": "COMPUTED",
        "scope": "pinned bounded provider reproduction",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/provider-report.json",
                "sha256": _digest(report_path),
            }
        ],
        "limitations": [],
    }
    _write_json(app / "submission.json", submission)
    return task, app, logs


def test_lean_repl_rejects_vacuous_empty_task_report(tmp_path: Path) -> None:
    task, app, logs = _lean_case(tmp_path, [])
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_unhashable_task_id_without_crashing(
    tmp_path: Path,
) -> None:
    tasks = [
        {
            "task_id": ["CONJUNCTION-DECOMPOSITION"],
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "constructor", "goal_count": 0, "error_count": 0}],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_boolean_error_count(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": task_id,
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": tactic, "goal_count": 0, "error_count": False}],
        }
        for task_id, tactic in (
            ("CONJUNCTION-DECOMPOSITION", "constructor"),
            ("LOCAL-PREMISE-APPLICATION", "exact h hP"),
        )
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_derives_completion_from_final_goal_count(
    tmp_path: Path,
) -> None:
    tasks = [
        {
            "task_id": task_id,
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": tactic, "goal_count": 999, "error_count": 0}],
        }
        for task_id, tactic in (
            ("CONJUNCTION-DECOMPOSITION", "constructor"),
            ("LOCAL-PREMISE-APPLICATION", "exact h hP"),
        )
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_rejects_one_step_constructor_trace(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "CONJUNCTION-DECOMPOSITION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "constructor", "goal_count": 0, "error_count": 0}],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_lean_repl_accepts_complete_distinct_task_traces(tmp_path: Path) -> None:
    tasks = [
        {
            "task_id": "CONJUNCTION-DECOMPOSITION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [
                {"tactic": "constructor", "goal_count": 2, "error_count": 0},
                {"tactic": "exact hP", "goal_count": 1, "error_count": 0},
                {"tactic": "exact hQ", "goal_count": 0, "error_count": 0},
            ],
        },
        {
            "task_id": "LOCAL-PREMISE-APPLICATION",
            "completed": True,
            "decomposition_observed": True,
            "tactics": [{"tactic": "exact h hP", "goal_count": 0, "error_count": 0}],
        },
    ]
    task, app, logs = _lean_case(tmp_path, tasks)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == pytest.approx(1.0)


def _provider_report_case(tmp_path: Path, task_name: str, report: dict) -> tuple:
    task = DATASETS / "provider-feasibility-v1" / task_name
    app = tmp_path / task_name / "app"
    logs = tmp_path / task_name / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    report_path = app / "evidence" / "provider-report.json"
    _write_json(report_path, report)
    expected = json.loads((task / "tests" / "expected.json").read_text())
    submission = {
        "task_id": expected["task_id"],
        "conclusion": "FEASIBLE",
        "result": {
            "provider": expected["provider"],
            "contract": expected["contract"],
            "status": "COMPLETED",
            "pin_sha256": expected["pin_sha256"],
        },
        "claimed_assurance": "COMPUTED",
        "scope": "pinned bounded provider reproduction",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/provider-report.json",
                "sha256": _digest(report_path),
            }
        ],
        "limitations": [
            "provider output is not an operator-authorized independent verification",
        ],
    }
    _write_json(app / "submission.json", submission)
    return task, app, logs


def test_cddlib_rejects_fabricated_nonempty_cases(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "cddlib" / "tests" / "expected.json"
        ).read_text()
    )
    frozen = expected["reproduction"]
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "runtime": {"python": "3.12.0"},
            "versions": frozen["mathematical_output"]["versions"],
        },
        "reproduction": {
            "scope": frozen["scope"],
            "provider_output_sha256": fake_digest,
            "cases": [{"case_id": "fabricated"}],
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "cddlib", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_regina_rejects_fabricated_nonempty_cases(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "regina" / "tests" / "expected.json"
        ).read_text()
    )
    frozen = expected["reproduction"]
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "runtime": {"python": "3.12.0"},
            "distribution_version": frozen["expected_provider_output"][
                "distribution_version"
            ],
        },
        "reproduction": {
            "scope": frozen["scope"],
            "provider_output_sha256": fake_digest,
            "cases": [{"case_id": "fabricated"}],
            "normal_surfaces": {"surface_count": 0, "surfaces": []},
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "regina", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_cgal_rejects_fabricated_reproduction_digests(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "cgal" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "executable": "/usr/local/bin/cgal-spike",
            "executable_sha256": fake_digest,
        },
        "reproductions": {
            name: {
                **case,
                "observed_output_sha256": fake_digest,
                "expected_output_sha256": fake_digest,
            }
            for name, case in expected["reproductions"].items()
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "cgal", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_cgal_rejects_unbound_source_and_adapter_identity(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "cgal" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "executable": "/usr/local/bin/cgal-spike",
            "executable_sha256": fake_digest,
            "adapter_source_sha256": fake_digest,
            "source": {"archive_sha256": fake_digest},
        },
        "reproductions": {
            name: {
                **case,
                "observed_output_sha256": case["expected_output_sha256"],
            }
            for name, case in expected["reproductions"].items()
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "cgal", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_gudhi_rejects_fabricated_persistence_shape(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "gudhi" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "runtime": {"gudhi": "3.13.0", "python": "3.12.0", "numpy": "2.0"}
        },
        "reproduction": {
            "pairs": [{}],
            "filtration": [{}],
            "provider_output_sha256": fake_digest,
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "gudhi", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_gudhi_rejects_missing_mathematical_output_digest(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "gudhi" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "runtime": {"gudhi": "3.13.0", "python": "3.12.0", "numpy": "2.0"}
        },
        "reproduction": {
            "pairs": expected["reproduction"]["pairs"],
            "filtration": expected["reproduction"]["filtration"],
            "provider_output_sha256": fake_digest,
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "gudhi", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_nauty_rejects_fabricated_count_and_digests(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "nauty" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "executables": {"geng": {"path": "/bin/geng", "sha256": fake_digest}}
        },
        "reproduction": {
            "expected_graph6": ["ZZ"],
            "observed_count": -1,
            "observed_output_sha256": fake_digest,
            "expected_output_sha256": fake_digest,
        },
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "nauty", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_nauty_rejects_unbound_canonicalization(tmp_path: Path) -> None:
    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / "nauty" / "tests" / "expected.json"
        ).read_text()
    )
    fake_digest = "sha256:" + ("0" * 64)
    reproduction = expected["reproduction"]
    canonicalization = {
        **expected["canonicalization"],
        "observed_output_sha256": fake_digest,
        "isomorphic_inputs_converged": True,
    }
    report = {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {
            "executables": {"geng": {"path": "/bin/geng", "sha256": fake_digest}}
        },
        "reproduction": {
            **reproduction,
            "observed_output_sha256": reproduction["expected_output_sha256"],
        },
        "canonicalization": canonicalization,
        "limitations": ["fabricated"],
        "extra": True,
    }
    task, app, logs = _provider_report_case(tmp_path, "nauty", report)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def _minimal_report(task_name: str) -> dict:
    """Return the smallest report that passes the evidence digest binding."""

    expected = json.loads(
        (
            DATASETS / "provider-feasibility-v1" / task_name / "tests" / "expected.json"
        ).read_text()
    )
    return {
        "contract": expected["contract"],
        "status": "COMPLETED",
        "conclusion": expected["report_conclusion"],
        "assurance": expected["report_assurance"],
        "provider": {"runtime": {"python": "3.12.0"}},
        "reproduction": {"scope": "fabricated"},
        "limitations": ["fabricated"],
        "extra": True,
    }


def _verified_claim_case(tmp_path: Path, task_name: str) -> tuple:
    """Build a structurally valid provider submission that claims VERIFIED.

    The public schema only allows ``UNVERIFIED`` and ``COMPUTED``, so
    ``load_submission`` returns ``None`` for this fixture.  The verifier must
    still detect the ``VERIFIED`` claim from the raw submission bytes and
    report ``false_certification: true`` while keeping reward at zero.
    """

    task, app, logs = _provider_report_case(
        tmp_path, task_name, _minimal_report(task_name)
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    _write_json(app / "submission.json", submission)
    return task, app, logs


@pytest.mark.parametrize("task_name", PROVIDER_TASKS)
def test_provider_verifier_flags_verified_claim_as_false_certification(
    task_name: str, tmp_path: Path
) -> None:
    """A VERIFIED claim must produce false_certification=true and reward=0.

    The public schema rejects ``VERIFIED`` so ``load_submission`` returns
    ``None``, but the verifier must inspect the raw submission to expose
    the assurance miscalibration diagnostic without awarding any reward.
    """

    task, app, logs = _verified_claim_case(tmp_path, task_name)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True
