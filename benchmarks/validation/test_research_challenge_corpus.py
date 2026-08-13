from __future__ import annotations

import json
import tomllib
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

REPO_ROOT = Path(__file__).parents[2]
DATASET = REPO_ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"
TASKS = REPO_ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"


def _task_dirs() -> list[Path]:
    return sorted(ref.path for ref in get_suite("research-diagnostics-v1").tasks)


def test_research_diagnostics_are_one_public_answer_visible_task_each() -> None:
    tasks = _task_dirs()
    task_ids = {path.name for path in tasks}
    assert task_ids, "expected research-diagnostics tasks"
    assert all(
        path.name.startswith("jcb-postdoc-")
        and path.name[len("jcb-postdoc-") :].isdigit()
        for path in tasks
    )
    members = sorted((DATASET / "members").glob("*.toml"))
    assert {path.stem for path in members} == task_ids
    assert not (DATASET / "dataset.toml").exists()
    for task in tasks:
        cfg = tomllib.loads((task / "task.toml").read_text())
        metadata = cfg["metadata"]
        assert metadata["evaluation_kind"] == "research-diagnostic"
        assert metadata["answer_visibility"] == "public"
        assert metadata["assurance_ceiling"] == "COMPUTED"
        assert metadata["provenance_class"] == "public-answer-visible-diagnostic"
        assert (task / "README.md").is_file()
        assert (task / "instruction.md").is_file()
        assert (task / "environment" / "input.json").is_file()
        assert (task / "solution").is_dir()
        assert (task / "tests" / "verifier.py").is_file()
        prompt = (task / "instruction.md").read_text().lower()
        assert "http://" not in prompt and "https://" not in prompt


def test_research_tasks_keep_source_answers_out_of_agent_environment() -> None:
    for task in _task_dirs():
        visible = [task / "instruction.md", *(task / "environment").rglob("*")]
        for path in visible:
            if path.is_file() and path.suffix in {".json", ".md", ".py", ".toml"}:
                text = path.read_text(errors="replace").lower()
                assert "source_answer" not in text
                assert "oracle_summary" not in text
                if path.name == "input.json":
                    json.loads(text)


_STATUS_VOCABULARY = {
    "historical_fit": {"DIRECT", "PARTIAL", "MISSING"},
    "current_status": {"COVERED", "PARTIAL", "OPEN_GAP"},
    "evaluation_status": {
        "REGRESSION_COVERED",
        "RUNNABLE_PUBLIC_REPRODUCTION",
        "BLOCKED_ON_INTERVENTION",
    },
}
_REQUIRED_STATUS_FIELDS = set(_STATUS_VOCABULARY) | {"next_action"}


def test_research_status_overlay_is_folded_into_task_maintainer_metadata() -> None:
    for task in _task_dirs():
        cfg = tomllib.loads((task / "task.toml").read_text())
        metadata = cfg["metadata"]
        assert metadata.keys() >= _REQUIRED_STATUS_FIELDS
        for field, allowed in _STATUS_VOCABULARY.items():
            assert metadata[field] in allowed, (
                f"{task.name}: unsupported {field} value {metadata[field]!r}; "
                f"expected one of {sorted(allowed)}"
            )
        assert metadata["next_action"]
        readme = (task / "README.md").read_text()
        assert "## Portfolio status" in readme
        assert metadata["current_status"] in readme


def test_research_status_overlay_readme_matches_task_metadata() -> None:
    """The README portfolio-status block must mirror task.toml exactly.

    Catches drift where a task bump introduces unsupported status vocabulary
    (e.g. ``BOUNDED_OPERATION_AVAILABLE``) or drops the canonical
    ``## Portfolio status`` section, even when the task.toml values are later
    repaired. Each status field must appear in the README under backticks so
    the maintainer view and the machine-readable metadata stay consistent.
    """
    for task in _task_dirs():
        cfg = tomllib.loads((task / "task.toml").read_text())
        metadata = cfg["metadata"]
        readme = (task / "README.md").read_text()
        assert "## Portfolio status" in readme, (
            f"{task.name}: README is missing the canonical '## Portfolio status' section"
        )
        for field in _STATUS_VOCABULARY:
            value = metadata[field]
            assert f"`{value}`" in readme, (
                f"{task.name}: README does not quote {field} value {value!r} "
                "under the portfolio-status section"
            )
