from __future__ import annotations

import json
import tomllib
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

REPO_ROOT = Path(__file__).parents[2]
DATASET = REPO_ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"
TASKS = REPO_ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"
EXPECTED_RESEARCH_TASKS = {f"jcb-postdoc-{index:03d}" for index in range(1, 19)}


def _task_dirs() -> list[Path]:
    return sorted(ref.path for ref in get_suite("research-diagnostics-v1").tasks)


def test_research_diagnostics_are_one_public_answer_visible_task_each() -> None:
    tasks = _task_dirs()
    assert {path.name for path in tasks} == EXPECTED_RESEARCH_TASKS
    members = sorted((DATASET / "members").glob("*.toml"))
    assert {path.stem for path in members} == EXPECTED_RESEARCH_TASKS
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


def test_research_status_overlay_is_folded_into_task_maintainer_metadata() -> None:
    required = {"historical_fit", "current_status", "evaluation_status", "next_action"}
    for task in _task_dirs():
        cfg = tomllib.loads((task / "task.toml").read_text())
        metadata = cfg["metadata"]
        assert required <= metadata.keys()
        assert metadata["historical_fit"] in {"DIRECT", "PARTIAL", "MISSING"}
        assert metadata["current_status"] in {"COVERED", "PARTIAL", "OPEN_GAP"}
        assert metadata["evaluation_status"] in {
            "REGRESSION_COVERED",
            "RUNNABLE_PUBLIC_REPRODUCTION",
            "BLOCKED_ON_INTERVENTION",
        }
        assert metadata["next_action"]
        readme = (task / "README.md").read_text()
        assert "## Portfolio status" in readme
        assert metadata["current_status"] in readme
