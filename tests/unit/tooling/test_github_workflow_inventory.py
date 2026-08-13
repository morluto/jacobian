from __future__ import annotations

from pathlib import Path

from tools.inventory_github_workflows import classify, workflow_stems


def test_workflow_stems_read_yaml_files(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    (workflows / "benchmarks.yaml").write_text("name: benchmarks\n", encoding="utf-8")
    (workflows / "notes.txt").write_text("ignored\n", encoding="utf-8")

    assert workflow_stems(tmp_path) == {"ci", "benchmarks"}


def test_classify_historical_agent_leftovers() -> None:
    report = classify(
        {"ci", "agent-port-alpha", "agent-rebase-beta", "retired-other"},
        {"ci"},
    )

    assert report["historical_agent_leftovers"] == (
        "agent-port-alpha",
        "agent-rebase-beta",
    )
    assert report["registered_without_files"] == (
        "agent-port-alpha",
        "agent-rebase-beta",
        "retired-other",
    )
    assert report["files_without_registration"] == ()
