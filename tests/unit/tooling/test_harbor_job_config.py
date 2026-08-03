from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[3]
JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / "jobs"
    / "jacobian-observation.json"
)
CONTROL_JOB = ROOT / "benchmarks" / "config" / "agent-workflow-v1-control.json"
MCP_CONFIG = ROOT / "benchmarks" / "config" / "jacobian.mcp.json"


def test_observation_job_uses_harbor_dataset_selection() -> None:
    job = json.loads(JOB.read_text())

    assert "tasks" not in job
    assert job["datasets"] == [
        {
            "path": "benchmarks/datasets/agent-workflow-v1",
            "task_names": ["graph-counterexample"],
        }
    ]
    assert job["agents"] == [{"name": "codex", "kwargs": {"web_search": "disabled"}}]
    assert job["environment"]["extra_allowed_hosts"] == [
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
        "deb.debian.org",
        "nodejs.org",
        "npmjs.org",
        "registry.npmjs.org",
        "raw.githubusercontent.com",
    ]


def test_observation_mcp_config_is_external_to_the_task_job() -> None:
    job = json.loads(JOB.read_text())
    control = json.loads(CONTROL_JOB.read_text())
    mcp = json.loads(MCP_CONFIG.read_text())

    assert "mcp_servers" not in job["agents"][0]
    assert "mcp_servers" not in control["agents"][0]
    assert mcp["mcp_servers"] == [
        {
            "name": "jacobian",
            "transport": "streamable-http",
            "url": "http://jacobian:8000/mcp",
        }
    ]


def test_observation_dataset_contains_the_canonical_task() -> None:
    suite = get_suite("agent-workflow-v1")

    assert any(ref.path.name == "graph-counterexample" for ref in suite.tasks)


def test_paired_jobs_use_three_attempts_per_condition() -> None:
    treatment = json.loads(JOB.read_text())
    control = json.loads(CONTROL_JOB.read_text())

    assert treatment["n_attempts"] == 3
    assert control["n_attempts"] == 3
