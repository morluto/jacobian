from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[2]


JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "jobs"
    / "jacobian-observation.json"
)
CONTROL_JOB = ROOT / "benchmarks" / "config" / "mathematical-benchmarks-v1-control.json"
MCP_CONFIG = ROOT / "benchmarks" / "config" / "jacobian.mcp.json"
LOOPBACK_MCP_CONFIG = ROOT / "benchmarks" / "config" / "jacobian-loopback.mcp.json"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_observation_mcp_config_is_external_to_the_task_job() -> None:
    job = _read_json(JOB)
    control = _read_json(CONTROL_JOB)
    mcp = _read_json(MCP_CONFIG)
    loopback_mcp = _read_json(LOOPBACK_MCP_CONFIG)

    assert "mcp_servers" not in job["agents"][0]
    assert "mcp_servers" not in control["agents"][0]
    assert mcp["mcp_servers"] == [
        {
            "name": "jacobian",
            "transport": "streamable-http",
            "url": "http://jacobian:8000/mcp",
        }
    ]
    assert loopback_mcp["mcp_servers"] == [
        {
            "name": "jacobian",
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    ]


def test_observation_dataset_contains_the_canonical_task() -> None:
    suite = get_suite("mathematical-benchmarks-v1")

    assert any(ref.path.name == "graph-counterexample" for ref in suite.tasks)
