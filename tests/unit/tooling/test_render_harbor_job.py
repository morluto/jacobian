from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from benchmarks.tooling.harbor_suite import (
    HarborSuiteError,
    get_suite,
    render_suite_job,
)

ROOT = Path(__file__).parents[3]
RENDERER = ROOT / "tools" / "render_harbor_job.py"
JOB = (
    ROOT
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / "jobs"
    / "jacobian-observation.json"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_harbor_job", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_job_config_resolves_only_the_model_placeholder() -> None:
    renderer = load_renderer()
    source = json.loads(JOB.read_text())
    rendered = renderer.render_job_config(source, model="gpt-5.6-sol")

    assert source["agents"][0]["model_name"] == "${JACOBIAN_MODEL}"
    assert rendered["agents"][0]["model_name"] == "gpt-5.6-sol"
    assert rendered["environment"] == source["environment"]
    assert rendered["tasks"] == source["tasks"]


@pytest.mark.parametrize("model", ["", "  "])
def test_render_job_config_rejects_empty_model(model: str) -> None:
    renderer = load_renderer()
    source = json.loads(JOB.read_text())
    with pytest.raises(ValueError, match="model must be non-empty"):
        renderer.render_job_config(source, model=model)


def test_render_job_config_requires_an_explicit_placeholder() -> None:
    renderer = load_renderer()
    source = json.loads(JOB.read_text())
    source["agents"][0]["model_name"] = "already-resolved"
    with pytest.raises(ValueError, match="does not contain"):
        renderer.render_job_config(source, model="gpt-5.6-sol")


def test_render_suite_job_expands_nested_tasks_explicitly() -> None:
    rendered = render_suite_job(get_suite("agent-workflow-v1"), role="oracle")

    assert "datasets" not in rendered
    assert len(rendered["tasks"]) == 26
    assert all(
        entry["path"].startswith(
            "benchmarks/datasets/agent-workflow-v1/tasks/mathematical-sciences/"
        )
        for entry in rendered["tasks"]
    )


def test_render_suite_job_filters_provider_tasks() -> None:
    rendered = render_suite_job(
        get_suite("provider-feasibility-v1"), role="oracle", provider="cgal"
    )

    assert len(rendered["tasks"]) == 1
    assert rendered["tasks"][0]["path"].endswith("/provider-integration/cgal")


def test_render_suite_job_rejects_unknown_provider() -> None:
    with pytest.raises(HarborSuiteError, match="no task requiring provider"):
        render_suite_job(
            get_suite("provider-feasibility-v1"),
            role="oracle",
            provider="missing-provider",
        )
