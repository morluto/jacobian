from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
RENDERER = ROOT / "tools" / "render_harbor_job.py"
JOB = ROOT / "benchmarks" / "regression-v1" / "job-jacobian.json"


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
    assert rendered["datasets"] == source["datasets"]


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
