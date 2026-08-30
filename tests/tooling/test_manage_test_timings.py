"""Regression tests for the benchmark timing-history lifecycle."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def _script() -> ModuleType:
    path = Path(__file__).parents[2] / ".github/scripts/manage-test-timings"
    loader = SourceFileLoader("manage_test_timings", str(path))
    spec = importlib.util.spec_from_loader("manage_test_timings", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_payload(*, generated_at: datetime) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "benchmark-test-durations.json",
            json.dumps(
                {
                    "version": 1,
                    "suite": "benchmark",
                    "source_sha": "a" * 40,
                    "generated_at": generated_at.isoformat(),
                    "python_version": "3.12",
                    "shard_count": 4,
                    "pytest_split_version": "0.11.0",
                    "durations": {
                        "benchmarks/validation/test_example.py::test_case": 1.25
                    },
                }
            ),
        )
    return buffer.getvalue()


def test_stale_timing_artifact_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _script()
    now = datetime(2026, 8, 27, tzinfo=UTC)
    artifact = {
        "workflow_run": {"id": 1},
        "archive_download_url": "https://example.invalid/timings.zip",
        "created_at": (now - timedelta(days=31)).isoformat(),
    }
    monkeypatch.setattr(
        module,
        "api_json",
        lambda *_args, **_kwargs: {"head_branch": "main", "conclusion": "success"},
    )
    monkeypatch.setattr(
        module,
        "download",
        lambda *_args, **_kwargs: _artifact_payload(generated_at=now),
    )

    with pytest.raises(ValueError, match="timing artifact creation exceeds"):
        module.artifact_durations(
            artifact,
            api="https://api.example.invalid",
            repository="example/repository",
            token="unused",
            suite="benchmark",
            now=now,
        )


def test_fresh_main_timing_artifact_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script()
    now = datetime(2026, 8, 27, tzinfo=UTC)
    artifact = {
        "workflow_run": {"id": 1},
        "archive_download_url": "https://example.invalid/timings.zip",
        "created_at": now.isoformat(),
    }
    monkeypatch.setattr(
        module,
        "api_json",
        lambda *_args, **_kwargs: {
            "head_branch": "main",
            "conclusion": "success",
            "head_sha": "a" * 40,
        },
    )
    monkeypatch.setattr(
        module,
        "download",
        lambda *_args, **_kwargs: _artifact_payload(generated_at=now),
    )

    assert module.artifact_durations(
        artifact,
        api="https://api.example.invalid",
        repository="example/repository",
        token="unused",
        suite="benchmark",
        now=now,
    ) == {"benchmarks/validation/test_example.py::test_case": 1.25}


def test_recent_timing_artifact_requires_recent_generation_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script()
    now = datetime(2026, 8, 27, tzinfo=UTC)
    artifact = {
        "workflow_run": {"id": 1},
        "archive_download_url": "https://example.invalid/timings.zip",
        "created_at": now.isoformat(),
    }
    monkeypatch.setattr(
        module,
        "api_json",
        lambda *_args, **_kwargs: {"head_branch": "main", "conclusion": "success"},
    )
    monkeypatch.setattr(
        module,
        "download",
        lambda *_args, **_kwargs: _artifact_payload(
            generated_at=now - timedelta(days=31)
        ),
    )

    with pytest.raises(ValueError, match="timing artifact generation exceeds"):
        module.artifact_durations(
            artifact,
            api="https://api.example.invalid",
            repository="example/repository",
            token="unused",
            suite="benchmark",
            now=now,
        )


def test_timing_artifact_source_must_match_its_successful_workflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script()
    now = datetime(2026, 8, 27, tzinfo=UTC)
    artifact = {
        "workflow_run": {"id": 1},
        "archive_download_url": "https://example.invalid/timings.zip",
        "created_at": now.isoformat(),
    }
    monkeypatch.setattr(
        module,
        "api_json",
        lambda *_args, **_kwargs: {
            "head_branch": "main",
            "conclusion": "success",
            "head_sha": "b" * 40,
        },
    )
    monkeypatch.setattr(
        module,
        "download",
        lambda *_args, **_kwargs: _artifact_payload(generated_at=now),
    )

    with pytest.raises(ValueError, match="source SHA does not match"):
        module.artifact_durations(
            artifact,
            api="https://api.example.invalid",
            repository="example/repository",
            token="unused",
            suite="benchmark",
            now=now,
        )


def test_math_timing_history_uses_four_shards_and_math_node_ids() -> None:
    module = _script()

    assert module.shard_count("math") == 4
    assert module.validate_durations(
        {"tests/math/logic/test_cnf.py::test_case": 1.25},
        "math-test-durations.json",
        "math",
    ) == {"tests/math/logic/test_cnf.py::test_case": 1.25}


def test_math_timing_history_admits_more_than_ten_thousand_cases() -> None:
    module = _script()
    durations = {
        f"tests/math/generated/test_{index}.py::test_case": 0.01
        for index in range(10_100)
    }

    assert len(module.validate_durations(durations, "timings.json", "math")) == 10_100


def test_math_timing_history_merges_four_shards(tmp_path: Path) -> None:
    module = _script()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    third = tmp_path / "third.json"
    fourth = tmp_path / "fourth.json"
    output = tmp_path / "math-test-durations.json"
    first.write_text(
        json.dumps({"tests/math/logic/test_cnf.py::test_case": 1.25}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"tests/math/topology/test_graph.py::test_case": 2.5}),
        encoding="utf-8",
    )
    third.write_text(
        json.dumps({"tests/math/logic/test_sat.py::test_case": 3.75}),
        encoding="utf-8",
    )
    fourth.write_text(
        json.dumps({"tests/math/topology/test_hypergraph.py::test_case": 5.0}),
        encoding="utf-8",
    )

    module.merge(
        [first, second, third, fourth],
        output,
        "a" * 40,
        "3.12",
        "0.11.0",
        "math",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["suite"] == "math"
    assert payload["shard_count"] == 4
    assert payload["durations"] == {
        "tests/math/logic/test_cnf.py::test_case": 1.25,
        "tests/math/logic/test_sat.py::test_case": 3.75,
        "tests/math/topology/test_graph.py::test_case": 2.5,
        "tests/math/topology/test_hypergraph.py::test_case": 5.0,
    }
