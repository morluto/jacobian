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
