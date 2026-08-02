from __future__ import annotations

import json
from pathlib import Path

from tools import check_benchmark_adapters
from tools.check_benchmark_adapters import _failures


def test_adapter_lock_requires_disjoint_rows_and_pinned_outputs(tmp_path: Path) -> None:
    adapter = tmp_path / "source"
    adapter.mkdir()
    (adapter / "generate.py").write_text("", encoding="utf-8")
    (adapter / "check.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (adapter / "source.lock.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "adapter_id": "source",
                "source": {
                    "url": "https://example.invalid/data.json",
                    "revision": "v1",
                    "sha256": "sha256:" + "a" * 64,
                    "license": "MIT",
                    "redistribution": "allowed",
                },
                "selection": {
                    "included_rows": ["row-1"],
                    "excluded_rows": ["row-1"],
                    "rule": "frozen fixture",
                },
                "dependencies": {"converter": "==1.0.0"},
                "outputs": [
                    {
                        "task_id": "case-1",
                        "dataset": "public-reproductions-v1",
                        "source_row": "row-1",
                        "task_digest": "sha256:" + "b" * 64,
                        "oracle_evidence_digest": "sha256:" + "c" * 64,
                        "parity_evidence_digest": "sha256:" + "d" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failures = _failures(adapter)

    assert any("included and excluded" in failure for failure in failures)


def test_adapter_output_digest_uses_the_pinned_harbor_task_model(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = tmp_path / "benchmarks" / "adapters" / "source"
    task = tmp_path / "benchmarks" / "datasets" / "suite" / "case"
    adapter.mkdir(parents=True)
    task.mkdir(parents=True)
    (task / "task.toml").write_text("schema_version = '1.4'\n", encoding="utf-8")
    (adapter / "generate.py").write_text("", encoding="utf-8")
    (adapter / "check.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    lock = {
        "schema_version": "1",
        "adapter_id": "source",
        "source": {
            "url": "https://example.invalid/data.json",
            "revision": "v1",
            "sha256": "sha256:" + "a" * 64,
            "license": "MIT",
            "redistribution": "allowed",
        },
        "selection": {"included_rows": ["row-1"], "excluded_rows": [], "rule": "all"},
        "dependencies": {"converter": "==1.0.0"},
        "outputs": [
            {
                "task_id": "case",
                "dataset": "suite",
                "source_row": "row-1",
                "task_digest": "sha256:" + "b" * 64,
                "oracle_evidence_digest": "sha256:" + "c" * 64,
                "parity_evidence_digest": "sha256:" + "d" * 64,
            }
        ],
    }
    (adapter / "source.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(check_benchmark_adapters, "ROOT", tmp_path)
    monkeypatch.setattr(
        "benchmarks.tooling.harbor_suite.task_digest", lambda _path: "b" * 64
    )

    assert _failures(adapter) == []
