from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import check_benchmark_adapters
from tools.check_benchmark_adapters import _failures


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_adapter_lock_requires_disjoint_rows_and_pinned_outputs(tmp_path: Path) -> None:
    adapter = tmp_path / "source"
    adapter.mkdir()
    (adapter / "generate.py").write_text("", encoding="utf-8")
    (adapter / "check.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    oracle_evidence = _evidence_json({"kind": "oracle"})
    parity_evidence = _evidence_json({"kind": "parity"})
    (adapter / "oracle-evidence.json").write_text(oracle_evidence, encoding="utf-8")
    (adapter / "parity-evidence.json").write_text(parity_evidence, encoding="utf-8")
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
                        "oracle_evidence_digest": _sha256_text(oracle_evidence),
                        "parity_evidence_digest": _sha256_text(parity_evidence),
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
    oracle_evidence = _evidence_json({"kind": "oracle"})
    parity_evidence = _evidence_json({"kind": "parity"})
    (adapter / "oracle-evidence.json").write_text(oracle_evidence, encoding="utf-8")
    (adapter / "parity-evidence.json").write_text(parity_evidence, encoding="utf-8")
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
                "oracle_evidence_digest": _sha256_text(oracle_evidence),
                "parity_evidence_digest": _sha256_text(parity_evidence),
            }
        ],
    }
    (adapter / "source.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(check_benchmark_adapters, "ROOT", tmp_path)
    monkeypatch.setattr(
        "benchmarks.tooling.harbor_suite.task_digest", lambda _path: "b" * 64
    )

    assert _failures(adapter) == []


def test_adapter_evidence_must_remain_present_and_digest_bound(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = tmp_path / "benchmarks" / "adapters" / "source"
    task = tmp_path / "benchmarks" / "datasets" / "suite" / "case"
    adapter.mkdir(parents=True)
    task.mkdir(parents=True)
    (task / "task.toml").write_text("schema_version = '1.4'\n", encoding="utf-8")
    (adapter / "generate.py").write_text("", encoding="utf-8")
    (adapter / "check.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    oracle_evidence = _evidence_json({"kind": "oracle"})
    parity_evidence = _evidence_json({"kind": "parity"})
    (adapter / "oracle-evidence.json").write_text(oracle_evidence, encoding="utf-8")
    (adapter / "parity-evidence.json").write_text(parity_evidence, encoding="utf-8")
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
                "oracle_evidence_digest": _sha256_text(oracle_evidence),
                "parity_evidence_digest": _sha256_text(parity_evidence),
            }
        ],
    }
    lock_path = adapter / "source.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(check_benchmark_adapters, "ROOT", tmp_path)
    monkeypatch.setattr(
        "benchmarks.tooling.harbor_suite.task_digest", lambda _path: "b" * 64
    )

    (adapter / "oracle-evidence.json").write_text(
        _evidence_json({"kind": "changed"}), encoding="utf-8"
    )
    failures = _failures(adapter)

    assert any("oracle_evidence_digest mismatch" in failure for failure in failures)

    (adapter / "oracle-evidence.json").unlink()
    failures = _failures(adapter)

    assert any("required evidence file is missing" in failure for failure in failures)
