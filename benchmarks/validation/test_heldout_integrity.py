from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from benchmarks.tooling import heldout_integrity
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_integrity import _digest, _safe_extract, verify_bundle
from benchmarks.validation.heldout_fixtures import _bundle, _manifest


def test_bundle_binds_complete_verifier_and_oracle_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_integrity, "task_digest", lambda _path: "a" * 64)
    verify_bundle(value, root)
    (root / value["tasks"][0]["verifier_root"] / "extra.py").write_text(
        "pass\n", encoding="utf-8"
    )

    with pytest.raises(HarborSuiteError, match="tree digest mismatch"):
        verify_bundle(value, root)


def test_bundle_rejects_dataset_manifest_task_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_integrity, "task_digest", lambda _path: "a" * 64)
    manifest = root / value["dataset"]["path"] / "dataset.toml"
    manifest.write_text(
        '[dataset]\nname = "jacobian/operation-held-out-v1"\n\n'
        '[[tasks]]\nname = "jacobian/held-out-0"\n'
        f'digest = "{value["tasks"][0]["digest"]}"\n',
        encoding="utf-8",
    )
    value["dataset"]["manifest_digest"] = _digest(manifest)

    with pytest.raises(HarborSuiteError, match="manifest task set/digest mismatch"):
        verify_bundle(value, root)


def test_bundle_rejects_missing_snapshot_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_integrity, "task_digest", lambda _path: "a" * 64)
    (root / "snapshot-lock.json").unlink()

    with pytest.raises(HarborSuiteError, match=r"missing snapshot-lock\.json"):
        verify_bundle(value, root)


def test_bundle_rejects_snapshot_lock_task_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _manifest()
    root = _bundle(tmp_path, value)
    monkeypatch.setattr(heldout_integrity, "task_digest", lambda _path: "a" * 64)
    lock_path = root / "snapshot-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["tasks"][0]["digest"] = "sha256:" + "z" * 64
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    value["snapshot_lock"]["lock_digest"] = _digest(lock_path)

    with pytest.raises(HarborSuiteError, match="do not agree with snapshot lock"):
        verify_bundle(value, root)


def test_private_archive_rejects_workspace_escape(tmp_path: Path) -> None:
    source = tmp_path / "secret.txt"
    source.write_text("oracle", encoding="utf-8")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="../oracle.txt")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(HarborSuiteError, match="escapes output"):
        _safe_extract(archive, output)
