"""Validate pinned external benchmark adapter contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "benchmarks" / "adapters"
SCHEMA = ROOT / "benchmarks" / "schemas" / "source-adapter-lock.schema.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_lock(lock_path: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{_display(lock_path)}: invalid JSON: {exc}"]
    if not isinstance(lock, dict):
        return None, [f"{_display(lock_path)}: lock must be a JSON object"]
    failures: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(lock),
        key=lambda item: list(item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path)
        failures.append(f"{_display(lock_path)} at {location}: {error.message}")
    return lock, failures


def _parity_evidence_failures(
    lock_path: Path, evidence_path: Path, payload: dict[str, object]
) -> list[str]:
    failures: list[str] = []
    expected_digest = payload.get("task_digest")
    if payload.get("task_digest_matches") is False:
        failures.append(
            f"{_display(lock_path)}: {_display(evidence_path)} reports "
            "task_digest_matches=false"
        )
    for key in ("source_task_digest", "generated_task_digest", "task_digest"):
        value = payload.get(key)
        if (
            value is not None
            and expected_digest is not None
            and value != expected_digest
        ):
            failures.append(
                f"{_display(lock_path)}: {_display(evidence_path)} {key} "
                f"{value!r} does not match lock output task_digest "
                f"{expected_digest!r}"
            )
    return failures


def _evidence_failures(
    adapter: Path, lock_path: Path, output: dict[str, object]
) -> list[str]:
    failures: list[str] = []
    for field, filename in (
        ("oracle_evidence_digest", "oracle-evidence.json"),
        ("parity_evidence_digest", "parity-evidence.json"),
    ):
        evidence_path = adapter / filename
        if not evidence_path.is_file():
            failures.append(
                f"{_display(lock_path)}: required evidence file is missing: "
                f"{_display(evidence_path)}"
            )
            continue
        if evidence_path.stat().st_size == 0:
            failures.append(
                f"{_display(lock_path)}: evidence file is empty: "
                f"{_display(evidence_path)}"
            )
            continue
        if output.get(field) != _sha256(evidence_path):
            failures.append(
                f"{_display(lock_path)}: {field} mismatch for {_display(evidence_path)}"
            )
            continue
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                f"{_display(lock_path)}: {_display(evidence_path)} is not JSON: {exc}"
            )
            continue
        if not isinstance(payload, dict):
            failures.append(
                f"{_display(lock_path)}: {_display(evidence_path)} must be a JSON object"
            )
            continue
        # Semantic provenance: digests inside evidence must match the lock's
        # current output task digest, not only the evidence file hash.
        expected_digest = output.get("task_digest")
        declared = (
            payload.get("task_digest")
            or payload.get("output_task_digest")
            or payload.get("source_task_digest")
        )
        if (
            declared is not None
            and expected_digest is not None
            and declared != expected_digest
        ):
            failures.append(
                f"{_display(lock_path)}: {_display(evidence_path)} task_digest "
                f"{declared!r} does not match lock output task_digest "
                f"{expected_digest!r}"
            )
        if filename == "parity-evidence.json":
            failures.extend(
                _parity_evidence_failures(lock_path, evidence_path, payload)
            )
    return failures


def _semantic_failures(
    adapter: Path, lock_path: Path, lock: dict[str, object]
) -> list[str]:
    failures: list[str] = []
    if lock.get("adapter_id") != adapter.name:
        failures.append(f"{_display(lock_path)}: adapter_id must match directory")
    selection = lock.get("selection", {})
    if isinstance(selection, dict):
        overlap = set(selection.get("included_rows", [])) & set(
            selection.get("excluded_rows", [])
        )
        if overlap:
            failures.append(
                f"{_display(lock_path)}: rows cannot be included and excluded: {sorted(overlap)}"
            )
    outputs = lock.get("outputs", [])
    if isinstance(outputs, list):
        from benchmarks.tooling.harbor_suite import task_digest

        pairs: set[tuple[str, str]] = set()
        for output in outputs:
            if not isinstance(output, dict):
                continue
            pair = (str(output.get("dataset")), str(output.get("task_id")))
            if pair in pairs:
                failures.append(
                    f"{_display(lock_path)}: duplicate output {pair[0]}/{pair[1]}"
                )
            pairs.add(pair)
            task_path = ROOT / "benchmarks" / "datasets" / pair[0] / pair[1]
            if not task_path.is_dir() or not (task_path / "task.toml").is_file():
                failures.append(
                    f"{_display(lock_path)}: adapter output task is missing: "
                    f"{pair[0]}/{pair[1]}"
                )
                continue
            actual_digest = "sha256:" + task_digest(task_path).removeprefix("sha256:")
            if output.get("task_digest") != actual_digest:
                failures.append(
                    f"{_display(lock_path)}: task digest mismatch for "
                    f"{pair[0]}/{pair[1]}"
                )
            failures.extend(_evidence_failures(adapter, lock_path, output))
    return failures


def _failures(adapter: Path) -> list[str]:
    lock_path = adapter / "source.lock.json"
    required = (lock_path, adapter / "generate.py", adapter / "check.sh")
    missing = [
        f"{_display(path)}: required adapter file missing"
        for path in required
        if not path.is_file()
    ]
    if missing:
        return missing
    lock, failures = _load_lock(lock_path)
    if lock is not None:
        failures.extend(_semantic_failures(adapter, lock_path, lock))
    return failures


def check(adapter_id: str | None = None) -> list[str]:
    if adapter_id is not None:
        candidates = [ADAPTERS / adapter_id]
    else:
        candidates = sorted(path for path in ADAPTERS.iterdir() if path.is_dir())
    failures: list[str] = []
    for adapter in candidates:
        if not adapter.is_dir():
            failures.append(f"unknown benchmark adapter: {adapter.name}")
            continue
        failures.extend(_failures(adapter))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    failures = check(args.adapter)
    if failures:
        print("Benchmark adapter failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Benchmark adapter contracts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
