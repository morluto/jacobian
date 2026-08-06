"""Build a content-bound inventory for the registered benchmark portfolio."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.command_runner import ToolCommandStatus, run_operator_command
from benchmarks.tooling.harbor_suite import ROOT, Suite, load_registry, task_digest


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    result = run_operator_command("git", args, cwd=ROOT, timeout_seconds=30.0)
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError(
            result.diagnostic or result.stderr.decode(errors="replace")[:1024]
        )
    return result.stdout.decode("utf-8", errors="strict").strip()


def _suite_inventory(suite: Suite) -> dict[str, Any]:
    distributions: dict[str, Counter[str]] = {
        key: Counter()
        for key in (
            "domain",
            "primary_domain",
            "field",
            "assurance_ceiling",
            "answer_visibility",
            "provenance_class",
            "required_provider",
        )
    }
    tasks: list[dict[str, str]] = []
    for ref in suite.tasks:
        raw = tomllib.loads((ref.path / "task.toml").read_text(encoding="utf-8"))
        metadata = raw["metadata"]
        for key, counts in distributions.items():
            counts[str(metadata[key])] += 1
        tasks.append(
            {
                "id": ref.path.name,
                "name": ref.name,
                "primary_domain": ref.primary_domain,
                "field": ref.field,
                "digest": "sha256:" + task_digest(ref.path).removeprefix("sha256:"),
                "verifier": (ref.path / "tests" / "verifier.py")
                .relative_to(ROOT)
                .as_posix(),
                "verifier_digest": _sha256(ref.path / "tests" / "verifier.py"),
                "verifier_support_digest": _sha256(
                    ref.path / "tests" / "verifier_support.py"
                ),
                "submission_schema_digest": _sha256(
                    ref.path / "environment" / "submission_schema.json"
                ),
            }
        )
    jobs = {"oracle": _sha256(suite.job_oracle)}
    if suite.job_observation is not None:
        jobs["observation"] = _sha256(suite.job_observation)
    return {
        "id": suite.id,
        "name": suite.dataset_name,
        "evaluation_kind": suite.evaluation_kind,
        "claim_class": suite.claim_class,
        "task_count": len(tasks),
        "jobs": jobs,
        "distributions": {
            key: dict(sorted(counts.items())) for key, counts in distributions.items()
        },
        "tasks": sorted(tasks, key=lambda item: item["id"]),
    }


def build_inventory(
    *,
    dataset: str | None = None,
    primary_domain: str | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    suites = load_registry()
    if dataset is not None:
        short_dataset = dataset.removeprefix("jacobian/")
        matches = [suite for suite in suites if suite.id == short_dataset]
        if not matches:
            raise ValueError(f"unknown dataset filter: {dataset}")
        suites = tuple(matches)
    if primary_domain is not None:
        known = {ref.primary_domain for suite in suites for ref in suite.tasks}
        if primary_domain not in known:
            raise ValueError(f"unknown primary_domain filter: {primary_domain}")
    if field is not None:
        known = {ref.field for suite in suites for ref in suite.tasks}
        if field not in known:
            raise ValueError(f"unknown field filter: {field}")
    if primary_domain is not None or field is not None:
        filtered = []
        for suite in suites:
            refs = tuple(
                ref
                for ref in suite.tasks
                if (primary_domain is None or ref.primary_domain == primary_domain)
                and (field is None or ref.field == field)
            )
            if refs:
                filtered.append(dataclasses.replace(suite, tasks=refs))
        suites = tuple(filtered)
        if not suites:
            raise ValueError(
                "primary_domain/field filters select no tasks; check the combination"
            )
    datasets = [_suite_inventory(suite) for suite in suites]
    inventory = {
        "schema_version": "2",
        "source_sha": _git(["rev-parse", "HEAD"]),
        "source_dirty": bool(_git(["status", "--porcelain"])),
        "registry_digest": _sha256(ROOT / "benchmarks" / "registry.toml"),
        "dataset_count": len(datasets),
        "task_count": sum(item["task_count"] for item in datasets),
        "datasets": datasets,
    }
    schema = json.loads(
        (ROOT / "benchmarks" / "schemas" / "benchmark-inventory.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(inventory)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--primary-domain")
    parser.add_argument("--field")
    args = parser.parse_args()
    try:
        inventory = build_inventory(
            dataset=args.dataset,
            primary_domain=args.primary_domain,
            field=args.field,
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_inventory"]
