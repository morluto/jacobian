"""Comparison and reporting for normalized observation evidence."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.errors import HarborSuiteError

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "observation-evidence.schema.json"
)
CORE_METRICS = ("correctness", "false_certification")


def _validate_contract(value: dict[str, Any]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read evidence schema: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=str)
    if errors:
        raise HarborSuiteError(
            "observation evidence violates its public schema: "
            + "; ".join(error.message for error in errors[:5])
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _trial_metric(trial: dict[str, Any], metric: str) -> float | None:
    if metric in {
        "correctness",
        "witness_validity",
        "scope_accuracy",
        "assurance_calibration",
        "reward",
    }:
        rewards = trial.get("rewards")
        return _number(rewards.get(metric)) if isinstance(rewards, dict) else None
    if metric == "false_certification":
        return _number(trial.get(metric))
    if metric in {"cost_usd", "agent_seconds"}:
        return _number(trial.get(metric))
    if metric.startswith("tokens."):
        tokens = trial.get("tokens")
        return (
            _number(tokens.get(metric.split(".", 1)[1]))
            if isinstance(tokens, dict)
            else None
        )
    return None


def _comparison_failures(
    control: dict[str, Any], treatment: dict[str, Any]
) -> list[str]:
    failures = [
        f"{name} evidence is not VALID"
        for name, value in (("control", control), ("treatment", treatment))
        if value.get("status") != "VALID"
    ]
    failures.extend(
        f"{name} evidence claims VALID but has non-COMPLETED trials"
        for name, value in (("control", control), ("treatment", treatment))
        if value.get("status") == "VALID"
        and any(
            isinstance(trial, dict) and trial.get("status") != "COMPLETED"
            for trial in value.get("trials", [])
        )
    )
    if (control.get("condition"), treatment.get("condition")) not in {
        ("control", "treatment"),
        ("C1", "C2"),
    }:
        failures.append("conditions must be a distinct control/treatment or C1/C2 pair")
    failures.extend(
        f"{name} evidence has an invalid public-claim boundary"
        for name, value in (("control", control), ("treatment", treatment))
        if value.get("causal_claim_authorized") is not False
    )
    failures.extend(
        f"fixed invariant differs: {key}"
        for key in ("source_sha", "dataset", "snapshot_id", "harbor_version")
        if control.get(key) != treatment.get(key)
    )
    if control.get("fixed_invariants") != treatment.get("fixed_invariants"):
        failures.append("fixed invariants differ")
    if control.get("job", {}).get("comparison_signature") != treatment.get(
        "job", {}
    ).get("comparison_signature"):
        failures.append("job configuration differs outside the condition allowlist")
    classes = {control.get("evidence_class"), treatment.get("evidence_class")}
    if len(classes) != 1:
        failures.append("evidence classes differ")
    return failures


def _indexed_trials(value: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(item["task"]), int(item["repetition"])): item
        for item in value.get("trials", [])
    }


def _duplicate_pair_keys(value: dict[str, Any]) -> list[tuple[str, int]]:
    keys = [
        (str(item["task"]), int(item["repetition"])) for item in value.get("trials", [])
    ]
    return sorted(key for key, count in Counter(keys).items() if count > 1)


def _derived_comparison_class(
    control: dict[str, Any], treatment: dict[str, Any]
) -> str:
    classes = {control.get("evidence_class"), treatment.get("evidence_class")}
    if classes == {"held-out-comparative-evaluation"}:
        return "held-out-comparison"
    return "public-workflow-comparison"


def _task_average(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [row for row in rows if row["paired_delta"] is not None]
    return {
        "weighting": "equal-observed-task-means",
        "task_count": len(rows),
        "observed_task_count": len(observed),
        "missing_task_count": len(rows) - len(observed),
        **{
            key: sum(row[key] for row in observed) / len(observed) if observed else None
            for key in ("control_mean", "treatment_mean", "paired_delta")
        },
    }


def _family_map(
    value: dict[str, Any], failures: list[str], name: str
) -> dict[str, str] | None:
    binding = value.get("task_family_binding")
    if binding is None:
        return None
    for runtime in (value["runtime_snapshot"], value["fixed_invariants"]["runtime"]):
        if (
            runtime.get("manifest_digest") is not None
            and runtime["manifest_digest"] != binding["manifest_digest"]
        ):
            failures.append(
                f"{name} family binding differs from its frozen runtime manifest"
            )
            return None
    rows = binding["tasks"]
    mapping = {row["task"]: row["family"] for row in rows}
    source = {row["task"]: row["digest"] for row in value["fixed_invariants"]["tasks"]}
    if (
        len(mapping) != len(rows)
        or set(mapping) != set(source)
        or len(source) != len(value["fixed_invariants"]["tasks"])
    ):
        failures.append(
            f"{name} family binding does not cover the frozen task set exactly"
        )
        return None
    if any(row["digest"] != source[row["task"]] for row in rows):
        failures.append(f"{name} family binding task digest differs")
        return None
    if any(
        trial["task"] not in source or trial["task_digest"] != source[trial["task"]]
        for trial in value["trials"]
    ):
        failures.append(f"{name} trial does not bind the frozen family task")
        return None
    if {trial["task"] for trial in value["trials"]} != set(source):
        failures.append(f"{name} family binding has missing task observations")
        return None
    return mapping


def _metric_report(
    metric: str,
    pairs: list[tuple[str, int]],
    control_trials: dict[tuple[str, int], dict[str, Any]],
    treatment_trials: dict[tuple[str, int], dict[str, Any]],
    families: dict[str, str] | None,
) -> dict[str, Any]:
    values = [
        (
            _trial_metric(control_trials[pair], metric),
            _trial_metric(treatment_trials[pair], metric),
        )
        for pair in pairs
    ]
    complete = [
        (left, right)
        for left, right in values
        if left is not None and right is not None
    ]
    left = [item[0] for item in complete]
    right = [item[1] for item in complete]
    deltas = [treatment - control for control, treatment in complete]
    grouped: dict[str, list[tuple[float, float]]] = {}
    counts = Counter(task for task, _ in pairs)
    for (task, _), (left_value, right_value) in zip(pairs, values, strict=True):
        if left_value is not None and right_value is not None:
            grouped.setdefault(task, []).append((left_value, right_value))
    task_summaries: list[dict[str, Any]] = []
    for task, count in sorted(counts.items()):
        observed = grouped.get(task, [])
        task_summaries.append(
            {
                "task": task,
                "pair_count": len(observed),
                "missing_pair_count": count - len(observed),
                "control_mean": sum(left for left, _ in observed) / len(observed)
                if observed
                else None,
                "treatment_mean": sum(right for _, right in observed) / len(observed)
                if observed
                else None,
                "paired_delta": sum(right - left for left, right in observed)
                / len(observed)
                if observed
                else None,
            }
        )
    family_summaries = []
    if families is not None:
        for family in sorted(set(families.values())):
            members = [row for row in task_summaries if families[row["task"]] == family]
            family_summaries.append({"family": family, **_task_average(members)})
    return {
        "pair_count": len(deltas),
        "missing_pair_count": len(pairs) - len(deltas),
        "weighting": "equal-observed-pairs",
        "tasks": task_summaries,
        "task_average": _task_average(task_summaries),
        "families": family_summaries if families is not None else None,
        "control_mean": sum(left) / len(left) if left else None,
        "treatment_mean": sum(right) / len(right) if right else None,
        "paired_delta": sum(deltas) / len(deltas) if deltas else None,
        "bootstrap_95_interval": None,
        "mcnemar_exact_p": None,
        "interpretation": "descriptive-only",
        "inference_unavailable_reason": "No frozen sampling estimand or cluster inference protocol is declared; repetitions and family labels do not establish independent observations.",
    }


def compare_evidence(
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, Any]:
    _validate_contract(control)
    _validate_contract(treatment)
    failures = _comparison_failures(control, treatment)
    for name, value in (("control", control), ("treatment", treatment)):
        duplicates = _duplicate_pair_keys(value)
        if duplicates:
            failures.append(f"{name} evidence has duplicate task/repetition pairs")
    control_trials = _indexed_trials(control)
    treatment_trials = _indexed_trials(treatment)
    if set(control_trials) != set(treatment_trials):
        failures.append("control/treatment trials do not pair exactly")
    pairs = sorted(set(control_trials) & set(treatment_trials))
    control_families = _family_map(control, failures, "control")
    treatment_families = _family_map(treatment, failures, "treatment")
    families = None
    if control_families is not None and treatment_families is not None:
        if (
            control_families != treatment_families
            or control["task_family_binding"]["manifest_digest"]
            != treatment["task_family_binding"]["manifest_digest"]
        ):
            failures.append("frozen task-family bindings differ")
        else:
            families = control_families
    elif (control.get("task_family_binding") is None) != (
        treatment.get("task_family_binding") is None
    ):
        failures.append("task-family binding is missing from one comparison arm")
    task_counts = Counter(task for task, _ in pairs)
    metric_names = (
        "correctness",
        "witness_validity",
        "scope_accuracy",
        "assurance_calibration",
        "false_certification",
        "reward",
        "tokens.input",
        "tokens.output",
        "cost_usd",
        "agent_seconds",
    )
    metrics = {
        metric: _metric_report(
            metric, pairs, control_trials, treatment_trials, families
        )
        for metric in metric_names
    }
    for metric in CORE_METRICS:
        if metrics[metric]["pair_count"] != len(pairs):
            failures.append(f"core metric is missing from a complete pair: {metric}")
    return {
        "schema_version": "3",
        "evidence_class": _derived_comparison_class(control, treatment),
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INVALID",
        "dataset": control.get("dataset"),
        "source_sha": control.get("source_sha"),
        "snapshot_id": (
            control["snapshot_id"]
            if control["snapshot_id"] == treatment["snapshot_id"]
            else None
        ),
        "conditions": {
            "control": control.get("condition"),
            "treatment": treatment.get("condition"),
        },
        "pair_count": len(pairs),
        "task_count": len(task_counts),
        "family_count": len(set(families.values())) if families is not None else None,
        "family_count_meaning": "declared labels, not established independent units",
        "repetitions_per_task": dict(sorted(task_counts.items())),
        "metrics": metrics,
        "validation_failures": failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Jacobian workflow comparison",
        "",
        f"Status: **{report['status']}**. This report remains evaluation evidence; it does not itself authorize a causal operation claim.",
        "",
        f"Benchmark snapshot: {report['snapshot_id'] or 'no agreed snapshot'}",
        "",
        f"Paired observations: {report['pair_count']}; tasks: {report['task_count']}; declared families: {report['family_count'] if report['family_count'] is not None else 'unknown'}.",
        "Family labels do not prove independence. All summaries are descriptive; no inferential interval or p-value is reported.",
        "Raw means below weight each observed pair equally. JSON task averages weight observed task means equally; family summaries use the same weighting within each family. Missing metric pairs are excluded and counted explicitly.",
        "",
        "| Metric | Pairs | Control | Treatment | Paired delta | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, metric in report["metrics"].items():

        def fmt(value: Any) -> str:
            return "unknown" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {name} | {metric['pair_count']} | {fmt(metric['control_mean'])} | {fmt(metric['treatment_mean'])} | {fmt(metric['paired_delta'])} | {metric['interpretation']} |"
        )
    if report["validation_failures"]:
        lines.extend(["", "## Validation failures", ""])
        lines.extend(f"- {failure}" for failure in report["validation_failures"])
    return "\n".join(lines) + "\n"


__all__ = ["compare_evidence", "render_markdown"]
