"""Comparison and reporting for normalized observation evidence."""

from __future__ import annotations

import json
import math
import random
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
        "evidence_validity",
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


def _bootstrap_interval(deltas: list[float]) -> list[float] | None:
    if len(deltas) < 10:
        return None
    rng = random.Random(0)
    means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(2000)
    )
    return [means[49], means[1949]]


def _mcnemar_exact(control: list[float], treatment: list[float]) -> float | None:
    discordant = [
        (left >= 1.0, right >= 1.0)
        for left, right in zip(control, treatment, strict=True)
        if left in {0.0, 1.0} and right in {0.0, 1.0}
    ]
    b = sum(left and not right for left, right in discordant)
    c = sum(not left and right for left, right in discordant)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


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
        for key in ("source_sha", "dataset")
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


def _metric_report(
    metric: str,
    pairs: list[tuple[str, int]],
    control_trials: dict[tuple[str, int], dict[str, Any]],
    treatment_trials: dict[tuple[str, int], dict[str, Any]],
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
    return {
        "pair_count": len(deltas),
        "control_mean": sum(left) / len(left) if left else None,
        "treatment_mean": sum(right) / len(right) if right else None,
        "paired_delta": sum(deltas) / len(deltas) if deltas else None,
        "bootstrap_95_interval": _bootstrap_interval(deltas),
        "mcnemar_exact_p": _mcnemar_exact(left, right)
        if metric in {"correctness", "false_certification"} and left
        else None,
        "interpretation": "descriptive-small-sample"
        if len(deltas) < 10
        else "comparative",
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
    metric_names = (
        "correctness",
        "evidence_validity",
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
        metric: _metric_report(metric, pairs, control_trials, treatment_trials)
        for metric in metric_names
    }
    for metric in CORE_METRICS:
        if metrics[metric]["pair_count"] != len(pairs):
            failures.append(f"core metric is missing from a complete pair: {metric}")
    return {
        "schema_version": "1",
        "evidence_class": _derived_comparison_class(control, treatment),
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INVALID",
        "dataset": control.get("dataset"),
        "source_sha": control.get("source_sha"),
        "conditions": {
            "control": control.get("condition"),
            "treatment": treatment.get("condition"),
        },
        "pair_count": len(pairs),
        "metrics": metrics,
        "validation_failures": failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Jacobian workflow comparison",
        "",
        f"Status: **{report['status']}**. This report remains evaluation evidence; it does not itself authorize a causal capability claim.",
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
