from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

UNREDUCED_RATIONAL_TASKS = (
    "random-function-expectation-audit",
    "polynomial-normalization",
    "series-domain-junk-zero",
    "positive-lower-density-separation",
    "cyclic-lipschitz-duality",
    "nonclosed-projection-image",
    "closed-one-form-polynomial-classification",
    "nondifferentiable-maximum-construction",
    "convergence-mode-separation",
    "continuous-spike-integral-separation",
    "natural-subtraction-proof-repair",
    "monotone-inverse-continuity-audit",
    "gaussian-moment-generality-audit",
    "algebraic-independence-transfer-audit",
    "exponential-moment-rationality",
    "fiber-dimension-semicontinuity-repair",
    "ga-action-local-finiteness-certificate",
    "lagrangian-projection-proof-audit",
    "limsup-quantifier-alignment",
    "lp-integrability-separator",
    "marginal-joint-product-audit",
    "polynomial-tail-counterexample",
    "symbolic-block-determinant-decomposition",
    "fractional-ratio-proof-repair",
    "ternary-distance-code-optimum",
    "apollonius-gap-repair",
    "emerald-path-family-audit",
)


def _scale_first_rational(value: object, factor: int = 2) -> bool:
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"numerator", "denominator"} and type(value["numerator"]) is int:
            value["numerator"] *= factor
            value["denominator"] *= factor
            return True
        return any(_scale_first_rational(item, factor) for item in value.values())
    if isinstance(value, list):
        return any(_scale_first_rational(item, factor) for item in value)
    return False


@pytest.mark.parametrize("task_name", UNREDUCED_RATIONAL_TASKS)
def test_unreduced_structured_rational_keeps_reward(
    tmp_path: Path, task_name: str
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, task_name, "computed")
    oracle = _verifier._run_verifier(task, app, logs)
    assert oracle.reward == 1.0

    submission = json.loads((app / "submission.json").read_text())
    assert _scale_first_rational(submission["result"])
    _fixtures._write_json(app / "submission.json", submission)
    scaled = _verifier._run_verifier(task, app, logs)
    assert scaled.reward == 1.0
