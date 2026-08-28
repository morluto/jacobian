"""Native multiple-testing operations over canonical hypothesis values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, format_canonical_rational
from jacobian.math._labels import OpaqueLabel
from jacobian.math.probability.multiple_testing._models import (
    MAX_HYPOTHESES,
    BHStepUpResult,
    FDPResult,
    HypothesisSpec,
)


def bh_step_up(
    hypotheses: tuple[HypothesisSpec, ...], level: CanonicalRational
) -> BHStepUpResult:
    """Benjamini-Hochberg step-up procedure.

    Sorts p-values ascending, finds the largest k such that
    p_(k) <= k * q / n, and rejects all hypotheses with p <= p_(k).
    """
    if not hypotheses or len(hypotheses) > MAX_HYPOTHESES:
        raise ValueError(
            f"hypotheses must contain between 1 and {MAX_HYPOTHESES} entries"
        )
    ids = tuple(hypothesis.hypothesis_id for hypothesis in hypotheses)
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis IDs must be unique")
    level_value = level.as_fraction()
    if not 0 <= level_value <= 1:
        raise ValueError("level must be in [0, 1]")
    n = len(hypotheses)

    hyps = sorted(hypotheses, key=lambda h: h.p_value.as_fraction())
    level_value = Fraction(level_value)

    critical_k = 0
    cutoff = Fraction(0)

    for i, hyp in enumerate(hyps, 1):
        threshold = Fraction(i) * level_value / Fraction(n)
        if hyp.p_value.as_fraction() <= threshold:
            critical_k = i
            cutoff = hyp.p_value.as_fraction()

    rejected_ids = tuple(h.hypothesis_id for h in hyps[:critical_k])

    return BHStepUpResult(
        critical_index=critical_k,
        cutoff_threshold=format_canonical_rational(cutoff),
        rejected=tuple(sorted(rejected_ids)),
        total_hypotheses=n,
    )


def false_discovery_proportion(
    rejected_ids: tuple[OpaqueLabel, ...], true_null_ids: tuple[OpaqueLabel, ...]
) -> FDPResult:
    """Compute the false discovery proportion."""
    rejected = set(rejected_ids)
    nulls = set(true_null_ids)
    false_d = len(rejected & nulls)
    total = len(rejected)
    fdp = Fraction(false_d, total) if total > 0 else Fraction(0)
    return FDPResult(
        false_discoveries=false_d,
        total_rejections=total,
        fdp=format_canonical_rational(fdp),
    )


__all__ = ["bh_step_up", "false_discovery_proportion"]
