"""Domain-owned multiple testing operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.math.multiple_testing._models import (
    BHStepUpRequest,
    BHStepUpResult,
    FDPRequest,
    FDPResult,
)


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
    )


def compute_bh_step_up(request: BHStepUpRequest) -> BHStepUpResult:
    """Benjamini-Hochberg step-up procedure.

    Sorts p-values ascending, finds the largest k such that
    p_(k) <= k * q / n, and rejects all hypotheses with p <= p_(k).
    """
    n = len(request.hypotheses)
    level = request.level.as_fraction()

    hyps = sorted(request.hypotheses, key=lambda h: h.p_value.as_fraction())
    level = Fraction(level)

    critical_k = 0
    cutoff = Fraction(0)

    for i, hyp in enumerate(hyps, 1):
        threshold = Fraction(i) * level / Fraction(n)
        if hyp.p_value.as_fraction() <= threshold:
            critical_k = i
            cutoff = hyp.p_value.as_fraction()

    rejected_ids = tuple(h.hypothesis_id for h in hyps[:critical_k])

    return BHStepUpResult(
        critical_index=critical_k,
        cutoff_threshold=_format_rational(cutoff),
        rejected=tuple(sorted(rejected_ids)),
        total_hypotheses=n,
    )


def compute_fdp(request: FDPRequest) -> FDPResult:
    """Compute the false discovery proportion."""
    rejected = set(request.rejected_ids)
    nulls = set(request.true_null_ids)
    false_d = len(rejected & nulls)
    total = len(rejected)
    fdp = Fraction(false_d, total) if total > 0 else Fraction(0)
    return FDPResult(
        false_discoveries=false_d,
        total_rejections=total,
        fdp=_format_rational(fdp),
    )


__all__ = ["compute_bh_step_up", "compute_fdp"]
