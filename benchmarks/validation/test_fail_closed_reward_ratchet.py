"""Inventory ratchet: leaky weighted-reward formulas must not grow.

The historical leaky template soft-weights evidence (and other mandatory
dimensions) into aggregate reward so invalid digests still score ~0.9. All
known instances have been migrated to hard-gate evidence validity; this module
fails if any task reintroduces the pattern. The detection is base-constant-
agnostic: any additive reward that soft-weights evidence is flagged regardless
of the numeric base (0.7, 0.8, 0.6, etc.) or the evidence coefficient.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "benchmarks" / "datasets"
TEMPLATE_SUPPORT = (
    ROOT / "benchmarks" / "templates" / "task" / "tests" / "verifier_support.py"
)
TEMPLATE_VERIFIER = ROOT / "benchmarks" / "templates" / "task" / "tests" / "verifier.py"

# Soft-weighted multiplicative aggregate that does not hard-gate evidence
# (RC1 / A1). Base-constant-agnostic: any ``<base> * correct ... <coeff> *
# evidence`` shape is leaky because invalid evidence still earns partial reward.
# Variable-name-agnostic: catches ``ev``, ``evidence``, ``evidence_valid``,
# ``evidence_ok``, and ``good`` so aliases cannot evade detection.
_LEAKY_WEIGHTED = re.compile(
    r"\d+\.\d+\s*\*\s*(?:correct|math_correct|float\(\s*correct)"
    r".{0,120}?"
    r"\d+\.\d+\s*\*\s*(?:good|evidence|ev)\b",
    re.DOTALL,
)
_LEAKY_WEIGHTED_ALT = re.compile(
    r"\d+\.\d+\s*\*\s*(?:correct|math_correct).{0,200}?"
    r"\d+\.\d+\s*\*\s*(?:good|evidence_ok|evidence_valid|ev)\b",
    re.DOTALL,
)

# Additive soft-weighted aggregate: any ``<base> + ... <coeff> * evidence``
# shape gated only on correctness, so an invalid digest still scores partial
# reward (RC1 / A1, issue #538). Base-constant-agnostic so future verifiers
# cannot evade detection by changing the base from 0.7 to 0.8, 0.6, etc.
# Variable-name-agnostic: catches ``ev``, ``evidence``, ``evidence_valid``,
# ``evidence_ok``, and ``good`` so aliases cannot evade detection.
_LEAKY_ADDITIVE = re.compile(
    r"\d+\.\d+\s*\+.{0,120}?\d+\.\d+\s*\*\s*"
    r"(?:evidence|evidence_valid|evidence_ok|ev|good)\b",
    re.DOTALL,
)

# All known leaky reward verifiers have been migrated to hard-gate evidence
# validity. The inventory is empty; any new occurrence is unexpected growth.
KNOWN_LEAKY_REWARD_VERIFIERS: frozenset[str] = frozenset()

_REQUIRED_TEMPLATE_EXPORTS = frozenset(
    {
        "aggregate_reward",
        "evidence_list_is_bound",
        "load_submission",
        "strict_submission_contract",
    }
)


def _is_leaky(text: str) -> bool:
    return bool(
        _LEAKY_WEIGHTED.search(text)
        or _LEAKY_WEIGHTED_ALT.search(text)
        or _LEAKY_ADDITIVE.search(text)
    )


def _leaky_task_ids() -> set[str]:
    found: set[str] = set()
    for path in sorted(DATASETS.rglob("tests/verifier.py")):
        relative = path.relative_to(DATASETS)
        # datasets/<dataset>/<task>/tests/verifier.py
        if len(relative.parts) < 4:
            continue
        task_id = f"{relative.parts[0]}/{relative.parts[1]}"
        if _is_leaky(path.read_text(encoding="utf-8", errors="replace")):
            found.add(task_id)
    return found


def test_template_support_exports_fail_closed_aggregate_helper() -> None:
    text = TEMPLATE_SUPPORT.read_text(encoding="utf-8")
    assert "def aggregate_reward(" in text
    assert not _is_leaky(text)
    for name in _REQUIRED_TEMPLATE_EXPORTS:
        assert f'"{name}"' in text or f"'{name}'" in text
    # Template verifier remains a stub; it must not ship the leaky formula.
    assert not _is_leaky(TEMPLATE_VERIFIER.read_text(encoding="utf-8"))


def test_leaky_reward_inventory_does_not_grow() -> None:
    found = _leaky_task_ids()
    unexpected = sorted(found - KNOWN_LEAKY_REWARD_VERIFIERS)
    missing = sorted(KNOWN_LEAKY_REWARD_VERIFIERS - found)
    assert not unexpected, (
        "Leaky soft-evidence reward formulas appeared; migrate them to "
        f"aggregate_reward or hard-gate evidence validity: {unexpected}"
    )
    # The inventory is empty after full migration; any stale entry is a
    # regression in the ratchet itself.
    assert not missing, (
        "Known leaky inventory is stale (tasks already migrated). Remove them "
        f"from KNOWN_LEAKY_REWARD_VERIFIERS: {missing}"
    )
