import json
import re
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The verifier checks a complete finite incidence factorization and the general counting formula but does not replay the universal theorem in Lean."


def _is_int(value: object) -> bool:
    """Accept only genuine integers, rejecting JSON booleans.

    Python treats ``True == 1`` and ``False == 0``, so plain equality or
    isinstance checks would let booleans pass the agent-visible
    ``enum: [-1, 1]`` and numeric contracts.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _source() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _factorization(order: list[int], weights: list[int]) -> bool:
    size = len(order)
    zeta = [[int(t & a == t) for t in order] for a in order]
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            reconstructed = sum(
                zeta[i][k] * weights[k] * zeta[j][k] for k in range(size)
            )
            if reconstructed != int(bool(a & b)):
                return False
    return all(
        zeta[i][i] == 1 and all(zeta[i][j] == 0 for j in range(i + 1, size))
        for i in range(size)
    )


def _trace_valid(trace: list, expected_trace: list) -> bool:
    # Thread PRRT_kwDOThEfjc6Vu43n: reject booleans in trace numeric fields.
    trace_by_n = {}
    for entry in trace:
        if not isinstance(entry, dict) or set(entry) != {
            "n",
            "even_nonempty_count",
            "determinant",
        }:
            return False
        if not (
            _is_int(entry["n"])
            and _is_int(entry["even_nonempty_count"])
            and _is_int(entry["determinant"])
        ):
            return False
        if entry["n"] in trace_by_n:
            return False
        trace_by_n[entry["n"]] = entry
    return trace_by_n == {entry["n"]: entry for entry in expected_trace}


def _result(value: object, source: dict[str, Any]) -> bool:
    required = {
        "sample_n",
        "mask_order",
        "diagonal_weights",
        "trace",
        "general_even_count",
        "general_determinant",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    provenance = source.get("source", {})
    n = value.get("sample_n")
    if (
        provenance.get("revision") != "dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c"
        or not _is_int(value["sample_n"])
        or not 1 <= n <= 8
    ):
        return False
    order = value["mask_order"]
    weights = value["diagonal_weights"]
    # Thread PRRT_kwDOThEfjc6Vu43n: reject booleans in mask_order.
    if not isinstance(order, list) or not all(_is_int(m) for m in order):
        return False
    # Thread PRRT_kwDOThEfjc6VuwyR: reject booleans in diagonal_weights.
    if not isinstance(weights, list) or not all(_is_int(w) for w in weights):
        return False
    expected_order = sorted(range(1, 2**n), key=lambda mask: (mask.bit_count(), mask))
    expected_weights = [1 if mask.bit_count() % 2 else -1 for mask in expected_order]
    if (
        order != expected_order
        or weights != expected_weights
        or not _factorization(order, weights)
    ):
        return False
    trace = value["trace"]
    if not isinstance(trace, list) or len(trace) != source.get("trace_max_n"):
        return False
    expected_trace = [
        {
            "n": k,
            "even_nonempty_count": 2 ** (k - 1) - 1,
            "determinant": 1 if k == 1 else -1,
        }
        for k in range(1, source["trace_max_n"] + 1)
    ]
    if not _trace_valid(trace, expected_trace):
        return False
    return bool(
        value["general_even_count"] == "2^(n-1)-1"
        and value["general_determinant"] == "1_if_n_eq_1_else_minus_1"
    )


def _evidence(value: object) -> bool:
    # Thread PRRT_kwDOThEfjc6Vu43q: enforce the agent-visible maxItems: 1
    # contract on the evidence list before resolving any descriptor.
    if not isinstance(value, list) or len(value) != 1:
        return False
    if not evidence_list_is_bound(value):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text().lower()
    except (OSError, UnicodeError):
        return False
    return (
        len(text) >= 100
        and all(term in text for term in ("inclusion-exclusion", "zeta", "computed"))
        and re.search(r"\b(?:even|nonempty)\b.{0,80}\bsubsets?\b", text)
        and "determinant" in text
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    correct = bool(contract and _result(data.get("result"), _source()))
    evidence = bool(correct and _evidence(data.get("evidence")))
    scope = bool(
        contract
        and isinstance(data.get("scope"), str)
        and "nonempty" in data["scope"].casefold()
        and "subset" in data["scope"].casefold()
        and "intersection" in data["scope"].casefold()
    )
    assurance = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(
            isinstance(item, str)
            and "finite" in item.casefold()
            and re.search(r"\b(?:not|doesn['']?t|without|cannot)\b", item, re.I)
            and "lean" in item.casefold()
            for item in data["limitations"]
        )
    )
    false_verified = false_verified_claim(submission, verification_record_bound=False)
    passed = bool(
        correct
        and evidence
        and scope
        and assurance
        and limitations
        and not false_verified
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": float(passed),
                "false_certification": false_verified,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
