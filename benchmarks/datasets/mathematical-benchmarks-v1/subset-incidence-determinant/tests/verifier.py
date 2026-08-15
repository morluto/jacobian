import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


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


def _general_formulas_valid(even_count: object, determinant: object) -> bool:
    if (
        not isinstance(even_count, dict)
        or set(even_count) != {"base", "exponent_offset", "constant_offset"}
        or not all(type(value) is int for value in even_count.values())
        or not isinstance(determinant, dict)
        or set(determinant) != {"n_equals_1", "otherwise"}
        or not all(_is_int(value) for value in determinant.values())
    ):
        return False
    return bool(
        even_count == {"base": 2, "exponent_offset": -1, "constant_offset": -1}
        and determinant == {"n_equals_1": 1, "otherwise": -1}
    )


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
    expected_masks = set(range(1, 2**n))
    expected_weights = [1 if mask.bit_count() % 2 else -1 for mask in order]
    if (
        set(order) != expected_masks
        or len(order) != len(expected_masks)
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
    return _general_formulas_valid(
        value["general_even_count"], value["general_determinant"]
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    correct = bool(submission and _result(data.get("result"), _source()))
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "reward": float(correct),
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
