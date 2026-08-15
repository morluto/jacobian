import json
import math
from pathlib import Path
from typing import Any

from verifier_support import load_submission, normalize_reward_file

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _load_input() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _witness(value: object, *, unit: bool) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "p",
        "d",
        "d_p",
        "C",
        "left_residue",
        "right_residue",
    }:
        return False
    if any(type(value[k]) is not int for k in value):
        return False
    p, d, dp, c = value["p"], value["d"], value["d_p"], value["C"]
    if not (
        3 <= p <= 43
        and _prime(p)
        and p % 2
        and 1 <= d <= 80
        and math.gcd(d, p - 1) == 1
        and dp == d % (p - 1)
        and 1 <= dp <= p - 2
    ):
        return False
    if (math.gcd(c, p) == 1) is not unit:
        return False
    left, right = pow(c, d, p), pow(c, dp, p)
    return (
        value["left_residue"] == left
        and value["right_residue"] == right
        and left == right
    )


def _bounded_sanity(source: dict[str, Any]) -> bool:
    bounds = source.get("sanity_bounds", {})
    if bounds != {"maximum_prime": 43, "maximum_exponent": 80}:
        return False
    for p in range(3, 44, 2):
        if not _prime(p):
            continue
        for d in range(1, 81):
            if math.gcd(d, p - 1) != 1:
                continue
            dp = d % (p - 1)
            if not 1 <= dp <= p - 2:
                return False
            for c in range(p):
                if pow(c, d, p) != pow(c, dp, p):
                    return False
    return True


def _result(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "diagnosis",
        "remainder_bounds",
        "unit_branch",
        "nonunit_branch",
        "domain_split",
        "unit_witness",
        "nonunit_witness",
        "finite_testing_role",
    }:
        return False
    provenance = source.get("source", {})
    return bool(
        provenance.get("revision") == "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        and provenance.get("row") == 7
        and provenance.get("source_id") == 780
        and value["diagnosis"]
        == {
            "unsafe_step": "NEGATIVE_POWER_REQUIRES_UNIT",
            "missing_domain": "C_CONGRUENT_ZERO_MOD_P",
        }
        and value["remainder_bounds"]
        == {"lower": 1, "upper": "p-2", "reason": "COPRIMALITY_EXCLUDES_ZERO_REMAINDER"}
        and value["unit_branch"]
        == {
            "condition": "gcd(C,p)=1",
            "quotient_relation": "d=d_p+k*(p-1)",
            "quotient_bound": "k>=0",
            "identity": "C^d=C^d_p*(C^(p-1))^k",
        }
        and value["nonunit_branch"]
        == {
            "condition": "p|C",
            "d_positive": True,
            "d_p_positive": True,
            "residues": [0, 0],
        }
        and value["domain_split"] == ["gcd(C,p)=1", "p|C"]
        and _witness(value["unit_witness"], unit=True)
        and _witness(value["nonunit_witness"], unit=False)
        and value["finite_testing_role"] == "SANITY_ONLY_NOT_UNIVERSAL_PROOF"
        and _bounded_sanity(source)
    )


def main() -> None:
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if protocol_ok else {}
    math_correct = bool(protocol_ok and _result(data.get("result"), _load_input()))
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(math_correct),
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
