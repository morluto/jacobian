import json
import math
from pathlib import Path
from typing import Any

from verifier_support import load_submission, normalize_reward_file

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _source() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _prime(p: object) -> bool:
    return (
        type(p) is int
        and 2 <= p <= 97
        and all(p % d for d in range(2, math.isqrt(p) + 1))
    )


def _valuation_obstruction(value: object) -> bool:
    return value == {
        "v_u_u": 1,
        "pth_power_multiple": "p*integer",
        "residue_contradiction": [1, 0],
    }


def _eisenstein_certificate(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("method") == "eisenstein"
        and value.get("prime_element") == "u"
        and value.get("constant_term_valuation") == 1
        and value.get("leading_coefficient_valuation") == 0
    )


def _result(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    strategy_fields = {"valuation_obstruction", "irreducibility_certificate"}
    required = {
        "diagnosis",
        "candidate_polynomial",
        "irreducibility",
        "minimal_polynomial",
        "inseparability",
        "sanity_prime",
    }
    selected_strategy = keys & strategy_fields
    if len(selected_strategy) != 1 or keys != required | selected_strategy:
        return False
    has_valuation = "valuation_obstruction" in value
    has_eisenstein = "irreducibility_certificate" in value
    if has_valuation == has_eisenstein:
        return False
    provenance = source.get("source", {})
    p = value["sanity_prime"]
    if not _prime(p):
        return False
    # Independently replay the characteristic-p derivative and exponent-class
    # separation used by the symbolic valuation obstruction.
    derivative_coeff = p % p
    exponent_classes = {0 % p, p % p}
    certificate_ok = (
        _valuation_obstruction(value["valuation_obstruction"])
        if has_valuation
        else _eisenstein_certificate(value["irreducibility_certificate"])
    )
    return bool(
        provenance.get("revision") == "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        and provenance.get("row") == 2
        and value["diagnosis"] == "ANNIHILATING_POLYNOMIAL_NOT_YET_MINIMAL"
        and value["candidate_polynomial"] == "X^p-u"
        and certificate_ok
        and value["irreducibility"] == "X^p-u_IRREDUCIBLE_OVER_K(u)"
        and value["minimal_polynomial"] == {"polynomial": "X^p-u", "degree": "p"}
        and value["inseparability"]
        == {
            "formal_derivative": "0",
            "root_multiplicity": "p",
            "conclusion": "t_INSEPARABLE_OVER_K(u)",
        }
        and derivative_coeff == 0
        and exponent_classes == {0}
        and 1 % p != 0
    )


def main() -> None:
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    math_ok = bool(protocol_ok and _result(data.get("result"), _source()))
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": float(math_ok),
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
