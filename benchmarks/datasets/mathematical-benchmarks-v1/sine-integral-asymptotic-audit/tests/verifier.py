import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The checker replays an exact formal tail identity and bound under standard calculus lemmas; it does not machine-prove those lemmas or arbitrary transcendental asymptotics."


def _load() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _term_map(value: object) -> dict[tuple[str, int], int] | None:
    if not isinstance(value, list) or len(value) != 5:
        return None
    result: dict[tuple[str, int], int] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "function",
            "power",
            "coefficient",
        }:
            return None
        function, power, coefficient = (
            item["function"],
            item["power"],
            item["coefficient"],
        )
        if (
            type(function) is not str
            or function not in {"SIN", "COS"}
            or type(power) is not int
            or (type(coefficient) is not int)
            or (power < 1)
        ):
            return None
        key = (function, power)
        if key in result:
            return None
        result[key] = coefficient
    return result


def _formal_tail_identity(terms: dict[tuple[str, int], int], remainder: object) -> bool:
    if not isinstance(remainder, dict) or set(remainder) != {
        "integrand",
        "power",
        "coefficient",
    }:
        return False
    if (
        remainder.get("integrand") != "COS"
        or type(remainder.get("power")) is not int
        or remainder.get("power") != 6
        or (type(remainder.get("coefficient")) is not int)
    ):
        return False
    derivative: defaultdict[tuple[str, int], int] = defaultdict(int)
    for (function, power), coefficient in terms.items():
        if function == "COS":
            derivative["SIN", power] -= coefficient
            derivative["COS", power + 1] -= power * coefficient
        else:
            derivative["COS", power] += coefficient
            derivative["SIN", power + 1] -= power * coefficient
    derivative["COS", 6] -= remainder["coefficient"]
    return {key: value for key, value in derivative.items() if value} == {
        ("SIN", 1): -1
    }


def _result(value: object, frozen: dict[str, Any]) -> bool:
    required = {
        "tail_terms",
        "tail_remainder",
        "si_terms",
        "si_remainder",
        "absolute_remainder_bound",
        "published_sine_coefficient",
        "corrected_sine_coefficient",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    tail = _term_map(value["tail_terms"])
    si = _term_map(value["si_terms"])
    tr, sr, bound = (
        value["tail_remainder"],
        value["si_remainder"],
        value["absolute_remainder_bound"],
    )
    published = frozen.get("published_expansion")
    if tail is None or si is None or (not _formal_tail_identity(tail, tr)):
        return False
    if set(si) != set(tail) or any(
        (si[key] != -coefficient for key, coefficient in tail.items())
    ):
        return False
    if (
        not isinstance(tr, dict)
        or not isinstance(sr, dict)
        or type(sr.get("power")) is not int
        or (type(sr.get("coefficient")) is not int)
        or (sr != {**tr, "coefficient": -tr["coefficient"]})
    ):
        return False
    if (
        not isinstance(bound, dict)
        or set(bound) != {"numerator", "power", "domain"}
        or type(bound.get("numerator")) is not int
        or (type(bound.get("power")) is not int)
    ):
        return False
    if (
        bound != {"numerator": abs(sr["coefficient"]) // 5, "power": 5, "domain": "x>0"}
        or abs(sr["coefficient"]) % 5
    ):
        return False
    published_sine = (
        next(
            (
                item.get("coefficient")
                for item in published
                if isinstance(item, dict)
                and item.get("function") == "SIN"
                and (item.get("power") == 2)
            ),
            None,
        )
        if isinstance(published, list)
        else None
    )
    corrected = si.get(("SIN", 2))
    return (
        type(value["published_sine_coefficient"]) is int
        and type(value["corrected_sine_coefficient"]) is int
        and (value["published_sine_coefficient"] == published_sine == 1)
        and (value["corrected_sine_coefficient"] == corrected == -1)
        and (published_sine != corrected)
    )


def main() -> None:
    submission = load_submission(require_input_binding=False)
    protocol_ok = submission is not None and workspace_input_is_bound()
    data = submission if isinstance(submission, dict) else {}
    frozen = _load()
    math_correct = bool(_result(data.get("result"), frozen))
    reward = float(protocol_ok and math_correct)
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
