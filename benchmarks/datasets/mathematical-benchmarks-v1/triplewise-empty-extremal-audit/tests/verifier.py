import itertools
import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)


def valid_family(n, raw_family):
    try:
        if type(n) is not int or not isinstance(raw_family, list):
            return False
        if any(
            not isinstance(item, list) or any(type(value) is not int for value in item)
            for item in raw_family
        ):
            return False
        family = [tuple(item) for item in raw_family]
        family_sets = [frozenset(item) for item in family]
        if len(family_sets) != len(set(family_sets)):
            return False
        if any(len(item) != len(set(item)) for item in family):
            return False
        if any(value < 0 or value >= n for item in family for value in item):
            return False
        if len(family) != 1 + n + n // 2:
            return False
        sets = [set(item) for item in family]
        if any(
            first & second & third
            for first, second, third in itertools.combinations(sets, 3)
        ):
            return False
        return all(sum(value in item for item in sets) <= 2 for value in range(n))
    except (TypeError, ValueError):
        return False


def _result_shape_is_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "maximum_formula",
        "upper_bound_certificate",
        "constructions",
    }:
        return False
    certificate = result.get("upper_bound_certificate")
    if not isinstance(certificate, dict) or set(certificate) != {
        "element_frequency_cap",
        "singleton_cap",
        "nonsingleton_incidence_floor",
    }:
        return False
    constructions = result.get("constructions")
    if not isinstance(constructions, list) or len(constructions) != 3:
        return False
    return all(
        isinstance(item, dict)
        and set(item) == {"n", "family"}
        and type(item.get("n")) is int
        and isinstance(item.get("family"), list)
        for item in constructions
    )


def _math_claim_is_correct(result):
    if not _result_shape_is_valid(result):
        return False
    certificate = result["upper_bound_certificate"]
    constructions = result["constructions"]
    construction_map = {item["n"]: item["family"] for item in constructions}
    formula = result["maximum_formula"]
    if (
        not isinstance(formula, dict)
        or set(formula) != {"constant", "linear", "floor_half"}
        or any(type(formula[key]) is not int for key in formula)
    ):
        return False
    cap = certificate.get("singleton_cap")
    return bool(
        all(
            formula["constant"]
            + formula["linear"] * n
            + formula["floor_half"] * (n // 2)
            == 1 + n + n // 2
            for n in (7, 8, 11)
        )
        and certificate.get("element_frequency_cap") == 2
        and certificate.get("nonsingleton_incidence_floor") == 2
        and isinstance(cap, dict)
        and cap == {"variable": "n"}
        and set(construction_map) == {7, 8, 11}
        and len(construction_map) == len(constructions)
        and all(valid_family(n, construction_map[n]) for n in construction_map)
        and all(
            len(construction_map[n])
            == formula["constant"]
            + formula["linear"] * n
            + formula["floor_half"] * (n // 2)
            for n in construction_map
        )
    )


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if protocol_ok else {}
    result = data.get("result", {})
    math_correct = bool(protocol_ok and _math_claim_is_correct(result))
    reward = aggregate_reward(
        correctness=math_correct,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
