import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
SUPPORT = (-3, -1, 2, 5)
MARGINAL = (Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5))


ATTAINABLE_PRODUCTS = frozenset(x * y for x in SUPPORT for y in SUPPORT)


def canonical_fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        parsed = Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None
    if parsed < 0 or parsed > 1:
        return None
    return parsed


def parse_joint(entries):
    expected_pairs = {(x, y) for x in SUPPORT for y in SUPPORT}
    if not isinstance(entries, list) or len(entries) != len(expected_pairs):
        return None
    table = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"x", "y", "mass"}:
            return None
        pair = (entry.get("x"), entry.get("y"))
        if pair not in expected_pairs or pair in table:
            return None
        mass = canonical_fraction(entry.get("mass"))
        if mass is None:
            return None
        table[pair] = mass
    return table if set(table) == expected_pairs and sum(table.values()) == 1 else None


def marginals(table):
    left = {x: sum(table[x, y] for y in SUPPORT) for x in SUPPORT}
    right = {y: sum(table[x, y] for x in SUPPORT) for y in SUPPORT}
    return left, right


def target_marginal():
    return dict(zip(SUPPORT, MARGINAL, strict=True))


def product_pushforward(table):
    distribution = defaultdict(Fraction)
    for (x, y), mass in table.items():
        distribution[x * y] += mass
    return {value: mass for value, mass in sorted(distribution.items()) if mass}


def parse_product(entries):
    if not isinstance(entries, list) or not entries:
        return None
    distribution = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"value", "mass"}:
            return None
        value = entry.get("value")
        mass = canonical_fraction(entry.get("mass"))
        if type(value) is not int or mass is None or value in distribution:
            return None
        # Zero-mass entries are allowed only for attainable product values
        # (values in {x * y for x, y in SUPPORT}); unattainable zero entries
        # are rejected as malformed.
        if mass == 0 and value not in ATTAINABLE_PRODUCTS:
            return None
        distribution[value] = mass
    if sum(distribution.values()) != 1:
        return None
    return {value: mass for value, mass in distribution.items() if mass}


def result_is_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "diagnosis",
        "sequence_model",
        "support",
        "prelimit_joint",
        "limit_joint",
        "prelimit_product_distribution",
        "limit_product_distribution",
        "witness_product_value",
        "missing_assumption",
    }:
        return False
    if (
        result.get("diagnosis") != "MISSING_JOINT_LAW_CONTROL"
        or result.get("sequence_model") != "CONSTANT_IN_N"
    ):
        return False
    if result.get("support") != list(SUPPORT):
        return False
    if result.get("missing_assumption") not in {
        "JOINT_CONVERGENCE",
        "LIMIT_PAIR_INDEPENDENCE",
    }:
        return False
    prelimit = parse_joint(result.get("prelimit_joint"))
    limit = parse_joint(result.get("limit_joint"))
    if prelimit is None or limit is None:
        return False
    target = target_marginal()
    if marginals(prelimit) != (target, target) or marginals(limit) != (target, target):
        return False
    independent = {(x, y): target[x] * target[y] for x in SUPPORT for y in SUPPORT}
    if prelimit != independent or limit == independent:
        return False
    prelimit_product = product_pushforward(prelimit)
    limit_product = product_pushforward(limit)
    if parse_product(result.get("prelimit_product_distribution")) != prelimit_product:
        return False
    if parse_product(result.get("limit_product_distribution")) != limit_product:
        return False
    witness = result.get("witness_product_value")
    return type(witness) is int and prelimit_product.get(
        witness, 0
    ) != limit_product.get(witness, 0)


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    result = data.get("result")
    math_ok = bool(
        isinstance(submission, dict) and input_bound and result_is_valid(result)
    )
    correct = math_ok
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
