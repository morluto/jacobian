import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

support = [-3, -1, 2, 5]
masses = [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5)]


def text(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


prelimit = [
    {"x": x, "y": y, "mass": text(masses[i] * masses[j])}
    for i, x in enumerate(support)
    for j, y in enumerate(support)
]
limit = [
    {"x": x, "y": y, "mass": text(masses[i] if i == j else Fraction(0))}
    for i, x in enumerate(support)
    for j, y in enumerate(support)
]


def pushforward(entries):
    result = defaultdict(Fraction)
    for entry in entries:
        result[entry["x"] * entry["y"]] += Fraction(
            entry["mass"]["numerator"], entry["mass"]["denominator"]
        )
    return [
        {"value": value, "mass": text(mass)}
        for value, mass in sorted(result.items())
        if mass
    ]


result = {
    "diagnosis": "MISSING_JOINT_LAW_CONTROL",
    "sequence_model": "CONSTANT_IN_N",
    "support": support,
    "prelimit_joint": prelimit,
    "limit_joint": limit,
    "prelimit_product_distribution": pushforward(prelimit),
    "limit_product_distribution": pushforward(limit),
    "witness_product_value": 1,
    "missing_assumption": "JOINT_CONVERGENCE",
}
submission = {"result": result}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
