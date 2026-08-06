import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

support = [-3, -1, 2, 5]
masses = [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5)]


def text(value):
    return str(value)


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
        result[entry["x"] * entry["y"]] += Fraction(entry["mass"])
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
evidence = (
    "The marginal laws are constant in n, and every prelimit joint law is the "
    "independent product coupling. The named limit variables have the same "
    "marginals but a diagonal joint coupling, so their product law differs. "
    "Marginal convergence alone therefore does not supply joint convergence "
    "or independence of the limit pair.\n"
    f"RESULT_JSON:{json.dumps(result, sort_keys=True, separators=(',', ':'))}\n"
)
Path("/app/evidence").mkdir(parents=True, exist_ok=True)
Path("/app/evidence/answer.txt").write_text(evidence)
digest = hashlib.sha256(evidence.encode()).hexdigest()
submission = {
    "task_id": "jacobian/marginal-joint-product-audit",
    "conclusion": "MARGINAL_CONVERGENCE_INSUFFICIENT",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "frozen-four-point-marginal-and-submitted-couplings",
    "completeness": "COMPLETE",
    "evidence": [{"path": "evidence/answer.txt", "sha256": f"sha256:{digest}"}],
    "limitations": [
        "This exact finite-law countermodel does not machine-verify a general "
        "weak-convergence theorem or disambiguate the original prose."
    ],
}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
