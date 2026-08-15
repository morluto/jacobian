import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


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
    return (
        parsed
        if parsed.numerator == numerator and parsed.denominator == denominator
        else None
    )


def _compute_want(x):
    want = {}
    for t in x["terms"]:
        k = tuple(t["exponents"])
        want[k] = want.get(k, Fraction(0)) + Fraction(t["coefficient"])
    return {k: v for k, v in want.items() if v}


def _parse_terms(terms):
    got = {}
    try:
        if not isinstance(terms, list):
            raise TypeError
        for t in terms:
            if not isinstance(t, dict) or not isinstance(t.get("exponents"), list):
                raise TypeError
            exponents = t["exponents"]
            if len(exponents) != 2 or any(type(v) is not int for v in exponents):
                raise ValueError
            key = tuple(exponents)
            if key in got:
                raise ValueError
            coefficient = canonical_fraction(t["coefficient"])
            if coefficient is None:
                raise ValueError
            got[key] = coefficient
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        got = {}
    return got


def _witness_is_bound(witness: object) -> bool:
    if not isinstance(witness, list) or not witness:
        return False
    return resolve_evidence(witness[0], expected_path="evidence/answer.txt") is not None


def main():
    submission = load_submission(W / "submission.json")
    x = json.loads(next(E.glob("*input*.json")).read_text())
    r = submission.get("result") if isinstance(submission, dict) else None
    r = r if isinstance(r, dict) else {}
    terms = r.get("terms", [])
    want = _compute_want(x)
    got = _parse_terms(terms)
    input_bound = workspace_input_is_bound()
    math_correct = bool(
        isinstance(submission, dict)
        and input_bound
        and got == want
        and all(len(k) == 2 for k in got)
    )
    witness_ok = bool(
        math_correct
        and isinstance(submission, dict)
        and _witness_is_bound(submission.get("witness"))
    )
    correct = bool(math_correct and witness_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(witness_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
