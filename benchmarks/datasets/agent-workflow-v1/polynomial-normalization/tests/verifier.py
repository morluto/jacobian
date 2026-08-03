import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    authorized_record_is_bound,
    sha256_uri,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _digest(path):
    return sha256_uri(path)


def canonical_fraction(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    canonical = str(parsed.numerator)
    if parsed.denominator != 1:
        canonical += f"/{parsed.denominator}"
    return parsed if value == canonical else None


def verification_record_is_bound(submission):
    return authorized_record_is_bound(
        submission.get("verification_record_uri"),
        authorized_path=E / "authorized_record.json",
    )


def main():
    try:
        s = json.loads((W / "submission.json").read_text())
    except (OSError, ValueError):
        s = None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads(
        (E / "agent-workflow-v1-polynomial-normalization-expected.json").read_text()
    )
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}
    terms = r.get("terms", [])
    want = {}
    for t in x["terms"]:
        k = tuple(t["exponents"])
        want[k] = want.get(k, Fraction(0)) + Fraction(t["coefficient"])
    want = {k: v for k, v in want.items() if v}
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
    required = {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    expected_keys = required | (
        {"verification_record_uri"}
        if isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED"
        else set()
    )
    contract = (
        isinstance(s, dict)
        and set(s) == expected_keys
        and s.get("task_id") == e["task_id"]
        and s.get("conclusion") == "TRUE"
        and s.get("completeness") == "COMPLETE"
        and isinstance(s.get("scope"), str)
        and isinstance(s.get("limitations"), list)
        and isinstance(s.get("evidence"), list)
        and len(s["evidence"]) == 1
        and isinstance(s.get("claimed_assurance"), str)
        and s.get("claimed_assurance")
        in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
    )

    math_contract = (
        isinstance(s, dict)
        and required <= set(s) <= required | {"verification_record_uri"}
        and s.get("task_id") == e["task_id"]
        and s.get("conclusion") == "TRUE"
        and s.get("completeness") == "COMPLETE"
        and isinstance(s.get("scope"), str)
        and isinstance(s.get("limitations"), list)
        and isinstance(s.get("claimed_assurance"), str)
        and s.get("claimed_assurance")
        in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
    )

    record_bound = verification_record_is_bound(s) if isinstance(s, dict) else False
    math_correct = bool(
        workspace_input_is_bound()
        and math_contract
        and got == want
        and all(len(k) == 2 for k in got)
    )
    correct = bool(
        contract
        and math_correct
        and (s["claimed_assurance"] != "VERIFIED" or record_bound)
    )
    good = False
    if contract and isinstance(s["evidence"], list) and s["evidence"]:
        good = True
        for i in s["evidence"]:
            if (
                not isinstance(i, dict)
                or not isinstance(i.get("path"), str)
                or not isinstance(i.get("sha256"), str)
            ):
                good = False
                continue
            p = Path(i["path"])
            t = (W / p).resolve()
            good &= (
                isinstance(i, dict)
                and not p.is_absolute()
                and p == Path("evidence/answer.txt")
                and ".." not in p.parts
                and not (W / p).is_symlink()
                and t.is_relative_to(W.resolve())
                and t.is_file()
            )
            if good:
                good &= i.get("sha256") == _digest(t)
    scope = bool(contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(
        contract
        and (
            (
                s["claimed_assurance"] == e["maximum_assurance"]
                and s["claimed_assurance"] != "VERIFIED"
            )
            or (s["claimed_assurance"] == "VERIFIED" and record_bound)
        )
    )
    false = bool(
        isinstance(s, dict)
        and s.get("claimed_assurance") == "VERIFIED"
        and not record_bound
    )
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
