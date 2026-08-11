import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    aggregate_reward,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATION = "The external algebraic-independence theorem for delta and its derivatives is a trusted premise and is not verified here."


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(
            term in text
            for term in (
                "algebraic independence",
                "algebraic-independence",
                "transcendence",
            )
        )
        and any(term in text for term in ("trusted", "assum", "external", "premise"))
        and any(
            term in text
            for term in ("not verified", "not proved", "not checked", "unverified")
        )
    )


# Published reward-bearing prose obligations for evidence/answer.txt.  Each
# fact has several equivalent formulations, and the file is scanned in chunks
# so evidence size is not itself a hidden validity condition.
_EVIDENCE_FACTS = {
    "coordinate_inverse": (
        re.compile(r"\bbirational\b"),
        re.compile(
            r"(?:coordinate|change|substitution|map).{0,96}"
            r"(?:rational.{0,32})?invers"
        ),
    ),
    "conjugate_norm": (
        re.compile(r"\bconjugate\s+norm\b"),
        re.compile(r"\bconjugate\s+product\b"),
        re.compile(r"\bnorm\b"),
    ),
    "exact": (
        re.compile(r"\bexact(?:ly)?\b"),
        re.compile(r"\bcomputed?\s+without\s+approximation\b"),
    ),
    "rational_domain": (
        re.compile(r"\bqq\b"),
        re.compile(r"\brationals?\b"),
        re.compile(r"\brational\s+(?:numbers?|coefficients?)\b"),
    ),
    "independence_theorem": (
        re.compile(r"\balgebraic\s+independence\b"),
        re.compile(r"\bmodular[- ]form\s+independence\b"),
    ),
    "trusted_premise": (
        re.compile(r"\btrusted\b"),
        re.compile(r"\bpremise\b"),
        re.compile(r"\bassum(?:e|ed|ption)\b"),
        re.compile(r"\bexternal\b"),
    ),
}


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, Fraction()) + coefficient
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def scale(polynomial, scalar):
    return {
        exponent: coefficient * scalar
        for exponent, coefficient in polynomial.items()
        if coefficient * scalar
    }


def multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            exponent = tuple(x + y for x, y in zip(a, b, strict=True))
            result[exponent] = result.get(exponent, Fraction()) + ca * cb
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def monomial(exponents, coefficient=1):
    return {tuple(exponents): Fraction(coefficient)}


def parse_polynomial(value):
    if not isinstance(value, list) or not value:
        return None
    result = {}
    for term in value:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            return None
        try:
            coefficient = Fraction(term["coefficient"])
        except (ValueError, ZeroDivisionError):
            return None
        if str(coefficient) != term["coefficient"] or not coefficient:
            return None
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 3
            or any(type(x) is not int or x < 0 for x in exponents)
        ):
            return None
        key = tuple(exponents)
        if key in result:
            return None
        result[key] = coefficient
    return result


def expected_polynomials():
    p = monomial((1, 0, 0))
    q = monomial((0, 1, 0))
    s = monomial((0, 0, 1))
    even = add(multiply(p, p), scale(q, -1))
    odd_coefficient = add(p, q)
    norm = add(
        multiply(even, even),
        scale(multiply(multiply(odd_coefficient, odd_coefficient), s), -1),
        scale(multiply(odd_coefficient, multiply(s, s)), -2),
        scale(multiply(multiply(s, s), s), -1),
    )
    return {
        "p_numerator": monomial((0, 1, 0)),
        "p_denominator": monomial((1, 0, 0)),
        "q_numerator": add(
            scale(monomial((0, 2, 0)), 13), scale(monomial((1, 0, 1)), -12)
        ),
        "q_denominator": monomial((2, 0, 0)),
        "d_delta_inverse": monomial((1, 1, 0)),
        "d2_delta_numerator": add(
            scale(monomial((1, 2, 0)), 13), scale(monomial((1, 0, 1)), -1)
        ),
        "d2_delta_denominator": monomial((0, 0, 0), 12),
        "s_forward": add(monomial((0, 0, 3)), scale(monomial((1, 0, 0)), -1)),
        "delta_inverse": add(monomial((0, 3, 0)), scale(monomial((0, 0, 1)), -1)),
        "norm_polynomial": norm,
    }


def valid_result(result):
    if not isinstance(result, dict) or set(result) != set(expected_polynomials()):
        return False
    parsed = {name: parse_polynomial(value) for name, value in result.items()}
    return (
        all(value is not None for value in parsed.values())
        and parsed == expected_polynomials()
    )


def evidence_text_is_valid(path: Path) -> bool:
    """Stream evidence and require every published semantic fact."""

    matched = dict.fromkeys(_EVIDENCE_FACTS, False)
    carry = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                for name, alternatives in _EVIDENCE_FACTS.items():
                    if not matched[name] and any(
                        pattern.search(window) for pattern in alternatives
                    ):
                        matched[name] = True
                carry = window[-256:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return all(matched.values())


def evidence_ok(evidence):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    return evidence_text_is_valid(target)


def raw_submission():
    """Parse the bounded submission JSON without full-schema validation.

    Mathematical correctness, evidence, scope, and assurance are evaluated
    from this raw object so that an independent envelope fault (invalid
    evidence path, wrong scope, etc.) does not erase the mathematical
    diagnostic.  Full-schema validity is reserved for ``protocol_compliance``.
    """

    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    expected = json.loads((T / "expected.json").read_text())
    raw = raw_submission()
    input_binding = workspace_input_is_bound()
    submission = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    mathematical = valid_result(result)
    evidence = bool(
        isinstance(submission, dict) and evidence_ok(submission.get("evidence"))
    )
    scope = bool(
        isinstance(submission, dict)
        and submission.get("scope") == "exact-birational-and-quadratic-transfer-over-QQ"
        and submission.get("completeness") == "COMPLETE"
        and _limitations_valid(submission.get("limitations"))
    )
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": aggregate_reward(
                    correctness=mathematical,
                    evidence_validity=evidence,
                    scope_accuracy=scope,
                    assurance_calibration=assurance,
                    false_certification=false_certification,
                    protocol_ok=bool(contract and input_binding),
                    soft_assurance=True,
                ),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
