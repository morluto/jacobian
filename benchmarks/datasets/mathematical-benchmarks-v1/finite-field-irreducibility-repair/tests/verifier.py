from __future__ import annotations

import json
import re
from pathlib import Path

from verifier_support import (
    ASSURANCE_LEVELS,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/finite-field-irreducibility-repair"
CONCLUSION = "BAD_REDUCTION_DIAGNOSED_AND_IRREDUCIBILITY_REPAIRED"
SCOPE = "the irreducibility step for x^4-4x+1 only"
LIMITATIONS = [
    "The verifier does not assess the source proof's later Galois-group or density claims and does not invoke a proof assistant."
]
MAX_ANSWER_BYTES = 1_048_576


def _exact_int(value: object) -> bool:
    """Accept only a real int, rejecting bool which is an int subclass."""

    return type(value) is int


def _int_list(value: object, *, min_len: int, max_len: int) -> bool:
    """Accept a list of real ints with bounded cardinality, rejecting bools."""

    return (
        isinstance(value, list)
        and min_len <= len(value) <= max_len
        and all(type(x) is int for x in value)
    )


def _json_equal(a: object, b: object) -> bool:
    """Structural JSON equality that rejects bool where int is expected.

    Python's ``==`` treats ``True == 1`` and ``False == 0`` as equal, so a
    marker that substitutes a boolean for an integer field would match the
    validated result. This helper recursively requires exact types for
    scalars (``bool`` is never equal to ``int``) and element-wise equality
    for lists and dicts.
    """

    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return type(a) is type(b) and a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is None and b is None
    return False


def _trim(a: list[int], p: int) -> list[int]:
    a = [x % p for x in a]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _divmod(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]:
    a = _trim(a, p)
    b = _trim(b, p)
    if b == [0]:
        raise ZeroDivisionError
    q = [0] * max(1, len(a) - len(b) + 1)
    inv = pow(b[-1], -1, p)
    while a != [0] and len(a) >= len(b):
        shift = len(a) - len(b)
        coefficient = a[-1] * inv % p
        q[shift] = coefficient
        for i, value in enumerate(b):
            a[i + shift] = (a[i + shift] - coefficient * value) % p
        a = _trim(a, p)
    return _trim(q, p), a


def _gcd(a: list[int], b: list[int], p: int) -> list[int]:
    while _trim(b, p) != [0]:
        _, remainder = _divmod(a, b, p)
        a, b = b, remainder
    a = _trim(a, p)
    inv = pow(a[-1], -1, p)
    return _trim([x * inv for x in a], p)


def _mul_mod(a: list[int], b: list[int], modulus: list[int], p: int) -> list[int]:
    product = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] = (product[i + j] + x * y) % p
    return _divmod(product, modulus, p)[1]


def _pow_x(exponent: int, modulus: list[int], p: int) -> list[int]:
    result, base = [1], [0, 1]
    while exponent:
        if exponent & 1:
            result = _mul_mod(result, base, modulus, p)
        base = _mul_mod(base, base, modulus, p)
        exponent //= 2
    return _trim(result, p)


def _prime(p: int) -> bool:
    return p >= 2 and all(p % d for d in range(2, int(p**0.5) + 1))


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return (WORKSPACE / "input.json").read_bytes() == hidden and data["source"][
            "revision"
        ] == "71595210590450202b7b69225bc07e9e01b13c5c"
    except (OSError, ValueError, KeyError):
        return False


def _result(value: object) -> bool:
    required = {
        "bad_prime",
        "bad_reduction",
        "bad_factor",
        "bad_factor_power",
        "repair_prime",
        "p2_remainder",
        "p4_remainder",
        "rabin_gcd_degree",
        "rational_conclusion",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    if (
        not _exact_int(value["bad_prime"])
        or value["bad_prime"] != 2
        or not _int_list(value["bad_reduction"], min_len=5, max_len=5)
        or value["bad_reduction"] != [1, 0, 0, 0, 1]
        or not _int_list(value["bad_factor"], min_len=2, max_len=2)
        or value["bad_factor"] != [1, 1]
        or not _exact_int(value["bad_factor_power"])
        or value["bad_factor_power"] != 4
    ):
        return False
    p = value["repair_prime"]
    if not _exact_int(p) or not 3 <= p <= 97 or not _prime(p):
        return False
    modulus = [1 % p, (-4) % p, 0, 0, 1]
    p2 = _pow_x(p * p, modulus, p)
    p4 = _pow_x(p**4, modulus, p)
    submitted_p2 = value["p2_remainder"]
    submitted_p4 = value["p4_remainder"]
    if not _int_list(submitted_p2, min_len=1, max_len=4):
        return False
    if not _int_list(submitted_p4, min_len=1, max_len=4):
        return False
    x = [0, 1]
    difference = _trim(
        [
            (p2[i] if i < len(p2) else 0) - (x[i] if i < len(x) else 0)
            for i in range(max(len(p2), 2))
        ],
        p,
    )
    gcd = _gcd(modulus, difference, p)
    return bool(
        _trim(submitted_p2, p) == p2
        and _trim(submitted_p4, p) == p4
        and p4 == x
        and gcd == [1]
        and _exact_int(value["rabin_gcd_degree"])
        and value["rabin_gcd_degree"] == 0
        and value["rational_conclusion"] == "IRREDUCIBLE_OVER_Q"
    )


_FORMAL_CLAIM_PHRASES = (
    "proof assistant",
    "proof-assistant",
    "formally verified",
    "formal verification",
    "formally checked",
    "formally proved",
    "formally proven",
    "machine checked",
    "machine-checked",
    "machine verified",
    "machine-verified",
    "theorem prover",
    "theorem proving",
)
_PROVER_NAMES = (
    "lean",
    "coq",
    "isabelle",
    "agda",
    "metamath",
    "pvs",
    "acl2",
    "hol4",
    "mizar",
)
_VERIFY_VERBS = (
    "verified",
    "proved",
    "proven",
    "checked",
    "confirmed",
    "certified",
    "formal",
)


def _claims_proof_assistant(folded: str) -> bool:
    """Reject assertions of proof-assistant or formal verification; allow disclaimers.

    The instruction forbids claiming proof-assistant verification, not
    truthfully disclaiming it. A clause that mentions a proof assistant or
    formal verification alongside a negation term (e.g. "was not checked by a
    proof assistant") is a valid disclaimer, not a false claim. Affirmative
    claims using synonymous phrasing such as "Lean 4 formally verified this
    proof" are also rejected.
    """

    clauses = [clause.strip() for clause in re.split(r"[.!?]\s*", folded)]
    negations = ("not", "no", "without", "never", "neither", "nor")
    return any(
        (
            any(phrase in clause for phrase in _FORMAL_CLAIM_PHRASES)
            or any(
                name in clause and any(verb in clause for verb in _VERIFY_VERBS)
                for name in _PROVER_NAMES
            )
        )
        and not any(neg in clause for neg in negations)
        for clause in clauses
    )


def _explanation_valid(prose: str) -> bool:
    """Check the two documented mathematical obligations by clause."""

    folded = prose.casefold()
    clauses = [clause.strip() for clause in re.split(r"[.!?]\s*", folded)]
    explains_bad_implication = any(
        "reducible" in clause
        and any(
            phrase in clause
            for phrase in (
                "cannot support",
                "does not support",
                "invalid",
                "fails",
                "false implication",
                "not irreducible",
            )
        )
        for clause in clauses
    )
    explains_repair = any(
        "irreducible" in clause
        and any(phrase in clause for phrase in ("rational", "over q"))
        and any(
            phrase in clause
            for phrase in ("hence", "therefore", "proves", "shows", "criterion", "test")
        )
        for clause in clauses
    )
    explains_rabin_test = any(
        any(word in clause for word in ("rabin", "frobenius")) and "gcd" in clause
        for clause in clauses
    )
    return bool(
        len(prose) >= 140
        and explains_bad_implication
        and explains_repair
        and explains_rabin_test
        and not _claims_proof_assistant(folded)
    )


def _evidence(value: object, result: object) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    path = resolve_evidence(
        value[0], expected_path="evidence/answer.txt", max_bytes=MAX_ANSWER_BYTES
    )
    if path is None:
        return False
    try:
        if path.stat().st_size > MAX_ANSWER_BYTES:
            return False
        text = path.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line[12:].strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound = json.loads(markers[0])
    except (ValueError, RecursionError, MemoryError):
        return False
    prose = "\n".join(
        line for line in text.splitlines() if not line.startswith("RESULT_JSON:")
    )
    return bool(_json_equal(bound, result) and _explanation_valid(prose))


def _limitation_valid(limits: object) -> bool:
    """Require a limitation that excludes the density calculation from scope.

    The instructions state the verifier must not assess the later density
    calculation. A limitation such as ``The density is not purple.'' must be
    rejected because it does not state that the density step is outside the
    checked scope.
    """

    if not isinstance(limits, list):
        return False
    return limits == LIMITATIONS


def _evaluate(submission: object) -> dict[str, float | bool]:
    protocol = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    envelope_valid = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=ASSURANCE_LEVELS,
        verification_record="forbidden",
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    data = submission if isinstance(submission, dict) else {}
    limit_ok = _limitation_valid(data.get("limitations"))
    math_correct = bool(_source_is_bound() and _result(data.get("result")))
    evidence_valid = bool(_evidence(data.get("evidence"), data.get("result")))
    scope_correct = bool(envelope_valid and data.get("scope") == SCOPE and limit_ok)
    assurance_correct = bool(data.get("claimed_assurance") == "COMPUTED" and limit_ok)
    reward = (
        1.0
        if protocol
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
        else 0.0
    )
    return {
        "correctness": float(math_correct),
        "evidence_validity": float(evidence_valid),
        "scope_accuracy": float(scope_correct),
        "assurance_calibration": float(assurance_correct),
        "reward": reward,
        "false_certification": false_certification,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
