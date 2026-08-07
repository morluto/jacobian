import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
SUPPORT = (-3, -1, 2, 5)
MARGINAL = (Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5))
LIMITATION = (
    "This exact finite-law countermodel does not machine-verify a general "
    "weak-convergence theorem or disambiguate the original prose."
)


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(
            term in text
            for term in ("finite-law", "finite law", "countermodel", "four-point")
        )
        and any(
            term in text
            for term in ("weak convergence", "original prose", "general theorem")
        )
        and any(
            term in text
            for term in (
                "not prove",
                "does not prove",
                "not verify",
                "not machine",
                "does not disambiguate",
            )
        )
    )


ATTAINABLE_PRODUCTS = frozenset(x * y for x in SUPPORT for y in SUPPORT)

_EVIDENCE_FACTS = {
    "marginal_convergence": (
        re.compile(r"\bmarginal\s+convergence\b"),
        re.compile(r"\bmarginal\s+limits?\b"),
        re.compile(r"\bconvergent\s+marginals?\b"),
    ),
    "joint_law": (
        re.compile(r"\bjoint\s+(?:convergence|distribution|law|coupling)\b"),
        re.compile(r"\bcouplings?\b"),
    ),
    "product_law": (
        re.compile(r"\bproduct\s+(?:law|distribution|pushforward)\b"),
        re.compile(r"\blaw\s+of\s+the\s+product\b"),
    ),
    "insufficiency": (
        re.compile(r"\bdoes\s+not\s+(?:determine|imply|fix|supply)\b"),
        re.compile(r"\binsufficient\b.{0,48}\b(?:determine|pin\s+down|imply)\b"),
        re.compile(r"\bmarginals?\s+alone\b.{0,48}\b(?:cannot|do\s+not|fail)\b"),
        re.compile(r"\bwithout\s+joint\s+convergence\b"),
    ),
}
_EVIDENCE_CONTRADICTIONS = (
    re.compile(
        r"\bmarginal\s+convergence\b.{0,48}"
        r"(?<!not )(?<!n't )\bdetermines?\b.{0,32}\bjoint\b"
    ),
    re.compile(
        r"\bproduct\s+(?:law|distribution)\b.{0,48}\bfollows?\b.{0,32}\bmarginals?\b"
    ),
    re.compile(
        r"\bjoint\s+(?:law|distribution)\b.{0,48}\buniquely\s+(?:fixed|determined)\b"
    ),
)


def _evidence_scan(prose, carry, contradicted, matched):
    if not prose:
        return carry, contradicted
    window = (carry + "".join(prose)).lower()
    prose.clear()
    contradicted = contradicted or any(
        pattern.search(window) for pattern in _EVIDENCE_CONTRADICTIONS
    )
    for name, alternatives in _EVIDENCE_FACTS.items():
        if not matched[name] and any(
            pattern.search(window) for pattern in alternatives
        ):
            matched[name] = True
    return window[-256:], contradicted


def _evidence_process_char(character, state):
    if character == "\n":
        if not state["skip_line"]:
            if state["at_line_start"]:
                state["prose"].extend(state["line_prefix"])
            state["prose"].append(character)
        state["line_prefix"] = ""
        state["at_line_start"] = True
        state["skip_line"] = False
    elif state["skip_line"]:
        pass
    elif state["at_line_start"]:
        state["line_prefix"] += character
        marker = "RESULT_JSON:"
        if marker.startswith(state["line_prefix"]):
            if state["line_prefix"] == marker:
                state["skip_line"] = True
                state["at_line_start"] = False
                state["line_prefix"] = ""
        else:
            state["prose"].extend(state["line_prefix"])
            state["line_prefix"] = ""
            state["at_line_start"] = False
    else:
        state["prose"].append(character)


def _evidence_explains_clauses(path: Path) -> bool:
    """Stream prose while ignoring private RESULT_JSON marker lines."""

    matched = dict.fromkeys(_EVIDENCE_FACTS, False)
    contradicted = False
    carry = ""
    state = {
        "prose": [],
        "line_prefix": "",
        "at_line_start": True,
        "skip_line": False,
    }

    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                for character in chunk:
                    _evidence_process_char(character, state)
                    if len(state["prose"]) >= 65_536:
                        carry, contradicted = _evidence_scan(
                            state["prose"], carry, contradicted, matched
                        )
            if not state["skip_line"] and state["at_line_start"]:
                state["prose"].extend(state["line_prefix"])
            carry, contradicted = _evidence_scan(
                state["prose"], carry, contradicted, matched
            )
    except (OSError, UnicodeError, MemoryError):
        return False
    return not contradicted and all(matched.values())


def canonical_fraction(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(?:0|1|[1-9][0-9]*/[1-9][0-9]*)", value) is None
    ):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if parsed < 0 or parsed > 1 or str(parsed) != value:
        return None
    return parsed


def parse_joint(entries):
    expected_pairs = [(x, y) for x in SUPPORT for y in SUPPORT]
    if not isinstance(entries, list) or len(entries) != len(expected_pairs):
        return None
    table = {}
    for entry, pair in zip(entries, expected_pairs, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"x", "y", "mass"}:
            return None
        if (entry.get("x"), entry.get("y")) != pair:
            return None
        mass = canonical_fraction(entry.get("mass"))
        if mass is None:
            return None
        table[pair] = mass
    return table if sum(table.values()) == 1 else None


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
    prior = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"value", "mass"}:
            return None
        value = entry.get("value")
        mass = canonical_fraction(entry.get("mass"))
        if type(value) is not int or mass is None:
            return None
        if prior is not None and value <= prior:
            return None
        # Zero-mass entries are allowed only for attainable product values
        # (values in {x * y for x, y in SUPPORT}); unattainable zero entries
        # are rejected as malformed.
        if mass == 0 and value not in ATTAINABLE_PRODUCTS:
            return None
        distribution[value] = mass
        prior = value
    if sum(distribution.values()) != 1:
        return None
    return {value: mass for value, mass in distribution.items() if mass}


def evidence_matches(evidence):
    # The full finite laws and product pushforwards are independently replayed.
    # The public evidence contract promises one bound text artifact that
    # explains why marginal convergence does not determine joint convergence
    # or the product law.
    if not evidence_list_is_bound(evidence):
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    return _evidence_explains_clauses(path)


def raw_submission():
    """Parse the bounded submission without schema validation for math checks."""

    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


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
    expected = json.loads((T / "expected.json").read_text())
    input_bound = workspace_input_is_bound()
    raw = raw_submission()
    submission = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    # Use the raw (non-schema-validated) submission for math and evidence
    # checks so that a scope or limitation failure does not erase the
    # mathematical correctness diagnostic.
    result = raw.get("result") if isinstance(raw, dict) else None
    math_ok = result_is_valid(result)
    evidence_ok = bool(isinstance(raw, dict) and evidence_matches(raw.get("evidence")))
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == "frozen-four-point-marginal-and-submitted-couplings"
        and raw.get("completeness") == "COMPLETE"
        and _limitations_valid(raw.get("limitations"))
    )
    assurance_ok = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(raw, verification_record_bound=False)
    correct = bool(
        contract
        and input_bound
        and math_ok
        and evidence_ok
        and scope_ok
        and not false_cert
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_bound),
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
