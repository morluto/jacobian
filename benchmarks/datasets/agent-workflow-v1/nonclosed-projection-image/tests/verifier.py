import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_INPUT_BYTES = 1_048_576
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "The verifier checks exact sequence identities and analytic bounds but does "
    "not formalize Hilbert-space topology in a proof assistant."
)
# Minimum number of submitted limit coordinates so the tail bound is exercised
# well past the prefix instead of only at the truncation point.
MIN_VERIFICATION_TERMS = 100
PREFIX_LENGTH = 12
# Thread PRRT_kwDOThEfjc6VxiRv: cap the tail-bound exponent before exact
# exponentiation. Fraction(m) ** exponent builds an unbounded integer; a
# schema-valid bound_exponent of 10^10 would require over 1 GB for the
# integer alone, exceeding the verifier memory limit before reward.json is
# written. This conservative cap is far above any mathematically meaningful
# decay rate for a square-summability tail bound.
MAX_BOUND_EXPONENT = 100
# Each proof obligation must carry a substantive argument (not a keyword) that
# names the link it certifies. The result-bound RESULT_JSON marker ties the
# prose to the exact submitted witness.
PROOF_OBLIGATIONS = (
    ("boundedness", ("bound",)),
    ("closedness", ("closed",)),
    ("range_identification", ("range", "image")),
    ("convergence", ("converg", "tail")),
    ("absent_preimage", ("preimage", "ell2", "summable")),
)
MIN_PROOF_ARGUMENT_CHARS = 40
_RESULT_FIELDS = {
    "space",
    "operator",
    "subspace",
    "projection",
    "operator_bound",
    "prefixes",
    "limit_coordinates",
    "tail_bound",
    "limit_preimage",
}
_PREFIX_FIELDS = {
    "n",
    "weight",
    "preimage_coordinate",
    "limit_norm_sq_partial",
    "preimage_norm_sq_partial",
}
_TAIL_BOUND_FIELDS = {"bound_coefficient", "bound_exponent", "verification_terms"}
_GROWTH_FIELDS = {"bound_coefficient", "bound_exponent"}


def _source() -> dict[str, Any]:
    try:
        frozen_path = TESTS / "input.json"
        visible_path = WORKSPACE / "input.json"
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_INPUT_BYTES
            for path in (frozen_path, visible_path)
        ):
            return {}
        raw = frozen_path.read_bytes()
        if visible_path.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, RecursionError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or len(value) > 128:
        return None
    if not value.replace("/", "", 1).lstrip("+-").isdigit():
        return None
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if str(result) == value else None


def _positive_fraction(value: object) -> Fraction | None:
    parsed = _fraction(value)
    return parsed if parsed is not None and parsed > 0 else None


def _parse_tail_bound(
    value: object,
) -> tuple[Fraction, int, int] | None:
    if not isinstance(value, dict) or set(value) != _TAIL_BOUND_FIELDS:
        return None
    coefficient = _positive_fraction(value["bound_coefficient"])
    exponent = value["bound_exponent"]
    terms = value["verification_terms"]
    if (
        coefficient is None
        or not isinstance(exponent, int)
        or isinstance(exponent, bool)
        or exponent < 1
        or exponent > MAX_BOUND_EXPONENT
        or type(terms) is not int
        or terms < MIN_VERIFICATION_TERMS
    ):
        return None
    return coefficient, exponent, terms


def _parse_growth(value: object) -> tuple[Fraction, int] | None:
    if not isinstance(value, dict) or set(value) != _GROWTH_FIELDS:
        return None
    coefficient = _positive_fraction(value["bound_coefficient"])
    exponent = value["bound_exponent"]
    if (
        coefficient is None
        or type(exponent) is not int
        or exponent < 1
        or exponent > MAX_BOUND_EXPONENT
    ):
        return None
    return coefficient, exponent


def _parse_limit_coordinates(value: object, terms: int) -> list[Fraction] | None:
    if not isinstance(value, list) or len(value) != terms:
        return None
    parsed: list[Fraction] = []
    for entry in value:
        coordinate = _fraction(entry)
        if coordinate is None:
            return None
        parsed.append(coordinate)
    return parsed


def _prefixes_ok(
    prefixes: object,
    limit_coordinates: list[Fraction],
    bound: Fraction,
    length: int,
    growth: tuple[Fraction, int],
    weighted_shift: bool,
) -> bool:
    if not isinstance(prefixes, list) or len(prefixes) != length:
        return False
    limit_partial = Fraction(0)
    preimage_partial = Fraction(0)
    for index, item in enumerate(prefixes, start=1):
        if not isinstance(item, dict) or set(item) != _PREFIX_FIELDS:
            return False
        # Thread PRRT_kwDOThEfjc6VxiRy: reject JSON booleans for prefix indices.
        # Python treats True == 1, so a bare equality check accepts boolean n
        # values that violate the agent-visible integer schema.
        if type(item["n"]) is not int or item["n"] != index:
            return False
        weight = _positive_fraction(item["weight"])
        preimage_coordinate = _fraction(item["preimage_coordinate"])
        if (
            weight is None
            or preimage_coordinate is None
            or preimage_coordinate == 0
            or weight > bound
        ):
            return False
        # A diagonal witness relates coordinates at the same index. A weighted
        # shift has y_1=0 and y_n=w_{n-1}x_{n-1} for n>=2, so the row's weight
        # is applied to the preceding forced preimage coordinate.
        if weighted_shift:
            relation_ok = (
                limit_coordinates[index - 1] == 0
                if index == 1
                else limit_coordinates[index - 1]
                == weight * _fraction(prefixes[index - 2]["preimage_coordinate"])
            )
        else:
            relation_ok = limit_coordinates[index - 1] == weight * preimage_coordinate
        if not relation_ok:
            return False
        limit_partial += limit_coordinates[index - 1] ** 2
        preimage_partial += preimage_coordinate**2
        if (
            _fraction(item["limit_norm_sq_partial"]) != limit_partial
            or _fraction(item["preimage_norm_sq_partial"]) != preimage_partial
            or preimage_partial < growth[0] * index ** growth[1]
        ):
            return False
    return True


def _tail_bound_ok(
    limit_coordinates: list[Fraction],
    coefficient: Fraction,
    exponent: int,
    terms: int,
    length: int,
) -> bool:
    # sum_{n=m+1}^{terms} y_n^2 <= C / m^d for each prefix index m. exponent >= 1
    # forces the bound to zero, so sum y_n^2 converges and the declared limit is
    # square-summable.
    suffix_sums = [Fraction(0)] * (terms + 2)
    running = Fraction(0)
    for n in range(terms, 0, -1):
        running += limit_coordinates[n - 1] ** 2
        suffix_sums[n] = running
    return all(
        suffix_sums[m + 1] <= coefficient / Fraction(m) ** exponent
        for m in range(1, length + 1)
    )


def _witness(value: object, source: dict[str, Any]) -> bool:
    """Validate a diagonal-operator graph counterexample generically.

    Accepts any bounded positive diagonal weights with a square-summable limit
    ``y`` whose forced preimage ``x`` (related by ``y_n = w_n x_n``) is not
    square-summable, plus a tail bound proving convergence of ``sum y_n^2``.
    The hidden Oracle's exact construction is not required.
    """
    length = source.get("prefix_length")
    if not isinstance(length, int) or length != PREFIX_LENGTH:
        return False
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        return False
    for key in ("space", "operator", "subspace", "projection", "limit_preimage"):
        if not isinstance(value[key], str) or not value[key].strip():
            return False
    space = value["space"].casefold()
    operator = value["operator"].casefold()
    subspace = value["subspace"].casefold()
    projection = value["projection"].casefold()
    preimage = value["limit_preimage"].casefold()
    if "ell2" not in space or "graph" not in subspace or "closed" not in subspace:
        return False
    orthogonal_projection = bool(
        "second" in projection
        and (
            re.search(r"\borthogonal\b", projection)
            or "(0,v)" in projection.replace(" ", "")
        )
        and "nonorthogonal" not in projection
    )
    if not orthogonal_projection:
        return False
    if not any(
        term in preimage for term in ("not in ell2", "not square", "not summable")
    ):
        return False
    weighted_shift = "weighted shift" in operator
    if weighted_shift:
        if not any(term in operator for term in ("1/n", "1/(n", "1 / n", "/n", "/(n")):
            return False
    elif "diagonal" not in operator or "1/n" not in operator:
        return False
    bound = _positive_fraction(value["operator_bound"])
    if bound is None:
        return False
    tail = _parse_tail_bound(value["tail_bound"])
    if tail is None:
        return False
    coefficient, exponent, terms = tail
    limit_coordinates = _parse_limit_coordinates(value["limit_coordinates"], terms)
    if limit_coordinates is None:
        return False
    growth = _parse_growth(value.get("preimage_growth")) or (Fraction(1), 1)
    return bool(
        growth
        and _prefixes_ok(
            value["prefixes"],
            limit_coordinates,
            bound,
            length,
            growth,
            weighted_shift,
        )
        and _tail_bound_ok(limit_coordinates, coefficient, exponent, terms, length)
    )


def _extract_proof(text: str) -> dict[str, Any] | None:
    proof_marker = next(
        (
            line.removeprefix("PROOF_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("PROOF_JSON:")
        ),
        None,
    )
    if proof_marker is None:
        return None
    try:
        proof = json.loads(proof_marker)
    except (ValueError, RecursionError):
        return None
    if not isinstance(proof, dict) or set(proof) != {
        name for name, _ in PROOF_OBLIGATIONS
    }:
        return None
    return proof


def _proof_ok(proof: dict[str, Any]) -> bool:
    for name, terms in PROOF_OBLIGATIONS:
        argument = proof.get(name)
        if not isinstance(argument, str) or len(argument) < MIN_PROOF_ARGUMENT_CHARS:
            return False
        if not any(term in argument.lower() for term in terms):
            return False
    return True


def _evidence(value: object, result: object) -> bool:
    """Require result-bound proof evidence with a substantive argument per link.

    The evidence file must repeat the exact submitted result via a
    ``RESULT_JSON:`` marker and carry a ``PROOF_JSON:`` block whose obligation
    fields each contain a non-trivial argument naming the link they certify.
    """
    if not isinstance(value, list) or len(value) != 1:
        return False
    if not evidence_list_is_bound(value, expected_path="evidence/answer.txt"):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = path.read_text()
    except (OSError, UnicodeError):
        return False
    result_marker = "RESULT_JSON: " + json.dumps(
        result, sort_keys=True, separators=(",", ":")
    )
    if result_marker not in text:
        return False
    proof = _extract_proof(text)
    return proof is not None and _proof_ok(proof)


def _limitation_is_valid(value: str) -> bool:
    folded = value.casefold()
    if "topolog" not in folded or "proof assistant" not in folded:
        return False
    affirmative_pattern = re.compile(
        r"\b(?:formaliz(?:e|ed|es)|verif(?:y|ied|ies)|prove(?:s|d)?)\b"
        r"[^.;\n]{0,100}\b(?:topolog|proof assistant)\b",
    )
    for clause in re.split(r"[.;\n]", folded):
        affirmative = affirmative_pattern.search(clause)
        if affirmative and not re.search(
            r"\b(?:not|no|without|does not|doesn't|cannot)\b",
            clause[: affirmative.start()],
        ):
            return False
    return bool(
        re.search(
            r"\b(?:not|no|without|does not|doesn't|cannot)\b"
            r"[^.;\n]{0,100}\b(?:formal|topolog|proof assistant)\b",
            folded,
        )
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    source = _source()
    result = data.get("result")
    correct = bool(contract and _witness(result, source))
    evidence = bool(correct and _evidence(data.get("evidence"), result))
    declared_scope = data.get("scope")
    folded_scope = declared_scope.casefold() if isinstance(declared_scope, str) else ""
    # Thread PRRT_kwDOThEfjc6VxiRu: match "closed" as a word-boundary unit so the
    # substring "closed" inside "nonclosed" does not satisfy the closed-subspace
    # requirement. A scope describing "a nonclosed Hilbert subspace" describes
    # the wrong kind of M and must not receive scope credit.
    scope = bool(
        contract
        and isinstance(declared_scope, str)
        and (
            declared_scope == expected["required_scope"]
            or (
                re.search(r"\bclosed\b", folded_scope) is not None
                and not re.search(r"\b(?:non[- ]closed|not\s+closed)\b", folded_scope)
                and all(
                    term in folded_scope
                    for term in ("hilbert", "subspace", "orthogonal", "projection")
                )
            )
        )
    )
    assurance = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(
            isinstance(item, str) and _limitation_is_valid(item)
            for item in data["limitations"]
        )
    )
    false_verified = false_verified_claim(submission, verification_record_bound=False)
    passed = bool(
        correct
        and evidence
        and scope
        and assurance
        and limitations
        and not false_verified
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": float(passed),
                "false_certification": false_verified,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
