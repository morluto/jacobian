import json
import re
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
LIMITATION_OBLIGATIONS = (
    "claim:finite-action-counterexample",
    "limitation:no-general-classification-theorem",
)


def _limitations_valid(value: object) -> bool:
    return isinstance(value, list) and value == list(LIMITATION_OBLIGATIONS)


def matrix(value, q):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError
    if any(not isinstance(row, list) or len(row) != 3 for row in value):
        raise ValueError
    if any(type(x) is not int or not 0 <= x < q for row in value for x in row):
        raise ValueError
    return tuple(tuple(row) for row in value)


def vector(value, q):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(x) is not int or not 0 <= x < q for x in value)
    ):
        raise ValueError
    return tuple(value)


def multiply(a, b, q):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) % q for j in range(3))
        for i in range(3)
    )


def apply(a, v, q):
    return tuple(sum(a[i][j] * v[j] for j in range(3)) % q for i in range(3))


def determinant(a, q):
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    ) % q


def closure(generators, q, maximum):
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    seen, frontier = {identity}, [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = multiply(current, generator, q)
            if candidate not in seen:
                seen.add(candidate)
                if len(seen) > maximum:
                    raise ValueError
                frontier.append(candidate)
    return sorted(seen)


def rank(rows, q):
    work = [list(row) for row in rows if any(x % q for x in row)]
    pivot_row = 0
    for column in range(3):
        pivot = next(
            (i for i in range(pivot_row, len(work)) if work[i][column] % q), None
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column] % q, -1, q)
        work[pivot_row] = [(x * inverse) % q for x in work[pivot_row]]
        for i, row in enumerate(work):
            if i != pivot_row and row[column] % q:
                factor = row[column] % q
                work[i] = [
                    (x - factor * y) % q
                    for x, y in zip(row, work[pivot_row], strict=True)
                ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _elements_and_vectors_ok(elements_raw, vectors_raw, generated, q, contract):
    if not isinstance(elements_raw, list) or not isinstance(vectors_raw, list):
        return False
    elements = [matrix(item, q) for item in elements_raw]
    vectors = [vector(item, q) for item in vectors_raw]
    if (
        not contract["minimum_group_order"]
        <= len(generated)
        <= contract["maximum_group_order"]
    ):
        return False
    if elements != generated or len(vectors) != len(elements):
        return False
    if any(determinant(item, q) != 1 for item in elements):
        return False
    return not any(
        v == (0, 0, 0) or apply(a, v, q) != v
        for a, v in zip(elements, vectors, strict=True)
    )


def certificate_valid(result, frozen):
    try:
        if not isinstance(result, dict) or set(result) != {
            "field_prime",
            "generators",
            "group_elements",
            "fixed_vectors",
            "common_fixed_dimension",
        }:
            return False
        q = result["field_prime"]
        contract = frozen["contract"]
        if type(q) is not int or q not in contract["allowed_primes"]:
            return False
        generators_raw = result["generators"]
        if not isinstance(generators_raw, list) or len(generators_raw) != 2:
            return False
        generators = [matrix(item, q) for item in generators_raw]
        if any(determinant(item, q) != 1 for item in generators):
            return False
        generated = closure(generators, q, contract["maximum_group_order"])
        if not _elements_and_vectors_ok(
            result["group_elements"], result["fixed_vectors"], generated, q, contract
        ):
            return False
        fixed_equations = [
            tuple((a[i][j] - int(i == j)) % q for j in range(3))
            for a in generators
            for i in range(3)
        ]
        common_dimension = 3 - rank(fixed_equations, q)
        return (
            type(result["common_fixed_dimension"]) is int
            and result["common_fixed_dimension"] == common_dimension == 0
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
_EXPLANATION_FACTS = {
    "elementwise_nonzero": (
        re.compile(
            r"\b(?:each|every)\s+(?:group\s+)?element\b.{0,128}"
            r"\b(?:fix|preserv|stabil|invarian).{0,64}"
            r"\b(?:non[- ]?zero|nontrivial)\b",
            re.DOTALL,
        ),
        re.compile(
            r"\b(?:non[- ]?zero|nontrivial)\b.{0,64}"
            r"\b(?:fixed|invariant)\b.{0,96}\b(?:each|every)\b",
            re.DOTALL,
        ),
        re.compile(
            r"\b(?:each|every)\s+(?:group\s+)?element\b.{0,128}"
            r"\b(?:non[- ]?zero|nontrivial)\b.{0,32}\b(?:fixed|invariant)\b",
            re.DOTALL,
        ),
    ),
    "no_common_nonzero": (
        re.compile(
            r"\b(?:no|without)\b.{0,64}\b(?:common|global|shared)\b"
            r".{0,96}\b(?:fixed|invariant|vector|subspace)\b",
            re.DOTALL,
        ),
        re.compile(
            r"\b(?:common|global|shared)\b.{0,96}"
            r"\b(?:zero|trivial|absent|empty|does\s+not\s+exist)\b",
            re.DOTALL,
        ),
    ),
    "quantifier_separation": (
        re.compile(r"\bquantifier\b.{0,96}\b(?:order|swap|separat)"),
        re.compile(r"\b(?:does\s+not|doesn't|fails?\s+to|cannot)\s+imply\b"),
        re.compile(r"\b(?:counterexample|refut)"),
        re.compile(r"\bseparat"),
    ),
}


def _quantifier_explanation_valid(path: Path) -> bool:
    """Stream evidence and require the three documented mathematical facts."""

    matched = dict.fromkeys(_EXPLANATION_FACTS, False)
    carry = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                for name, alternatives in _EXPLANATION_FACTS.items():
                    if not matched[name] and any(
                        pattern.search(window) for pattern in alternatives
                    ):
                        matched[name] = True
                carry = window[-384:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return all(matched.values())


def evidence_valid(descriptors):
    # The finite group/action certificate is replayed from the typed result.
    # The public evidence contract promises one bound text artifact that
    # must contain a meaningful quantifier-failure explanation.
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        return False
    target = resolve_evidence(
        descriptors[0], expected_path="evidence/answer.txt", max_bytes=None
    )
    if target is None:
        return False
    return _quantifier_explanation_valid(target)


def main():
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    frozen = json.loads((E / "input.json").read_text())
    source = frozen.get("source", {})
    source_bound = bool(
        source.get("revision") == "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        and source.get("row") == 10001
        and source.get("row_sha256")
        == "sha256:8be03d5c7464658a27833f65a0d8b8198ea5d2d0658d088f667bb30754dd36da"
        and source.get("license") == "MIT"
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = bool(source_bound and certificate_valid(result, frozen))
    witness = bool(
        isinstance(submission, dict) and evidence_valid(submission.get("witness"))
    )
    correct = bool(input_binding and mathematical and witness)
    reward = 1.0 if correct else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "input_binding": float(input_binding),
                "correctness": float(mathematical),
                "witness_validity": float(witness),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
