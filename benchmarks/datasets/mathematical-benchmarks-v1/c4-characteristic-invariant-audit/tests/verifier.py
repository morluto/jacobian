import itertools
import json
import re
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
ROLES = {"C4_FREE_ZERO_COUNT", "MULTIPLE_INDUCED_C4", "CHORDED_C4_ZERO_INDUCED"}
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
_EVIDENCE_READ_BYTES = 64 * 1024
_MAX_EVIDENCE_LINE_BYTES = 1024 * 1024
_PROHIBITED_CLAIM = re.compile(
    r"\b(?:verified|proved|proven|confirmed|compile|compiles|compiled|"
    r"(?:has|have|had|admits|admit|possesses|possess)\s+(?:a\s+)?proof)\b"
)
_THEOREM_SUBJECT = re.compile(
    r"\s*(?:(?:no|the|this|that|its)\s+)?"
    r"(?:(?:upstream|source-corrected|corrected)\s+)*"
    r"(?:lean(?:\s+(?:theorem|compilation))?|theorem|conjecture|proof)\b"
)
_ELIDED_SUBJECT_PREFIX = re.compile(
    r"\s*(?:(?:also|actually|still|then|yet)\s+)?"
    r"(?:(?:is|are|was|were|has|have|had|do|does|did|can|could|may|might|"
    r"must|shall|should|will|would)(?:\s+been)?\s+)?(?:not\s+)?"
)
_EXPLICIT_OTHER_SUBJECT = re.compile(
    r"\s*(?:the|a|an|this|that|these|those|our|their|finite|graph)\b"
)


def _frozen_source():
    """Load the trusted frozen input from /tests without reading workspace bytes."""
    try:
        frozen = E / "input.json"
        if frozen.is_symlink() or not is_regular_bounded_file(frozen, max_bytes=None):
            return {}
        value = json.loads(frozen.read_bytes())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _graph(value, limits):
    if not isinstance(value, dict) or set(value) != {
        "role",
        "vertex_count",
        "edges",
        "induced_c4_count",
        "c4_free_characteristic",
    }:
        return None
    n, edges = value["vertex_count"], value["edges"]
    if (
        type(n) is not int
        or not limits[0] <= n <= limits[1]
        or not isinstance(edges, list)
        or len(edges) > 36
    ):
        return None
    if any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(type(x) is not int for x in edge)
        for edge in edges
    ):
        return None
    pairs = [tuple(edge) for edge in edges]
    if pairs != sorted(set(pairs)) or any(not (0 <= a < b < n) for a, b in pairs):
        return None
    adjacency = [set() for _ in range(n)]
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for u in adjacency[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    if len(seen) != n:
        return None
    return n, adjacency


def _has_cycle_on(vertices, adjacency):
    first = min(vertices)
    rest = [v for v in vertices if v != first]
    return any(
        all(order[(i + 1) % 4] in adjacency[order[i]] for i in range(4))
        for tail in itertools.permutations(rest)
        for order in [(first, *tail)]
    )


def _invariants(n, adjacency):
    induced = 0
    has_c4 = False
    for vertices in itertools.combinations(range(n), 4):
        cycle = _has_cycle_on(vertices, adjacency)
        has_c4 |= cycle
        edge_count = sum(
            v in adjacency[u] for u, v in itertools.combinations(vertices, 2)
        )
        degrees = [sum(v in adjacency[u] for v in vertices if v != u) for u in vertices]
        induced += int(edge_count == 4 and degrees == [2, 2, 2, 2])
    return induced, int(not has_c4)


def _is_exact_int(value):
    """Reject JSON booleans that compare equal to 0 or 1."""
    return type(value) is int


def _witness_valid(witness, contracts):
    """Validate a single graph witness, rejecting booleans and bad roles."""
    role = witness.get("role")
    if not _is_exact_int(witness.get("induced_c4_count")):
        return False
    if not _is_exact_int(witness.get("c4_free_characteristic")):
        return False
    parsed = _graph(witness, contracts.get(role, (-1, -1)))
    if parsed is None:
        return False
    induced, characteristic = _invariants(*parsed)
    if (
        witness["induced_c4_count"] != induced
        or witness["c4_free_characteristic"] != characteristic
    ):
        return False
    if role == "C4_FREE_ZERO_COUNT" and not (induced == 0 and characteristic == 1):
        return False
    if role == "MULTIPLE_INDUCED_C4" and not (induced >= 2 and characteristic == 0):
        return False
    if role == "CHORDED_C4_ZERO_INDUCED":
        return induced == 0 and characteristic == 0
    return True


def _valid(result, source):
    if (
        not isinstance(result, dict)
        or set(result) != {"semantic_status", "witnesses"}
        or result.get("semantic_status") != "NOT_EQUIVALENT"
    ):
        return False
    witnesses = result.get("witnesses")
    contracts = {
        x["role"]: (x["min_vertices"], x["max_vertices"])
        for x in source.get("witness_contracts", [])
        if isinstance(x, dict)
    }
    if (
        not isinstance(witnesses, list)
        or len(witnesses) != 3
        or not all(isinstance(x, dict) for x in witnesses)
    ):
        return False
    # Validate roles are hashable strings before constructing the set.
    if not all(isinstance(x.get("role"), str) for x in witnesses):
        return False
    if {x.get("role") for x in witnesses} != ROLES:
        return False
    return all(_witness_valid(witness, contracts) for witness in witnesses)


def _evidence(evidence, result):
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not isinstance(evidence[0], dict)
        or set(evidence[0]) != {"path", "sha256"}
        or evidence[0].get("path") != "evidence/answer.txt"
    ):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        marker = None
        has_induced = False
        has_characteristic = False
        has_lean_limitation = False
        has_conjecture_limitation = False
        for line in _bounded_evidence_lines(target):
            if line.startswith("RESULT_JSON:"):
                if marker is not None:
                    return False
                marker = line[12:].strip()
                continue
            folded = line.casefold()
            has_induced |= "induced" in folded
            has_characteristic |= "characteristic" in folded
            if _has_affirmative_prohibited_claim(folded):
                return False
            has_lean_limitation |= _has_limitation(folded, "lean")
            has_conjecture_limitation |= _has_limitation(folded, "conjecture")
        if marker is None:
            return False
        canonical_marker = json.dumps(
            json.loads(marker), sort_keys=True, separators=(",", ":")
        )
        canonical_result = json.dumps(result, sort_keys=True, separators=(",", ":"))
        return (
            canonical_marker == canonical_result
            and has_induced
            and has_characteristic
            and has_lean_limitation
            and has_conjecture_limitation
        )
    except (
        OSError,
        ValueError,
        UnicodeError,
        StopIteration,
        RecursionError,
        MemoryError,
    ):
        return False


def _bounded_evidence_lines(target):
    """Yield UTF-8 evidence lines without buffering an unbounded line."""
    pending = bytearray()
    with target.open("rb") as stream:
        while chunk := stream.read(_EVIDENCE_READ_BYTES):
            start = 0
            while True:
                newline = chunk.find(b"\n", start)
                fragment = chunk[start:] if newline == -1 else chunk[start:newline]
                if len(pending) + len(fragment) > _MAX_EVIDENCE_LINE_BYTES:
                    raise ValueError("evidence line exceeds parser bound")
                pending.extend(fragment)
                if newline == -1:
                    break
                yield pending.decode("utf-8")
                pending.clear()
                start = newline + 1
                if start == len(chunk):
                    break
        if pending:
            yield pending.decode("utf-8")


def _limitations_valid(limitations):
    """Reject affirmative Lean/proof overclaims.

    The task prohibits claiming Lean compilation or proof of the corrected
    conjecture. A valid limitation must explicitly state that Lean is not
    assessed, and must not assert that either Lean or the corrected conjecture
    was proved/verified.
    """
    if not isinstance(limitations, list):
        return False
    has_lean_limitation = False
    has_conjecture_limitation = False
    for item in limitations:
        if not isinstance(item, str):
            return False
        folded = item.casefold()
        if _has_affirmative_prohibited_claim(folded):
            return False
        has_lean_limitation |= _has_limitation(folded, "lean")
        has_conjecture_limitation |= _has_limitation(folded, "conjecture")
    return has_lean_limitation and has_conjecture_limitation


def _has_limitation(text, topic):
    return any(
        topic in clause
        and (
            "not assessed" in clause
            or re.search(r"\b(?:no|not|never)\b.{0,80}\b(?:claim|claimed)\b", clause)
        )
        for clause in re.split(r"[.;]", text)
    )


def _has_affirmative_prohibited_claim(text):
    for sentence in re.split(r"[.;]", text):
        theorem_context = False
        for clause in re.split(r"\s*,?\s+(?:and|but)\s+", sentence):
            matches = tuple(_PROHIBITED_CLAIM.finditer(clause))
            theorem_subject = bool(_THEOREM_SUBJECT.match(clause))
            claim_is_elided = bool(
                matches
                and _ELIDED_SUBJECT_PREFIX.fullmatch(clause[: matches[0].start()])
            )
            if theorem_subject:
                theorem_context = True
            elif (matches and not claim_is_elided) or (
                not matches and _EXPLICIT_OTHER_SUBJECT.match(clause)
            ):
                theorem_context = False
            if not theorem_context:
                continue
            for match in matches:
                prefix = clause[: match.start()][-80:]
                if not re.search(r"\b(?:no|never)\b|\bnot\b(?!\s+only\b)", prefix):
                    return True
    return False


def main():
    submission, source = load_submission(), _frozen_source()
    expected = json.loads((E / "expected.json").read_text())
    input_bound = workspace_input_is_bound(W / "input.json", tests=E)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED_ASSURANCES,
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(_valid(result, source))
    evidence_valid = bool(
        isinstance(submission, dict)
        and _evidence(submission.get("evidence"), submission.get("result"))
    )
    scope_correct = bool(
        isinstance(submission, dict)
        and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    completeness_correct = bool(
        isinstance(submission, dict) and submission.get("completeness") == "COMPLETE"
    )
    limitations = (
        submission.get("limitations", []) if isinstance(submission, dict) else []
    )
    limitations_correct = _limitations_valid(limitations)
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract
        and math_correct
        and evidence_valid
        and scope_correct
        and completeness_correct
        and limitations_correct
        and input_bound
        and not false_certification
    )
    reward = (
        0
        if not correct
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "completeness_accuracy": float(completeness_correct),
                "limitation_accuracy": float(limitations_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
