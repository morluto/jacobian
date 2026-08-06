import json
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    strict_submission_contract,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
LIMITATION = (
    "The verifier checks one finite linear action and does not machine-prove "
    "a general classification theorem."
)


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
        elements_raw = result["group_elements"]
        vectors_raw = result["fixed_vectors"]
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
        if any(
            v == (0, 0, 0) or apply(a, v, q) != v
            for a, v in zip(elements, vectors, strict=True)
        ):
            return False
        fixed_equations = [
            tuple((a[i][j] - int(i == j)) % q for j in range(3))
            for a in generators
            for i in range(3)
        ]
        common_dimension = 3 - rank(fixed_equations, q)
        return result["common_fixed_dimension"] == common_dimension == 0
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def raw_submission():
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def evidence_valid(descriptors):
    # The finite group/action certificate is replayed from the typed result.
    # The public evidence contract promises only one bound text artifact.
    return evidence_list_is_bound(descriptors)


def main():
    raw = raw_submission()
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    frozen = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    math_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="optional",
    )
    accepted_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    source = frozen.get("source", {})
    source_bound = bool(
        source.get("revision") == "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        and source.get("row") == 10001
        and source.get("row_sha256")
        == "sha256:8be03d5c7464658a27833f65a0d8b8198ea5d2d0658d088f667bb30754dd36da"
        and source.get("license") == "MIT"
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    mathematical = bool(source_bound and certificate_valid(result, frozen))
    evidence = bool(isinstance(raw, dict) and evidence_valid(raw.get("evidence")))
    scope = bool(
        isinstance(raw, dict) and raw.get("scope") == expected["required_scope"]
    )
    assurance = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED"
    )
    limitation = bool(isinstance(raw, dict) and raw.get("limitations") == [LIMITATION])
    false_certification = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
    )
    correct = bool(
        accepted_contract
        and input_binding
        and mathematical
        and evidence
        and scope
        and assurance
        and limitation
        and not false_certification
    )
    reward = 1.0 if correct else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(math_contract)),
                "input_binding": float(input_binding),
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "limitation_accuracy": float(limitation),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
