import json
from pathlib import Path

from verifier_support import (
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


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


def evidence_valid(descriptors, result):
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        return False
    target = resolve_evidence(descriptors[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text(encoding="utf-8")
        markers = [
            line[12:] for line in text.splitlines() if line.startswith("RESULT_JSON:")
        ]
        if len(markers) != 1:
            return False
        return (
            json.loads(markers[0]) == result
            and "every element" in text.lower()
            and "common fixed" in text.lower()
            and "not verified" in text.lower()
        )
    except (OSError, ValueError, UnicodeError, RecursionError):
        return False


def main():
    submission = load_submission()
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
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = bool(
        math_contract and source_bound and certificate_valid(result, frozen)
    )
    evidence = bool(
        math_contract and evidence_valid(submission.get("evidence"), result)
    )
    scope = bool(
        math_contract and submission.get("scope") == expected["required_scope"]
    )
    assurance = bool(
        math_contract and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    correct = bool(accepted_contract and mathematical and not false_certification)
    reward = (
        0.0 if not correct else 0.7 + 0.1 * evidence + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
