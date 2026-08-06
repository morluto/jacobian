import itertools
import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
FIXTURE_NAME = "mathematical-benchmarks-v1-finite-magma-countermodel-input.json"


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def evidence_matches_result(evidence, result):
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().casefold()
        return all(
            fragment in text
            for fragment in (
                "x diamond y = y",
                "x=0",
                "y=1",
                "one-element",
                "minimality",
            )
        )
    except (OSError, UnicodeError):
        return False


def evaluate(term, table, assignment):
    if not isinstance(term, dict) or len(term) != 1:
        raise ValueError("malformed term")
    if "var" in term:
        variable = term["var"]
        if variable not in assignment:
            raise ValueError("unknown variable")
        return assignment[variable]
    operands = term.get("op")
    if not isinstance(operands, list) or len(operands) != 2:
        raise ValueError("malformed operation")
    left = evaluate(operands[0], table, assignment)
    right = evaluate(operands[1], table, assignment)
    return table[left][right]


def law_holds(law, table, assignment):
    return evaluate(law["left"], table, assignment) == evaluate(
        law["right"], table, assignment
    )


def all_assignments(order):
    return ({"x": x, "y": y} for x in range(order) for y in range(order))


def table_valid(table, order):
    return bool(
        isinstance(table, list)
        and len(table) == order
        and all(
            isinstance(row, list)
            and len(row) == order
            and all(type(value) is int and 0 <= value < order for value in row)
            for row in table
        )
    )


def has_countermodel(order, premise, target):
    for flat_table in itertools.product(range(order), repeat=order * order):
        table = [
            list(flat_table[row * order : (row + 1) * order]) for row in range(order)
        ]
        if all(
            law_holds(premise, table, assignment)
            for assignment in all_assignments(order)
        ) and any(
            not law_holds(target, table, assignment)
            for assignment in all_assignments(order)
        ):
            return True
    return False


def result_valid(result, fixture):
    required = {
        "order",
        "table",
        "refuting_assignment",
        "premise_holds_universally",
        "target_holds_universally",
        "minimality_checked_orders",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    try:
        search_orders = fixture["search_orders"]
        premise = fixture["premise"]
        target = fixture["target"]
        order = result["order"]
        table = result["table"]
        assignment = result["refuting_assignment"]
        if (
            not isinstance(search_orders, list)
            or search_orders != sorted(set(search_orders))
            or any(type(item) is not int or item < 1 for item in search_orders)
            or type(order) is not int
            or order not in search_orders
            or not table_valid(table, order)
            or not isinstance(assignment, dict)
            or set(assignment) != {"x", "y"}
            or any(type(assignment[name]) is not int for name in ("x", "y"))
            or any(not 0 <= assignment[name] < order for name in ("x", "y"))
            or result["premise_holds_universally"] is not True
            or result["target_holds_universally"] is not False
        ):
            return False
        smaller_orders = [candidate for candidate in search_orders if candidate < order]
        return bool(
            result["minimality_checked_orders"] == smaller_orders
            and all(
                law_holds(premise, table, candidate)
                for candidate in all_assignments(order)
            )
            and not law_holds(target, table, assignment)
            and any(
                not law_holds(target, table, candidate)
                for candidate in all_assignments(order)
            )
            and all(
                not has_countermodel(candidate, premise, target)
                for candidate in smaller_orders
            )
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def main():
    submission = load_submission()
    fixture = load_json(E / FIXTURE_NAME)
    canonical_fixture = load_json(E / FIXTURE_NAME)
    expected = load_json(E / "expected.json")
    if not isinstance(expected, dict):
        expected = {}
    math_contract = strict_submission_contract(
        submission,
        task_id=expected.get("task_id", "invalid"),
        conclusion=expected.get("conclusion", "invalid"),
        verification_record="optional",
    )
    accepted_contract = strict_submission_contract(
        submission,
        task_id=expected.get("task_id", "invalid"),
        conclusion=expected.get("conclusion", "invalid"),
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    fixture_bound = bool(fixture is not None and fixture == canonical_fixture)
    math_correct = bool(
        math_contract and fixture_bound and result_valid(result, fixture)
    )
    correct = bool(accepted_contract and math_correct)
    evidence = bool(
        math_contract and evidence_matches_result(submission.get("evidence"), result)
    )
    scope = bool(
        math_contract and submission.get("scope") == expected.get("required_scope")
    )
    assurance = bool(
        math_contract
        and submission.get("claimed_assurance") == expected.get("maximum_assurance")
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0.0
        if not correct or false_certification
        else 0.7 + 0.1 * evidence + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
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
