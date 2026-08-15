import copy
import json
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W, E = Path("/app"), Path("/tests")


def _frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (W / "input.json").is_symlink() or (W / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _valid_node(node, budget):
    if not isinstance(node, dict) or budget[0] <= 0:
        return False
    budget[0] -= 1
    op = node.get("op")
    if op == "atom":
        return set(node) == {"op", "name"} and isinstance(node["name"], str)
    if op == "false":
        return set(node) == {"op"}
    if set(node) != {"op", "args"} or not isinstance(node["args"], list):
        return False
    arity = len(node["args"])
    if (op == "not" and arity != 1) or (op == "imp" and arity != 2):
        return False
    if op in {"and", "or"} and not 2 <= arity <= 16:
        return False
    if op not in {"not", "imp", "and", "or"}:
        return False
    return all(_valid_node(child, budget) for child in node["args"])


def _at(root, path):
    node = root
    for index in path:
        if type(index) is not int or not isinstance(node.get("args"), list):
            return None
        if index < 0 or index >= len(node["args"]):
            return None
        node = node["args"][index]
    return node


def _replace(root, path, replacement):
    if not path:
        return replacement
    result = copy.deepcopy(root)
    parent = _at(result, path[:-1])
    if parent is None:
        return None
    parent["args"][path[-1]] = replacement
    return result


def _neg(node):
    return {"op": "not", "args": [copy.deepcopy(node)]}


def _rewrite_de_morgan_or(node):
    if node.get("op") != "not":
        return None
    child = node["args"][0]
    if child.get("op") == "or":
        return {"op": "and", "args": [_neg(item) for item in child["args"]]}
    return None


def _rewrite_not_implication(node):
    if node.get("op") != "not":
        return None
    child = node["args"][0]
    if child.get("op") == "imp":
        return {
            "op": "and",
            "args": [copy.deepcopy(child["args"][0]), _neg(child["args"][1])],
        }
    return None


def _rewrite_double_negation(node):
    if node.get("op") != "not":
        return None
    child = node["args"][0]
    if child.get("op") == "not":
        return copy.deepcopy(child["args"][0])
    return None


def _rewrite_flatten_associative(node):
    if node.get("op") not in {"and", "or"}:
        return None
    operator = node["op"]
    if any(child.get("op") == operator for child in node["args"]):
        flattened = []
        for child in node["args"]:
            flattened.extend(child["args"] if child.get("op") == operator else [child])
        return {"op": operator, "args": copy.deepcopy(flattened)}
    return None


def _rewrite_contradiction(node):
    if node.get("op") != "and":
        return None
    args = node["args"]
    for candidate in args:
        if candidate.get("op") == "not" and candidate["args"][0] in args:
            return {"op": "false"}
        if _neg(candidate) in args:
            return {"op": "false"}
    return None


_REWRITERS = {
    "DE_MORGAN_OR": _rewrite_de_morgan_or,
    "NOT_IMPLICATION": _rewrite_not_implication,
    "DOUBLE_NEGATION": _rewrite_double_negation,
    "FLATTEN_ASSOCIATIVE": _rewrite_flatten_associative,
    "CONTRADICTION": _rewrite_contradiction,
}


def _rewrite(node, rule):
    rewriter = _REWRITERS.get(rule)
    if rewriter is None:
        return None
    return rewriter(node)


def _trace_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {"steps", "final_ast"}:
        return False
    steps = result["steps"]
    bounds = frozen.get("step_bounds", {})
    if (
        not isinstance(steps, list)
        or not isinstance(bounds, dict)
        or not bounds.get("minimum") <= len(steps) <= bounds.get("maximum")
    ):
        return False
    current = frozen.get("initial_ast")
    if not _valid_node(current, [500]):
        return False
    used = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"rule", "path", "after_ast"}:
            return False
        rule, path, after = step["rule"], step["path"], step["after_ast"]
        if rule not in frozen.get("registered_rules", []) or not isinstance(path, list):
            return False
        target = _at(current, path)
        replacement = _rewrite(target, rule) if target is not None else None
        computed = (
            _replace(current, path, replacement) if replacement is not None else None
        )
        if computed != after or not _valid_node(after, [500]):
            return False
        current = after
        used.add(rule)
    return bool(
        current == frozen.get("target_ast")
        and result["final_ast"] == current
        and {"DE_MORGAN_OR", "NOT_IMPLICATION", "DOUBLE_NEGATION", "CONTRADICTION"}
        <= used
    )


def main():
    submission, frozen = load_submission(), _frozen()
    math_correct = bool(submission and _trace_valid(submission.get("result"), frozen))
    evidence = None
    if (
        submission
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
    ):
        evidence = read_evidence_json(
            submission["witness"][0], expected_path="evidence/rewrite-trace.json"
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == "propositional-rewrite-trace-replay"
        and json_value_equal(evidence["result"], submission.get("result"))
    )
    reward = float(math_correct and evidence_valid)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(evidence_valid),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
