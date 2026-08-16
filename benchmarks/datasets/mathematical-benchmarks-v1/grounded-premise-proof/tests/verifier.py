import json
import re
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")
REQUIRED_PREMISES = {"subgroup_of_abelian_is_normal", "coset_product_definition"}
RULES = {
    "APPLY_NORMALITY_PREMISE": (
        {"G_ABELIAN", "N_SUBGROUP"},
        "N_NORMAL",
        "subgroup_of_abelian_is_normal",
    ),
    "FORM_QUOTIENT": ({"N_NORMAL"}, "QUOTIENT_EXISTS", None),
    "EXPAND_XY_COSET_PRODUCT": (
        {"QUOTIENT_EXISTS", "X_EQ_xN", "Y_EQ_yN"},
        "XY_EQ_xyN",
        "coset_product_definition",
    ),
    "COMMUTE_REPRESENTATIVES": ({"G_ABELIAN", "XY_EQ_xyN"}, "XY_EQ_yxN", None),
    "COLLAPSE_YX_COSET_PRODUCT": (
        {"QUOTIENT_EXISTS", "Y_EQ_yN", "X_EQ_xN"},
        "yxN_EQ_YX",
        "coset_product_definition",
    ),
    "CHAIN_EQUALITIES": ({"XY_EQ_yxN", "yxN_EQ_YX"}, "XY_EQ_YX", None),
}
STEP_ID_PATTERN = re.compile("^[a-z][a-z0-9_]{0,31}$")


def _valid_step_shape(step, step_ids):
    return bool(
        isinstance(step, dict)
        and set(step) == {"id", "rule", "inputs", "output"}
        and (type(step["id"]) is str)
        and (type(step["rule"]) is str)
        and (type(step["output"]) is str)
        and (STEP_ID_PATTERN.fullmatch(step["id"]) is not None)
        and (step["id"] not in step_ids)
        and (step["rule"] in RULES)
    )


def _valid_step_inputs(step, facts):
    inputs = step["inputs"]
    return bool(
        isinstance(inputs, list)
        and all(type(value) is str for value in inputs)
        and (len(inputs) == len(set(inputs)))
        and (set(inputs) <= facts)
    )


def _apply_rule(step, facts, selected, used_premises):
    required_inputs, output, premise = RULES[step["rule"]]
    if set(step["inputs"]) != required_inputs or step["output"] != output:
        return False
    if output in facts or (premise is not None and premise not in selected):
        return False
    if premise is not None:
        used_premises.add(premise)
    facts.add(output)
    return True


def _replay_proof(result, source):
    if not isinstance(result, dict) or set(result) != {
        "selected_premises",
        "proof_steps",
        "target_fact",
    }:
        return False
    selected = result["selected_premises"]
    if (
        not isinstance(selected, list)
        or not all(type(premise) is str for premise in selected)
        or len(selected) != len(REQUIRED_PREMISES)
        or (set(selected) != REQUIRED_PREMISES)
    ):
        return False
    available_premises = {
        premise["id"] for premise in source.get("candidate_premises", [])
    }
    if not set(selected) <= available_premises:
        return False
    facts = set(source.get("initial_facts", []))
    used_premises = set()
    step_ids = set()
    steps = result["proof_steps"]
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not _valid_step_shape(step, step_ids):
            return False
        if not _valid_step_inputs(step, facts):
            return False
        if not _apply_rule(step, facts, selected, used_premises):
            return False
        step_ids.add(step["id"])
    target = source.get("target_fact")
    return bool(
        type(result["target_fact"]) is str
        and result["target_fact"] == target
        and (target in facts)
        and (used_premises == REQUIRED_PREMISES)
    )


def main():
    submission = load_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    input_contract = source == json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = bool(
        input_contract
        and isinstance(submission, dict)
        and _replay_proof(submission.get("result"), source)
    )
    correct = math_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": float(correct)})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
