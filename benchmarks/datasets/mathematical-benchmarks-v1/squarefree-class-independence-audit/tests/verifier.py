import json
import re
from itertools import product
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
CLASSIFICATION = {
    "class_key": "SQUAREFREE_KERNEL",
    "product_square_iff": "KERNELS_EQUAL",
    "pair_count_formula": "SUM_OF_SQUARED_CLASS_SIZES",
    "independent_selection": "ONE_ELEMENT_PER_DISTINCT_CLASS",
}
LIMITATION = (
    "The verifier replays the modular obstruction and structured reduction but "
    "does not machine-check a formal proof of the squarefree-kernel lemma."
)


def load_frozen() -> dict:
    try:
        app_input = WORKSPACE / "input.json"
        test_input = TESTS / "input.json"
        if (
            any(
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > 1_048_576
                for path in (app_input, test_input)
            )
            or app_input.read_bytes() != test_input.read_bytes()
        ):
            return {}
        value = json.loads(test_input.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def certificate_valid(result: object, frozen: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "ordered_pair_count",
        "classification",
        "modular_obstruction",
        "consequence",
    }:
        return False
    obstruction = result.get("modular_obstruction")
    if not isinstance(obstruction, dict) or set(obstruction) != {
        "modulus",
        "target_residue",
        "quadratic_residues",
        "maximum_squares_ruled_out",
    }:
        return False
    bounds = frozen.get("certificate_bounds", {})
    modulus = obstruction.get("modulus")
    residues = obstruction.get("quadratic_residues")
    if (
        type(modulus) is not int
        or not bounds.get("minimum_modulus", 2)
        <= modulus
        <= bounds.get("maximum_modulus", 0)
        or not isinstance(residues, list)
        or any(type(value) is not int for value in residues)
        or residues != sorted(set(residues))
    ):
        return False
    expected_residues = sorted({pow(value, 2, modulus) for value in range(modulus)})
    target = 2023 % modulus
    if residues != expected_residues or obstruction.get("target_residue") != target:
        return False
    if obstruction.get("maximum_squares_ruled_out") != 3:
        return False

    # Zero padding makes this one exhaustive check for representations by
    # zero, one, two, or three integer squares.
    if any(sum(values) % modulus == target for values in product(residues, repeat=3)):
        return False
    return bool(
        result.get("ordered_pair_count") == 2023
        and result.get("classification") == CLASSIFICATION
        and result.get("consequence") == "AT_LEAST_FOUR_SQUAREFREE_CLASSES"
    )


def evidence_valid(evidence: object, result: object) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        if path.stat().st_size > 1_048_576:
            return False
        lines = path.read_text().splitlines()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in lines
            if line.startswith("RESULT_JSON:")
        )
        body = "\n".join(line for line in lines if not line.startswith("RESULT_JSON:"))
        text = body.casefold()
        compact = "".join(text.split())
        obstruction = result["modular_obstruction"] if isinstance(result, dict) else {}
        marker_result = json.loads(marker)
        expected_fragments = (
            f"ordered_pair_count={result.get('ordered_pair_count')}",
            f"modulus={obstruction.get('modulus')}",
            f"target_residue={obstruction.get('target_residue')}",
            f"quadratic_residues={obstruction.get('quadratic_residues')}",
            f"maximum_squares_ruled_out={obstruction.get('maximum_squares_ruled_out')}",
        )
    except (
        OSError,
        StopIteration,
        UnicodeError,
        ValueError,
        RecursionError,
        TypeError,
    ):
        return False
    contradictory = re.search(
        r"\b(?:not|never)\b[^.]{0,80}\bsquarefree\s+kernel", text
    ) or re.search(r"\b(?:not|never)\b[^.]{0,80}\bat\s+least\s+four", text)
    return bool(
        isinstance(result, dict)
        and marker_result == result
        and len(body) >= 160
        and not contradictory
        and all(fragment.replace(" ", "") in compact for fragment in expected_fragments)
        and "squarefree kernel" in text
        and "product" in text
        and "kernel" in text
        and any(
            phrase in text
            for phrase in ("kernels agree", "kernels equal", "same kernel")
        )
        and "sum" in text
        and "squared" in text
        and "class sizes" in text
        and "one element" in text
        and "distinct class" in text
        and "modular" in text
        and "2023" in text
        and "at least four" in text
        and "computed" in text
    )


def limitation_is_unchecked(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.casefold()
    verification = (
        r"(?:machine(?:[- ](?:check(?:ed)?|verified|verification|proof))?"
        r"|formal(?:ly)?(?:[- ](?:checked|verified|verification|proof))?"
        r"|proof[- ]assistant(?:[- ](?:checked|verified|verification|proof))?)"
    )
    negative_pattern = (
        r"\b(?:not|no|without|does not|doesn't)\b[^.]{0,60}\b" + verification + r"\b"
    )
    negative = re.search(negative_pattern, text)
    remainder = re.sub(negative_pattern, "", text)
    return bool(negative and not re.search(r"\b" + verification + r"\b", remainder))


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
    math_correct = bool(
        contract and certificate_valid(data.get("result"), load_frozen())
    )
    evidence_correct = bool(
        math_correct and evidence_valid(data.get("evidence"), data.get("result"))
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(limitation_is_unchecked(item) for item in data["limitations"])
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_correct
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_correct),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
