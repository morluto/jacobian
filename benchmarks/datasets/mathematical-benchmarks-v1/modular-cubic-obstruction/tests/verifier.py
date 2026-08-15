import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _complete_obstruction(result):
    if not isinstance(result, dict) or set(result) != {
        "modulus",
        "target_residue",
        "residue_cases",
    }:
        return False
    modulus = result["modulus"]
    target = result["target_residue"]
    cases = result["residue_cases"]
    if (
        type(modulus) is not int
        or modulus != 7
        or type(target) is not int
        or not 0 <= target < modulus
        or not isinstance(cases, list)
        or len(cases) != modulus
    ):
        return False

    observed = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "x_residue",
            "x_cube_residue",
            "lhs_residue",
        }:
            return False
        x = case["x_residue"]
        cube = case["x_cube_residue"]
        lhs = case["lhs_residue"]
        if (
            type(x) is not int
            or type(cube) is not int
            or type(lhs) is not int
            or not 0 <= x < modulus
            or x in observed
            or cube != pow(x, 3, modulus)
            or lhs != (4 * cube) % modulus
        ):
            return False
        observed[x] = lhs

    if set(observed) != set(range(modulus)) or target != 2003 % modulus:
        return False

    # Independently check every residue pair. This does not trust the submitted
    # shortcut that the y term vanishes for the selected modulus.
    return all(
        (4 * pow(x, 3, modulus) - 7 * pow(y, 3, modulus)) % modulus != target
        for x in range(modulus)
        for y in range(modulus)
    )


def main():
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    math_ok = bool(
        submission is not None and _complete_obstruction(submission.get("result"))
    )
    reward = float(math_ok and input_binding)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
