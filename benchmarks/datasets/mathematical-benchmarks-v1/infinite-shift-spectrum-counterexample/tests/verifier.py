import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _right(index):
    return index + 1


def _left(index):
    return None if index == 0 else index - 1


def _compose(outer, inner, index):
    intermediate = inner(index)
    return None if intermediate is None else outer(intermediate)


def _expected_orientation(orientation):
    if orientation == "S_RIGHT_T_LEFT":
        return _right, _left, "ST", "TS"
    if orientation == "S_LEFT_T_RIGHT":
        return _left, _right, "TS", "ST"
    return None


def _valid_actions(actions, s_action, t_action, start, end):
    if not isinstance(actions, list) or len(actions) != end - start + 1:
        return False
    seen = set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {
            "basis_index",
            "s_output",
            "t_output",
            "st_output",
            "ts_output",
        }:
            return False
        index = action["basis_index"]
        if type(index) is not int or index in seen or not start <= index <= end:
            return False
        seen.add(index)
        expected = {
            "basis_index": index,
            "s_output": s_action(index),
            "t_output": t_action(index),
            "st_output": _compose(s_action, t_action, index),
            "ts_output": _compose(t_action, s_action, index),
        }
        for key, expected_value in expected.items():
            actual_value = action[key]
            if expected_value is None:
                if actual_value is not None:
                    return False
            elif type(actual_value) is not int or actual_value != expected_value:
                return False
        if action != expected:
            return False
    return True


def _valid_result(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "orientation",
        "basis_window",
        "actions",
        "zero_eigenvalue_product",
        "identity_product",
        "zero_eigenvector_basis_index",
        "spectral_conclusion",
        "missing_assumption",
    }:
        return False
    orientation = _expected_orientation(result.get("orientation"))
    window = frozen.get("basis_window")
    if (
        orientation is None
        or window != [0, 8]
        or result.get("basis_window") != window
        or not all(type(value) is int for value in result["basis_window"])
    ):
        return False
    s_action, t_action, zero_product, identity_product = orientation
    return bool(
        _valid_actions(result.get("actions"), s_action, t_action, *window)
        and result.get("zero_eigenvalue_product") == zero_product
        and result.get("identity_product") == identity_product
        and type(result.get("zero_eigenvector_basis_index")) is int
        and result.get("zero_eigenvector_basis_index") == 0
        and result.get("spectral_conclusion") == "EIGENVALUE_SETS_DIFFER"
        and result.get("missing_assumption") == "FINITE_DIMENSIONALITY"
    )


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    frozen = _load_frozen_input()
    math_correct = bool(protocol_ok and _valid_result(submission.get("result"), frozen))
    reward = aggregate_reward(
        correctness=math_correct,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
