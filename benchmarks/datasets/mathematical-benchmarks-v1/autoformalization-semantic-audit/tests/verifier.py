import json
import re
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
NON_COMPILATION_LIMITATION = (
    "Lean parsing, elaboration, and compilation are not assessed."
)


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


REQUIRED_DEFECTS = {
    "MISSING_DIMENSION_PREMISE",
    "DOT_PRODUCT_REPLACED_BY_COORDINATEWISE_ZERO",
}


def _is_non_compilation_limitation(value):
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.casefold().split())
    return bool(
        "lean" in normalized
        and re.search(
            r"compil(?:e|ed|es|ation)|elaborat(?:e|ed|ion)|pars(?:e|ing)", normalized
        )
        and re.search(
            r"\b(?:not|no|without|outside|unchecked|unassessed|unverified|"
            r"not checked|not assessed|not evaluated|does not)\b",
            normalized,
        )
    )


def _integer_vector(value, dimension):
    return bool(
        isinstance(value, list)
        and len(value) == dimension
        and all(type(entry) is int for entry in value)
    )


def _missing_premise_is_certified(certificate):
    if not isinstance(certificate, dict) or set(certificate) != {
        "dimension",
        "x",
        "forced_y",
    }:
        return False
    dimension = certificate["dimension"]
    if type(dimension) is not int or dimension != 1:
        return False
    x = certificate["x"]
    forced_y = certificate["forced_y"]
    return bool(
        _integer_vector(x, dimension)
        and _integer_vector(forced_y, dimension)
        and x[0] != 0
        and forced_y == [0]
        and x[0] * forced_y[0] == 0
        and not any(forced_y)
    )


def _operator_mismatch_is_certified(certificate):
    if not isinstance(certificate, dict) or set(certificate) != {
        "dimension",
        "x",
        "y",
        "dot_product",
        "coordinate_products",
    }:
        return False
    dimension = certificate["dimension"]
    if type(dimension) is not int or dimension != 2:
        return False
    x = certificate["x"]
    y = certificate["y"]
    products = certificate["coordinate_products"]
    if not all(_integer_vector(vector, dimension) for vector in (x, y, products)):
        return False
    actual_products = [left * right for left, right in zip(x, y, strict=True)]
    actual_dot_product = sum(actual_products)
    return bool(
        any(y)
        and products == actual_products
        and certificate["dot_product"] == actual_dot_product == 0
        and any(product != 0 for product in actual_products)
    )


def _valid_semantic_audit(result, source):
    if not isinstance(result, dict) or set(result) != {
        "semantic_status",
        "defects",
        "missing_premise_certificate",
        "operator_mismatch_certificate",
    }:
        return False
    if source.get("audit_scope", {}).get("lean_compilation") is not False:
        return False
    defects = result["defects"]
    return bool(
        result["semantic_status"] == "NOT_EQUIVALENT"
        and isinstance(defects, list)
        and all(type(defect) is str for defect in defects)
        and set(defects) == REQUIRED_DEFECTS
        and len(defects) == len(REQUIRED_DEFECTS)
        and _missing_premise_is_certified(result["missing_premise_certificate"])
        and _operator_mismatch_is_certified(result["operator_mismatch_certificate"])
    )


def main():
    submission = load_submission()
    source = _load_frozen_input()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(_valid_semantic_audit(result, source))

    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(math_correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
