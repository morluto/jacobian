import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _source() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _add(left: list[int], right: list[int]) -> list[int]:
    size = max(len(left), len(right))
    result = [
        (left[i] if i < len(left) else 0) + (right[i] if i < len(right) else 0)
        for i in range(size)
    ]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _mul(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _scale(poly: list[int], scalar: int) -> list[int]:
    return [scalar * value for value in poly]


def _cube(poly: list[int]) -> list[int]:
    return _mul(_mul(poly, poly), poly)


def _affine(value: object) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    slope, intercept = value
    return [intercept, slope]


def _strict_int_list(value: object) -> bool:
    return isinstance(value, list) and all(type(item) is int for item in value)


def _family(value: object) -> set[int] | None:
    if not isinstance(value, dict) or set(value) != {
        "parameter_min",
        "A",
        "B",
        "C",
        "value",
        "covered_residues",
    }:
        return None
    minimum = value["parameter_min"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        return None
    coordinates = [_affine(value[name]) for name in ("A", "B", "C")]
    target = _affine(value["value"])
    if any(poly is None for poly in coordinates) or target is None:
        return None
    assert all(poly is not None for poly in coordinates)
    polys = [poly for poly in coordinates if poly is not None]
    if any(poly[1] < 0 or poly[0] + poly[1] * minimum < 0 for poly in polys):
        return None
    expression = _add(_add(_cube(polys[0]), _cube(polys[1])), _cube(polys[2]))
    expression = _add(expression, _scale(_mul(_mul(polys[0], polys[1]), polys[2]), -3))
    normalized_target = _add(target, [0]) if target is not None else None
    if (
        expression != normalized_target
        or target is None
        or target[1] < 0
        or target[0] + target[1] * minimum < 0
    ):
        return None
    residues = {(target[1] * (minimum + step) + target[0]) % 9 for step in range(9)}
    declared = value["covered_residues"]
    if (
        not _strict_int_list(declared)
        or len(declared) != len(residues)
        or set(declared) != residues
    ):
        return None
    return residues


def _residue_lists_match(
    value: dict[str, Any], image: list[int], excluded: list[int]
) -> bool:
    return (
        _strict_int_list(value["image_residues_mod_9"])
        and _strict_int_list(value["excluded_residues_mod_9"])
        and value["image_residues_mod_9"] == image
        and value["excluded_residues_mod_9"] == excluded
    )


def _collect_family_coverage(
    families: list[object], excluded: list[int]
) -> tuple[set[int], set[int]] | None:
    covers: set[int] = set()
    covered_values: set[int] = set()
    for family in families:
        residues = _family(family)
        if residues is None or residues & set(excluded):
            return None
        covers.update(residues)
        target = _affine(family["value"])
        assert target is not None
        if target[1] == 0:
            if 0 <= target[0] <= 500:
                covered_values.add(target[0])
            continue
        for parameter in range(family["parameter_min"], 501):
            candidate = target[0] + target[1] * parameter
            if 0 <= candidate <= 500:
                covered_values.add(candidate)
    return covers, covered_values


def _result(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "factorization",
        "image_residues_mod_9",
        "excluded_residues_mod_9",
        "families",
    }:
        return False
    provenance = source.get("source", {})
    if provenance.get("revision") != "dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c":
        return False
    if value["factorization"] != {
        "linear": "A+B+C",
        "quadratic": "A^2+B^2+C^2-AB-AC-BC",
    }:
        return False
    image = sorted(
        {
            (a**3 + b**3 + c**3 - 3 * a * b * c) % 9
            for a in range(9)
            for b in range(9)
            for c in range(9)
        }
    )
    excluded = sorted(set(range(9)) - set(image))
    if not _residue_lists_match(value, image, excluded):
        return False
    families = value["families"]
    if not isinstance(families, list) or len(families) < 3:
        return False
    coverage = _collect_family_coverage(families, excluded)
    if coverage is None:
        return False
    covers, covered_values = coverage
    return covers == set(image) and all(
        value in covered_values for value in range(501) if value % 9 in image
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(
        isinstance(submission, dict) and _result(data.get("result"), _source())
    )
    correct = math_correct
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
