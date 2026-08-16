import json
from decimal import Decimal, DecimalException
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    normalize_reward_file,
    workspace_input_is_bound,
)

E = Path("/tests")


def _is_integer(value):
    """Accept any schema-valid integral JSON number while rejecting booleans.

    JSON Schema's ``integer`` type accepts numbers with a zero fractional part
    (e.g. ``12.0``), so the verifier must validate mathematical integrality
    rather than requiring Python's ``int`` representation. Booleans are still
    rejected because ``False == 0`` would otherwise spoof a zero element.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, Decimal):
        return value == value.to_integral_value()
    return False


def _is_int_list(value):
    return isinstance(value, list) and all(_is_integer(item) for item in value)


def _is_int_matrix(value):
    return bool(
        isinstance(value, list)
        and all(isinstance(row, list) for row in value)
        and all(_is_integer(item) for row in value for item in row)
    )


def _fits_small_integer(value, maximum):
    if not _is_integer(value):
        return False
    try:
        return -maximum <= value <= maximum
    except (DecimalException, OverflowError):
        return False


def _bounded_index(value, maximum):
    if not _is_integer(value):
        return None
    try:
        if value < 0 or value > maximum:
            return None
        return int(value)
    except (DecimalException, OverflowError):
        return None


def _valid_cover(result, bounds):
    n = result.get("modulus")
    step = result.get("subgroup_step")
    if (
        not _is_integer(n)
        or not _is_integer(step)
        or not bounds["minimum_modulus"] <= n <= bounds["maximum_modulus"]
        or not _fits_small_integer(n, bounds["maximum_modulus"])
        or not _fits_small_integer(step, bounds["maximum_modulus"])
    ):
        return False
    n = int(n)
    step = int(step)
    if (
        step < bounds["minimum_cosets"]
        or n % step
        or n // step < bounds["minimum_coset_size"]
    ):
        return False
    subgroup = list(range(0, n, step))
    representatives = list(range(step))
    cosets = [
        sorted((representative + value) % n for value in subgroup)
        for representative in representatives
    ]
    submitted_subgroup = result.get("subgroup")
    submitted_representatives = result.get("representatives")
    submitted_cosets = result.get("cosets")
    if not (
        _is_int_list(submitted_subgroup)
        and _is_int_list(submitted_representatives)
        and _is_int_matrix(submitted_cosets)
    ):
        return False
    # The published schema requires only unique integer elements, so the
    # subgroup and each coset are compared as unordered collections. The coset
    # list order is still fixed because covering-part references and the
    # duplicate index pair address cosets by position.
    return bool(
        sorted(submitted_subgroup) == sorted(subgroup)
        and submitted_representatives == representatives
        and [sorted(coset) for coset in submitted_cosets] == cosets
        and len({value for coset in cosets for value in coset}) == n
        and sum(len(coset) for coset in cosets) == n
    )


def _valid_predicates(result):
    artifact = result.get("part_artifact")
    references = result.get("covering_part_references")
    cosets = result.get("cosets")
    pair = result.get("duplicate_indices")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"id", "kind", "elements"}
        or not _is_integer(artifact.get("id"))
        or artifact.get("id") != 0
        or artifact.get("kind") != "SUBGROUP"
        or not _is_int_list(artifact.get("elements"))
        or sorted(artifact.get("elements")) != sorted(result.get("subgroup"))
        or not isinstance(references, list)
        or not isinstance(cosets, list)
        or len(references) != len(cosets)
        or any(not _is_integer(reference) or reference != 0 for reference in references)
    ):
        return False
    if not isinstance(pair, list) or len(pair) != 2:
        return False
    left = _bounded_index(pair[0], len(references) - 1)
    right = _bounded_index(pair[1], len(references) - 1)
    if left is None or right is None:
        return False
    # The submission schema only requires two distinct in-range integers, so the
    # exposing pair is accepted in either order; the remaining checks are
    # symmetric in left/right.
    if not (
        0 <= left < len(references) and 0 <= right < len(references) and left != right
    ):
        return False
    unique_parts = list(dict.fromkeys(references))
    sizes = {part: len(cosets[references.index(part)]) for part in unique_parts}
    range_pairwise = all(
        sizes[a] != sizes[b]
        for i, a in enumerate(unique_parts)
        for b in unique_parts[i + 1 :]
    )
    indexed_pairwise = all(
        len(cosets[i]) != len(cosets[j])
        for i in range(len(cosets))
        for j in range(i + 1, len(cosets))
    )
    return bool(
        result.get("range_pairwise_truth") is range_pairwise is True
        and result.get("indexed_pairwise_truth") is indexed_pairwise is False
        and references[left] == references[right] == artifact["id"]
        and len(cosets[left]) == len(cosets[right])
    )


def _valid_result(result, source):
    required = {
        "modulus",
        "subgroup_step",
        "subgroup",
        "representatives",
        "cosets",
        "part_artifact",
        "covering_part_references",
        "range_pairwise_truth",
        "indexed_pairwise_truth",
        "duplicate_indices",
    }
    return bool(
        isinstance(result, dict)
        and set(result) == required
        and _valid_cover(result, source["bounds"])
        and _valid_predicates(result)
    )


def _load_exact_submission():
    path = Path("/app/submission.json")
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
            return None
        value = json.loads(path.read_text(), parse_float=Decimal)
        return value if isinstance(value, dict) and set(value) == {"result"} else None
    except (
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
        DecimalException,
    ):
        return None


def main():
    submission = _load_exact_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok
        and workspace_input_is_bound()
        and _valid_result(submission.get("result"), source)
    )
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
