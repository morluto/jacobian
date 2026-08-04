"""Independent finite difference-set replay using only the standard library.

This module imports neither the combinatorics producer nor its helper functions.
Only passive artifact-bound JSON crosses the checker boundary.
"""

from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_META_INTEGER = {
    "exactness": "EXACT_INTEGER",
    "determinism": "DETERMINISTIC",
    "backend": "python-stdlib",
    "verification": "UNVERIFIED",
}
_META_FINITE = {
    "exactness": "EXACT_FINITE",
    "determinism": "DETERMINISTIC",
    "backend": "python-stdlib",
    "verification": "UNVERIFIED",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str, *, exhaustive: bool = False) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE" if exhaustive else "DIRECT_WITNESS",
        "coverage": "EXHAUSTIVE" if exhaustive else "NOT_APPLICABLE",
        "detail": detail,
    }


def _canonical_integer(value: object, *, max_digits: int) -> int:
    if (
        not isinstance(value, str)
        or len(value.lstrip("-")) > max_digits
        or _INTEGER.fullmatch(value) is None
    ):
        raise ValueError("integer is outside the checker scope")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("integer is not canonical")
    return parsed


def _integer_list(
    value: object,
    *,
    minimum: int,
    maximum: int,
    max_digits: int,
) -> list[int]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError("integer list is outside the checker scope")
    parsed = [_canonical_integer(item, max_digits=max_digits) for item in value]
    if len(parsed) != len(set(parsed)):
        raise ValueError("integer list is not a set")
    return parsed


def _metadata(
    result: dict[str, Any], fields: set[str], expected: dict[str, str]
) -> bool:
    return set(result) == fields | set(expected) and all(
        result.get(key) == value for key, value in expected.items()
    )


def _replay_sidon(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"elements"} or not _metadata(
        result,
        {
            "semantics_version",
            "normalized_elements",
            "ordered_differences",
            "is_sidon",
        },
        _META_INTEGER,
    ):
        return False
    source_values = _integer_list(
        source["elements"], minimum=0, maximum=32, max_digits=128
    )
    normalized = sorted(source_values)
    if result["semantics_version"] != "integer-sidon.ordered-differences.v1" or result[
        "normalized_elements"
    ] != [str(value) for value in normalized]:
        return False
    expected = [
        {
            "minuend": str(left),
            "subtrahend": str(right),
            "difference": str(left - right),
        }
        for left in normalized
        for right in normalized
        if left != right
    ]
    differences = [int(item["difference"]) for item in expected]
    return (
        result["ordered_differences"] == expected
        and type(result["is_sidon"]) is bool
        and result["is_sidon"] == (len(set(differences)) == len(differences))
    )


def _residue_counts(residues: list[int], modulus: int) -> list[int]:
    counts = [0] * modulus
    for left in residues:
        for right in residues:
            if left != right:
                counts[(left - right) % modulus] += 1
    return counts


def _is_perfect(residues: list[int], modulus: int) -> bool:
    if len(residues) != len(set(residues)):
        return False
    order = len(residues)
    if modulus != order * (order - 1) + 1:
        return False
    counts = _residue_counts(residues, modulus)
    return counts[0] == 0 and all(count == 1 for count in counts[1:])


def _replay_perfect(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"modulus", "residues"} or not _metadata(
        result,
        {
            "semantics_version",
            "modulus",
            "normalized_residues",
            "order",
            "expected_modulus",
            "difference_multiplicities",
            "missing_residues",
            "repeated_residues",
            "is_perfect",
        },
        _META_FINITE,
    ):
        return False
    modulus = source["modulus"]
    raw_residues = source["residues"]
    if (
        type(modulus) is not int
        or not 2 <= modulus <= 4_096
        or not isinstance(raw_residues, list)
        or not 1 <= len(raw_residues) <= 64
        or any(
            type(value) is not int or not 0 <= value < modulus for value in raw_residues
        )
        or len(raw_residues) != len(set(raw_residues))
    ):
        return False
    residues = sorted(raw_residues)
    counts = _residue_counts(residues, modulus)
    profile = [
        {"residue": residue, "multiplicity": counts[residue]}
        for residue in range(1, modulus)
    ]
    missing = [residue for residue in range(1, modulus) if counts[residue] == 0]
    repeated = [residue for residue in range(1, modulus) if counts[residue] > 1]
    order = len(residues)
    expected_modulus = order * (order - 1) + 1
    return result == {
        "semantics_version": "cyclic-perfect-difference-set.v1",
        "modulus": modulus,
        "normalized_residues": residues,
        "order": order,
        "expected_modulus": expected_modulus,
        "difference_multiplicities": profile,
        "missing_residues": missing,
        "repeated_residues": repeated,
        "is_perfect": modulus == expected_modulus and not missing and not repeated,
        **_META_FINITE,
    }


def _extension_scope(source: dict[str, Any]) -> tuple[list[int], int, int, int]:
    if set(source) != {"base_elements", "target_order"}:
        raise ValueError("extension request fields are malformed")
    base_elements = _integer_list(
        source["base_elements"], minimum=1, maximum=64, max_digits=128
    )
    order = source["target_order"]
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("extension order is outside the checker scope")
    modulus = order * (order - 1) + 1
    if modulus > 4_096:
        raise ValueError("extension modulus is outside the checker scope")
    base = sorted({value % modulus for value in base_elements})
    additional = order - len(base)
    if additional < 0 or additional > 3:
        raise ValueError("extension cardinality is outside the checker scope")
    candidate_count = math.comb(modulus - len(base), additional)
    if candidate_count > 50_000:
        raise ValueError("extension candidate space is outside the checker scope")
    return base, order, modulus, candidate_count


def _replay_extension(
    source: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, bool]:
    if not _metadata(
        result,
        {
            "semantics_version",
            "target_order",
            "modulus",
            "base_residues",
            "candidate_space_size",
            "decision",
            "extension",
            "coverage",
        },
        _META_FINITE,
    ):
        return False, False
    base, order, modulus, candidate_count = _extension_scope(source)
    if (
        result["semantics_version"] != "cyclic-pds-extension.fixed-order.v1"
        or result["target_order"] != order
        or result["modulus"] != modulus
        or result["base_residues"] != base
        or result["candidate_space_size"] != candidate_count
    ):
        return False, False
    if result["decision"] == "EXTENDS":
        extension = result["extension"]
        if (
            result["coverage"] != "WITNESS"
            or not isinstance(extension, list)
            or extension != sorted(set(extension))
            or len(extension) != order
            or any(
                type(value) is not int or not 0 <= value < modulus
                for value in extension
            )
            or not set(base) <= set(extension)
        ):
            return False, False
        return _is_perfect(extension, modulus), False
    if (
        result["decision"] != "DOES_NOT_EXTEND"
        or result["coverage"] != "ALL_CANDIDATES"
        or result["extension"] != []
    ):
        return False, False
    pool = [value for value in range(modulus) if value not in set(base)]
    additional = order - len(base)
    for extra in combinations(pool, additional):
        if _is_perfect(sorted((*base, *extra)), modulus):
            return False, True
    return True, True


def _run_simple(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay: Any,
) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject("declared result does not match independent finite replay")
        return _accept(f"independent finite replay accepted {operation_id}")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def check_integer_sidon(request: object) -> dict[str, Any]:
    return _run_simple(
        request,
        operation_id="combinatorics.integer_set.sidon.decide",
        witness_format="combinatorics.integer-sidon.ordered-difference-replay",
        replay=_replay_sidon,
    )


def check_cyclic_perfect_difference_set(request: object) -> dict[str, Any]:
    return _run_simple(
        request,
        operation_id="combinatorics.cyclic_difference_set.perfect.decide",
        witness_format="combinatorics.cyclic-pds.residue-profile-replay",
        replay=_replay_perfect,
    )


def check_cyclic_difference_set_extension(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="combinatorics.cyclic_difference_set.extension.decide",
            witness_format="combinatorics.cyclic-pds-extension.exhaustive-replay",
        )
        accepted, exhaustive = _replay_extension(source, result)
        if not accepted:
            return _reject("declared extension decision failed independent replay")
        return _accept(
            "independent fixed-order PDS extension replay accepted the decision",
            exhaustive=exhaustive,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = [
    "check_cyclic_difference_set_extension",
    "check_cyclic_perfect_difference_set",
    "check_integer_sidon",
]
