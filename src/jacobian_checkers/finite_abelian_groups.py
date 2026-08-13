"""Independent stdlib replay for bounded finite abelian group results."""

from __future__ import annotations

from collections import Counter
from itertools import product
from math import prod
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _valid_factor(value: object, rank: int) -> bool:
    return bool(
        isinstance(value, list)
        and 1 <= len(value) <= 256
        and all(
            isinstance(element, list)
            and len(element) == rank
            and all(
                type(coordinate) is int and abs(coordinate) <= 1_000_000
                for coordinate in element
            )
            for element in value
        )
    )


def _source(
    source: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if set(source) != {"moduli", "left", "right"}:
        raise ValueError("malformed finite-group request")
    raw_moduli = source["moduli"]
    left_source = source["left"]
    right_source = source["right"]
    if not isinstance(raw_moduli, list) or not 1 <= len(raw_moduli) <= 6:
        raise ValueError("malformed finite-group moduli")
    if any(
        type(value) is not int or not 2 <= value <= 1_000_000 for value in raw_moduli
    ):
        raise ValueError("malformed finite-group modulus")
    moduli = tuple(raw_moduli)
    if prod(moduli) > 4_096:
        raise ValueError("finite-group order exceeds checker budget")
    if not _valid_factor(left_source, len(moduli)) or not _valid_factor(
        right_source, len(moduli)
    ):
        raise ValueError("malformed finite-group factor")
    assert isinstance(left_source, list) and isinstance(right_source, list)
    if len(left_source) * len(right_source) > 4_096:
        raise ValueError("factor product exceeds checker budget")

    def normalize(element: list[int]) -> tuple[int, ...]:
        return tuple(
            coordinate % modulus
            for coordinate, modulus in zip(element, moduli, strict=True)
        )

    left = tuple(sorted(normalize(element) for element in left_source))
    right = tuple(sorted(normalize(element) for element in right_source))
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError("finite-group factors contain normalized duplicates")
    return moduli, left, right


def _duplicate_witness(
    element: tuple[int, ...] | None,
    representations: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ],
) -> dict[str, object] | None:
    if element is None:
        return None
    first, second = representations[element][:2]
    return {
        "element": list(element),
        "left": list(first[0]),
        "right": list(first[1]),
        "other_left": list(second[0]),
        "other_right": list(second[1]),
    }


def _expected(
    moduli: tuple[int, ...],
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    representations: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ] = {}
    for left_element in left:
        for right_element in right:
            total = tuple(
                (a + b) % modulus
                for a, b, modulus in zip(
                    left_element, right_element, moduli, strict=True
                )
            )
            representations.setdefault(total, []).append((left_element, right_element))
    group = tuple(product(*(range(modulus) for modulus in moduli)))
    histogram = Counter(len(representations.get(element, ())) for element in group)
    missing = next(
        (element for element in group if element not in representations),
        None,
    )
    duplicate_element = next(
        (element for element in group if len(representations.get(element, ())) > 1),
        None,
    )
    duplicate = _duplicate_witness(duplicate_element, representations)
    order = prod(moduli)
    exact = len(left) * len(right) == order and histogram == {1: order}
    return {
        "moduli": list(moduli),
        "normalized_left": [list(element) for element in left],
        "normalized_right": [list(element) for element in right],
        "group_order": order,
        "pair_count": len(left) * len(right),
        "distinct_sum_count": len(representations),
        "representation_histogram": [
            {"representation_count": count, "element_count": histogram[count]}
            for count in sorted(histogram)
        ],
        "is_exact_factorization": exact,
        "first_missing": None if exact else (list(missing) if missing else None),
        "first_duplicate": None if exact else duplicate,
        "convention": "UNIQUE_SUM_REPRESENTATION_IN_PRODUCT_OF_CYCLIC_GROUPS",
    }


def _strict_json_equal(candidate: object, expected: object) -> bool:
    if type(candidate) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(candidate, dict):
            return False
        return set(candidate) == set(expected) and all(
            _strict_json_equal(candidate[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(candidate, list):
            return False
        return len(candidate) == len(expected) and all(
            _strict_json_equal(candidate_item, expected_item)
            for candidate_item, expected_item in zip(candidate, expected, strict=True)
        )
    return candidate == expected


def check_finite_abelian_group_exact_factorization(
    request: object,
) -> dict[str, Any]:
    operation_id = "finite_abelian_group.exact_factorization.compute"
    try:
        source, result = bound_request(
            request,
            operation_id=operation_id,
            witness_format="finite-abelian-group.exact-factorization.stdlib-replay",
        )
        moduli, left, right = _source(source)
        if not _strict_json_equal(result, _expected(moduli, left, right)):
            return _reject("result does not match exhaustive finite-group replay")
        return _accept(f"independent exhaustive integer replay accepted {operation_id}")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_finite_abelian_group_exact_factorization"]
