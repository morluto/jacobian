"""Independent standard-library replay for bounded exact finite posets.

This checker deliberately imports neither NetworkX, the poset producer, nor
the public poset contracts. Only passive artifact-bound JSON and the generic
binding parser cross the clean-process checker boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request as _bound_request

_MAX_ELEMENTS = 64
_MAX_RELATIONS = _MAX_ELEMENTS * _MAX_ELEMENTS
_MAX_LINEAR_ELEMENTS = 16
_LABEL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
)
_META = {
    "exactness": "EXACT_FINITE",
    "determinism": "DETERMINISTIC",
    "backend": "networkx",
    "backend_version": "3.6.1",
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


def _accept(operation_id: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": f"independent exact replay accepted {operation_id}",
    }


def _strict_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected a strict integer")
    return value


def _label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 32
        or value[0] not in _LABEL_CHARACTERS - frozenset("_.:-")
        or any(character not in _LABEL_CHARACTERS for character in value)
    ):
        raise ValueError("invalid finite-poset label")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _closure(
    elements: list[str],
    pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    successors: dict[str, set[str]] = {element: set() for element in elements}
    for lower, upper in pairs:
        successors[lower].add(upper)
    for pivot in elements:
        reaching = {element for element in elements if pivot in successors[element]}
        for lower in reaching:
            successors[lower].update(successors[pivot])
    result = {(lower, upper) for lower in elements for upper in successors[lower]}
    if any(lower == upper for lower, upper in result):
        raise ValueError("cyclic relation")
    return result


def _reduction(
    elements: list[str],
    closure: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    return {
        (lower, upper)
        for lower, upper in closure
        if not any(
            (lower, middle) in closure and (middle, upper) in closure
            for middle in elements
            if middle not in {lower, upper}
        )
    }


def _ranks(
    elements: list[str],
    covers: set[tuple[str, str]],
) -> list[dict[str, Any]] | None:
    predecessors: dict[str, set[str]] = {element: set() for element in elements}
    successors: dict[str, set[str]] = {element: set() for element in elements}
    for lower, upper in covers:
        predecessors[upper].add(lower)
        successors[lower].add(upper)
    rank_for: dict[str, int] = {}
    remaining = set(elements)
    while remaining:
        ready = sorted(
            element for element in remaining if predecessors[element].issubset(rank_for)
        )
        if not ready:
            raise ValueError("cyclic cover relation")
        for element in ready:
            parent_ranks = {rank_for[parent] for parent in predecessors[element]}
            if len(parent_ranks) > 1:
                return None
            rank_for[element] = 0 if not parent_ranks else next(iter(parent_ranks)) + 1
            remaining.remove(element)
    if len({rank_for[element] for element in elements if not successors[element]}) > 1:
        return None
    return [{"element": element, "rank": rank_for[element]} for element in elements]


def _expected_poset_from_presentation(source: object) -> dict[str, Any]:
    if not isinstance(source, dict) or set(source) != {
        "elements",
        "relation",
        "interpretation",
        "reflexive_pairs",
    }:
        raise ValueError("finite-poset presentation is malformed")
    raw_elements = source["elements"]
    if not isinstance(raw_elements, list) or len(raw_elements) > _MAX_ELEMENTS:
        raise ValueError("finite-poset carrier is malformed")
    input_elements = [_label(value) for value in raw_elements]
    if len(input_elements) != len(set(input_elements)):
        raise ValueError("finite-poset carrier repeats a label")
    elements = sorted(input_elements)
    carrier = set(elements)
    raw_relation = source["relation"]
    if not isinstance(raw_relation, list) or len(raw_relation) > _MAX_RELATIONS:
        raise ValueError("finite-poset relation is malformed")
    supplied_list: list[tuple[str, str]] = []
    for item in raw_relation:
        if not isinstance(item, dict) or set(item) != {"lower", "upper"}:
            raise ValueError("finite-poset relation pair is malformed")
        lower, upper = _label(item["lower"]), _label(item["upper"])
        if lower not in carrier or upper not in carrier:
            raise ValueError("relation endpoint is outside the carrier")
        supplied_list.append((lower, upper))
    if len(supplied_list) != len(set(supplied_list)):
        raise ValueError("finite-poset relation repeats a pair")
    supplied = set(supplied_list)
    interpretation = source["interpretation"]
    reflexive = source["reflexive_pairs"]
    if interpretation not in {"COVER_EDGES", "COMPARABLE_PAIRS"}:
        raise ValueError("unknown relation interpretation")
    if reflexive not in {"FORBIDDEN", "REQUIRED"}:
        raise ValueError("unknown reflexive policy")
    diagonal = {(element, element) for element in elements}
    if interpretation == "COVER_EDGES":
        if reflexive != "FORBIDDEN" or supplied & diagonal:
            raise ValueError("cover relation has invalid reflexive semantics")
        strict = supplied
    else:
        if reflexive == "FORBIDDEN" and supplied & diagonal:
            raise ValueError("forbidden diagonal is present")
        if reflexive == "REQUIRED" and supplied & diagonal != diagonal:
            raise ValueError("required diagonal is incomplete")
        strict = supplied - diagonal
    if any((upper, lower) in strict for lower, upper in strict):
        raise ValueError("relation is not antisymmetric")
    closure = _closure(elements, strict)
    covers = _reduction(elements, closure)
    if interpretation == "COVER_EDGES" and strict != covers:
        raise ValueError("cover relation is transitively redundant")
    if interpretation == "COMPARABLE_PAIRS" and strict != closure:
        raise ValueError("comparable relation is not transitively complete")
    ranks = _ranks(elements, covers)
    incomparable = [
        {"left": left, "right": right}
        for index, left in enumerate(elements)
        for right in elements[index + 1 :]
        if (left, right) not in closure and (right, left) not in closure
    ]
    payload = {
        "poset_format": "jacobian.finite-poset/v1",
        "elements": elements,
        "strict_order_pairs": [
            {"lower": lower, "upper": upper} for lower, upper in sorted(closure)
        ],
        "cover_relations": [
            {"lower": lower, "upper": upper} for lower, upper in sorted(covers)
        ],
        "incomparable_pairs": incomparable,
        "minimal_elements": [
            element
            for element in elements
            if not any(upper == element for _, upper in closure)
        ],
        "maximal_elements": [
            element
            for element in elements
            if not any(lower == element for lower, _ in closure)
        ],
        "graded": ranks is not None,
        "ranks": ranks,
    }
    return {**payload, "poset_digest": _digest(payload)}


def _parse_poset(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "poset_format",
        "elements",
        "strict_order_pairs",
        "cover_relations",
        "incomparable_pairs",
        "minimal_elements",
        "maximal_elements",
        "graded",
        "ranks",
        "poset_digest",
    }:
        raise ValueError("finite-poset artifact is malformed")
    elements = value["elements"]
    strict_pairs = value["strict_order_pairs"]
    if not isinstance(elements, list) or not isinstance(strict_pairs, list):
        raise ValueError("finite-poset artifact carrier or relation is malformed")
    source = {
        "elements": elements,
        "relation": strict_pairs,
        "interpretation": "COMPARABLE_PAIRS",
        "reflexive_pairs": "FORBIDDEN",
    }
    expected = _expected_poset_from_presentation(source)
    if value != expected:
        raise ValueError("finite-poset artifact is not canonical and complete")
    return expected


def _result_with_meta(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys | set(_META):
        raise ValueError("finite-poset result is malformed")
    if any(value[key] != expected for key, expected in _META.items()):
        raise ValueError("finite-poset result metadata is unsupported")
    return value


def _replay_materialization(source: dict[str, Any], result: dict[str, Any]) -> bool:
    expected_poset = _expected_poset_from_presentation(source)
    expected = {
        **_META,
        "poset": expected_poset,
        "completeness": "COMPLETE_CLOSURE_AND_REDUCTION",
    }
    return result == expected


def _relation_set(poset: dict[str, Any]) -> set[tuple[str, str]]:
    return {(item["lower"], item["upper"]) for item in poset["strict_order_pairs"]}


def _replay_width(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"poset"}:
        return False
    poset = _parse_poset(source["poset"])
    result = _result_with_meta(
        result,
        {
            "poset_digest",
            "width",
            "maximum_antichain",
            "minimum_chain_cover",
            "matching",
            "matching_size",
            "certificate",
        },
    )
    elements = poset["elements"]
    carrier = set(elements)
    relation = _relation_set(poset)
    width = _strict_int(result["width"])
    antichain = result["maximum_antichain"]
    chains = result["minimum_chain_cover"]
    matching = result["matching"]
    if (
        result["poset_digest"] != poset["poset_digest"]
        or result["certificate"] != "DILWORTH_ANTICHAIN_CHAIN_COVER"
        or not 0 <= width <= len(elements)
        or not isinstance(antichain, list)
        or antichain != sorted(set(antichain))
        or any(_label(element) not in carrier for element in antichain)
        or len(antichain) != width
        or any(
            (left, right) in relation or (right, left) in relation
            for index, left in enumerate(antichain)
            for right in antichain[index + 1 :]
        )
        or not isinstance(chains, list)
        or len(chains) != width
    ):
        return False
    flattened: list[str] = []
    transitions: set[tuple[str, str]] = set()
    for chain in chains:
        if not isinstance(chain, dict) or set(chain) != {"elements"}:
            return False
        members = chain["elements"]
        if (
            not isinstance(members, list)
            or not members
            or any(_label(member) not in carrier for member in members)
            or any((lower, upper) not in relation for lower, upper in pairwise(members))
        ):
            return False
        flattened.extend(members)
        transitions.update(pairwise(members))
    if sorted(flattened) != elements or len(flattened) != len(set(flattened)):
        return False
    if not isinstance(matching, list):
        return False
    matching_pairs: list[tuple[str, str]] = []
    for edge in matching:
        if not isinstance(edge, dict) or set(edge) != {"left", "right"}:
            return False
        pair = (_label(edge["left"]), _label(edge["right"]))
        if pair not in relation:
            return False
        matching_pairs.append(pair)
    matching_size = _strict_int(result["matching_size"])
    return (
        matching_pairs == sorted(set(matching_pairs))
        and len({left for left, _ in matching_pairs}) == len(matching_pairs)
        and len({right for _, right in matching_pairs}) == len(matching_pairs)
        and set(matching_pairs) == transitions
        and matching_size == len(matching_pairs)
        and matching_size + width == len(elements)
    )


def _linear_state_table(poset: dict[str, Any]) -> list[dict[str, Any]]:
    elements = poset["elements"]
    if len(elements) > _MAX_LINEAR_ELEMENTS:
        raise ValueError("linear-extension checker scope exceeded")
    index = {element: position for position, element in enumerate(elements)}
    predecessors = [0] * len(elements)
    successors = [0] * len(elements)
    for lower, upper in _relation_set(poset):
        predecessors[index[upper]] |= 1 << index[lower]
        successors[index[lower]] |= 1 << index[upper]
    counts = {0: 1}
    states: list[dict[str, Any]] = [
        {
            "ideal_mask": 0,
            "cardinality": 0,
            "removable_maximal_elements": [],
            "count": 1,
        }
    ]
    for mask in range(1, 1 << len(elements)):
        if any(
            mask & (1 << position)
            and predecessors[position] & mask != predecessors[position]
            for position in range(len(elements))
        ):
            continue
        removable = [
            elements[position]
            for position in range(len(elements))
            if mask & (1 << position) and successors[position] & mask == 0
        ]
        count = sum(counts[mask ^ (1 << index[element])] for element in removable)
        counts[mask] = count
        states.append(
            {
                "ideal_mask": mask,
                "cardinality": mask.bit_count(),
                "removable_maximal_elements": removable,
                "count": count,
            }
        )
    return states


def _replay_linear_extensions(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if set(source) != {"poset"}:
        return False
    poset = _parse_poset(source["poset"])
    states = _linear_state_table(poset)
    full_mask = (1 << len(poset["elements"])) - 1
    expected = {
        **_META,
        "poset_digest": poset["poset_digest"],
        "element_order": poset["elements"],
        "count": states[-1]["count"] if states[-1]["ideal_mask"] == full_mask else None,
        "states": states,
        "state_count": len(states),
        "explored_subset_count": 1 << len(poset["elements"]),
        "memo_digest": _digest(states),
        "state_scope": "ALL_ORDER_IDEALS",
        "completeness": "COMPLETE",
    }
    return result == expected


def _topological_order(
    elements: list[str],
    relation: set[tuple[str, str]],
) -> list[str]:
    predecessors = {
        element: {lower for lower, upper in relation if upper == element}
        for element in elements
    }
    order: list[str] = []
    remaining = set(elements)
    while remaining:
        ready = sorted(
            element
            for element in remaining
            if predecessors[element].isdisjoint(remaining)
        )
        if not ready:
            raise ValueError("cyclic finite poset")
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def _mobius_table(
    poset: dict[str, Any],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], list[dict[str, Any]]]]:
    relation = _relation_set(poset)
    order = _topological_order(poset["elements"], relation)
    mu: dict[tuple[str, str], int] = {}
    ledgers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for lower_index, lower in enumerate(order):
        mu[(lower, lower)] = 1
        ledgers[(lower, lower)] = []
        for upper_index in range(lower_index + 1, len(order)):
            upper = order[upper_index]
            if (lower, upper) not in relation:
                continue
            term_pairs = sorted(
                (middle, mu[(lower, middle)])
                for middle in order[:upper_index]
                if middle == lower
                or ((lower, middle) in relation and (middle, upper) in relation)
            )
            terms = [
                {"intermediate": middle, "value": value} for middle, value in term_pairs
            ]
            mu[(lower, upper)] = -sum(value for _, value in term_pairs)
            ledgers[(lower, upper)] = terms
    return mu, ledgers


def _replay_mobius(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"poset", "scope", "intervals"}:
        return False
    poset = _parse_poset(source["poset"])
    scope = source["scope"]
    raw_intervals = source["intervals"]
    if scope not in {"COMPLETE_MATRIX", "SELECTED_INTERVALS"}:
        return False
    if not isinstance(raw_intervals, list):
        return False
    selected: list[tuple[str, str]] = []
    for interval in raw_intervals:
        if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
            return False
        selected.append((_label(interval["lower"]), _label(interval["upper"])))
    if len(selected) != len(set(selected)):
        return False
    relation = _relation_set(poset)
    if scope == "COMPLETE_MATRIX":
        if selected:
            return False
        requested = sorted(
            (lower, upper)
            for lower in poset["elements"]
            for upper in poset["elements"]
            if lower == upper or (lower, upper) in relation
        )
        intervals: list[dict[str, str]] = []
    else:
        if not selected or any(
            lower not in poset["elements"]
            or upper not in poset["elements"]
            or (lower != upper and (lower, upper) not in relation)
            for lower, upper in selected
        ):
            return False
        requested = sorted(selected)
        intervals = [{"lower": lower, "upper": upper} for lower, upper in requested]
    mu, ledgers = _mobius_table(poset)
    expected = {
        **_META,
        "poset_digest": poset["poset_digest"],
        "element_order": poset["elements"],
        "scope": scope,
        "intervals": intervals,
        "values": [
            {
                "lower": lower,
                "upper": upper,
                "value": mu[(lower, upper)],
                "recurrence_contributions": ledgers[(lower, upper)],
            }
            for lower, upper in requested
        ],
        "completeness": scope,
        "recurrence_identity": "SUM_LOWER_TO_UPPER_EQUALS_DELTA",
    }
    return result == expected


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    try:
        source, result = _bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject("finite-poset candidate failed independent exact replay")
        return _accept(operation_id)
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
        return _reject("malformed, unsupported, or mismatched finite-poset request")


def check_finite_poset_materialization(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="poset.finite.materialize",
        witness_format="poset.finite.closure-reduction-replay",
        replay=_replay_materialization,
    )


def check_poset_width(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="poset.width.compute",
        witness_format="poset.width.dilworth-dual-replay",
        replay=_replay_width,
    )


def check_linear_extension_count(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="poset.linear_extensions.count",
        witness_format="poset.linear-extensions.complete-ideal-dp-replay",
        replay=_replay_linear_extensions,
    )


def check_poset_mobius_function(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="poset.mobius_function.compute",
        witness_format="poset.mobius.interval-convolution-replay",
        replay=_replay_mobius,
    )


__all__ = [
    "check_finite_poset_materialization",
    "check_linear_extension_count",
    "check_poset_mobius_function",
    "check_poset_width",
]
