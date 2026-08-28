"""Contract and exactness tests for rank-3 chirotope validation (#1767)."""

# ruff: noqa: SIM905

from __future__ import annotations

from itertools import combinations
from typing import Any, cast

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.matroids.oriented._models import (
    MAX_B2_EXCHANGE_INSTANCES,
    MAX_EXECUTION_B2_EXCHANGE_INSTANCES,
    ChirotopeCheckRequest,
    ChirotopeCheckResult,
    ChirotopeCheckStatus,
    UniformRank3Chirotope,
)
from jacobian.math.combinatorics.matroids.oriented._operations import (
    _alternating_value,
    check_chirotope,
)
from jacobian.math.combinatorics.matroids.oriented._tools import TOOLS

type Triple = tuple[int, int, int]


def _increasing_triples(ground_size: int) -> tuple[Triple, ...]:
    return tuple(combinations(range(ground_size), 3))


def _table_from_signs(ground_size: int, signs: dict[Triple, int]) -> dict[str, Any]:
    return {
        "ground_size": ground_size,
        "entries": [
            {"triple": triple, "sign": signs[triple]}
            for triple in _increasing_triples(ground_size)
        ],
    }


def _alternating_table(ground_size: int) -> dict[str, Any]:
    return _table_from_signs(
        ground_size,
        dict.fromkeys(_increasing_triples(ground_size), 1),
    )


def _ringel_table() -> dict[str, Any]:
    """Independent transcription of Atlas's two-projection Ringel fixture."""

    # Convention/known-answer fixture only: it does not identify the historical
    # pictured object or assert realizability. Immutable upstream source:
    # https://raw.githubusercontent.com/techno-optimist/erdos-frontier-atlas/04e44966932bf8093553eb79788652fa6cba71a1/certificates/ringel-nonstretchability/fidelity/chirotope_axioms.py
    # Revision 04e44966932bf8093553eb79788652fa6cba71a1; SHA-256
    # 939079eda82bbfb79f0e1a6683408f3579e69420139d4a1205dd3502cd565bc9.

    z_positive = (
        "127 134 135 136 137 138 145 146 147 148 157 167 168 234 235 236 "
        "237 238 245 246 247 248 256 257 258 267 268 345 346 347 348 357 367 "
        "368 567 568"
    ).split()
    z_negative = (
        "123 124 125 126 128 156 158 178 278 356 358 378 456 457 458 467 468 "
        "478 578 678"
    ).split()
    s_positive = (
        "081 082 083 085 086 084 012 013 015 016 014 023 025 026 024 035 036 "
        "034 056 812 815 835 135 136 134 235 236 234 256"
    ).split()
    s_negative = (
        "054 064 813 816 814 823 825 826 824 836 834 856 854 864 123 125 126 "
        "124 156 154 164 254 264 356 354 364 564"
    ).split()

    def permutation_sign(triple: Triple) -> int:
        inversions = sum(
            triple[left] > triple[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        return -1 if inversions % 2 else 1

    def parse(positive: list[str], negative: list[str]) -> dict[Triple, int]:
        signs: dict[Triple, int] = {}
        for raw, sign in [(item, 1) for item in positive] + [
            (item, -1) for item in negative
        ]:
            triple = cast(Triple, tuple(int(value) for value in raw))
            signs[cast(Triple, tuple(sorted(triple)))] = sign * permutation_sign(triple)
        return signs

    first_projection = parse(z_positive, z_negative)
    second_projection = {
        triple: sign * (-1 if 8 in triple else 1)
        for triple, sign in parse(s_positive, s_negative).items()
    }
    signs = first_projection | second_projection
    for index in (1, 2, 3, 4, 5, 6, 8):
        signs[cast(Triple, tuple(sorted((0, index, 7))))] = 1
    assert len(signs) == 84
    return _table_from_signs(9, signs)


def _reorient(table: dict[str, Any], reoriented: set[int]) -> dict[str, Any]:
    signs = {tuple(entry["triple"]): entry["sign"] for entry in table["entries"]}
    return _table_from_signs(
        table["ground_size"],
        {
            triple: sign * (-1 if len(set(triple) & reoriented) % 2 else 1)
            for triple, sign in signs.items()
        },
    )


def _relabel(table: dict[str, Any], permutation: tuple[int, ...]) -> dict[str, Any]:
    old_signs = {tuple(entry["triple"]): entry["sign"] for entry in table["entries"]}
    inverse = {new: old for old, new in enumerate(permutation)}
    old_values = old_signs
    return _table_from_signs(
        table["ground_size"],
        {
            triple: _alternating_value(
                old_values, cast(Triple, tuple(inverse[index] for index in triple))
            )
            for triple in _increasing_triples(table["ground_size"])
        },
    )


class TestCanonicalUniformRank3Table:
    def test_rejects_zero_sign_before_axiom_enumeration(self) -> None:
        table = _alternating_table(4)
        table["entries"][0]["sign"] = 0
        with pytest.raises(ValidationError) as exc_info:
            ChirotopeCheckRequest.model_validate({"chirotope": table})
        assert exc_info.value.errors()[0]["type"] == "literal_error"

    def test_rejects_missing_or_reordered_triples(self) -> None:
        table = _alternating_table(4)
        table["entries"][1], table["entries"][2] = (
            table["entries"][2],
            table["entries"][1],
        )
        with pytest.raises(ValidationError) as exc_info:
            UniformRank3Chirotope.model_validate(table)
        assert (
            exc_info.value.errors()[0]["type"]
            == "oriented_matroid.canonical_table.entries"
        )

    def test_b2_budget_admits_nine_and_ten_but_rejects_eleven(self) -> None:
        assert (
            UniformRank3Chirotope.model_validate(_alternating_table(10)).ground_size
            == 10
        )
        oversized = _alternating_table(11)
        with pytest.raises(ValidationError) as exc_info:
            UniformRank3Chirotope.model_validate(oversized)
        assert exc_info.value.errors()[0]["type"] == "less_than_equal"


class TestChirotopeCheck:
    def test_alternating_rank3_fixture_has_exact_counts(self) -> None:
        result = check_chirotope(
            ChirotopeCheckRequest.model_validate({"chirotope": _alternating_table(4)})
        )
        assert result.status is ChirotopeCheckStatus.VALID
        assert result.b2_exchange_instances_checked == 4**6
        assert result.obstruction is None

    def test_ringel_fixture_replays_all_cited_b2_instances(self) -> None:
        result = check_chirotope(
            ChirotopeCheckRequest.model_validate({"chirotope": _ringel_table()})
        )
        assert result.status is ChirotopeCheckStatus.VALID
        assert result.b2_exchange_instances_checked == 9**6 == 531_441
        assert ChirotopeCheckResult.model_validate(result.model_dump()) == result

    def test_one_entry_mutation_returns_deterministic_b2_obstruction(self) -> None:
        table = _ringel_table()
        table["entries"][0]["sign"] *= -1
        request = ChirotopeCheckRequest.model_validate({"chirotope": table})
        first = check_chirotope(request)
        second = check_chirotope(request)
        assert first == second
        assert first.status is ChirotopeCheckStatus.B2_OBSTRUCTION
        assert first.b2_exchange_instances_checked < 9**6
        assert first.obstruction is not None
        assert first.obstruction.kind == "B2"
        assert all(value >= 0 for value in first.obstruction.premise_products)
        assert first.obstruction.conclusion_product < 0
        assert all(
            left * right == product
            for (left, right), product in zip(
                first.obstruction.premise_factors,
                first.obstruction.premise_products,
                strict=True,
            )
        )
        assert (
            first.obstruction.conclusion_factors[0]
            * first.obstruction.conclusion_factors[1]
            == first.obstruction.conclusion_product
        )

    def test_reorientation_and_relabelling_preserve_validity(self) -> None:
        table = _ringel_table()
        reoriented = _reorient(table, {1, 4, 8})
        relabelled = _relabel(reoriented, (8, 4, 7, 0, 5, 2, 1, 3, 6))
        result = check_chirotope(
            ChirotopeCheckRequest.model_validate({"chirotope": relabelled})
        )
        assert result.status is ChirotopeCheckStatus.VALID
        assert result.b2_exchange_instances_checked == 531_441

    def test_boundary_ten_reserves_one_b2_scan(self) -> None:
        result = check_chirotope(
            ChirotopeCheckRequest.model_validate({"chirotope": _alternating_table(10)})
        )
        assert result.status is ChirotopeCheckStatus.VALID
        assert result.b2_exchange_instances_checked == MAX_B2_EXCHANGE_INSTANCES
        assert MAX_EXECUTION_B2_EXCHANGE_INSTANCES == 1_000_000


class TestIndependentOracleAndCatalog:
    def test_exact_vandermonde_determinant_signs_are_valid(self) -> None:
        """A direct determinant formula independently supplies chi(i,j,k)."""

        parameters = (-5, -2, 0, 3, 7, 11)
        signs = {
            (i, j, k): 1
            if (parameters[j] - parameters[i])
            * (parameters[k] - parameters[i])
            * (parameters[k] - parameters[j])
            > 0
            else -1
            for i, j, k in combinations(range(len(parameters)), 3)
        }
        result = check_chirotope(
            ChirotopeCheckRequest.model_validate(
                {"chirotope": _table_from_signs(len(parameters), signs)}
            )
        )
        assert result.status is ChirotopeCheckStatus.VALID

    def test_catalog_example_executes_the_published_contract(self) -> None:
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "oriented_matroid.chirotope.check"
        )
        request = tool.request_type.model_validate(tool.examples[0].input)
        assert tool.run(request).status is ChirotopeCheckStatus.VALID
