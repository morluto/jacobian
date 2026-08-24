"""Exact finite rank-3 chirotope-axiom enumeration."""

from __future__ import annotations

from itertools import product
from typing import Literal, cast

from jacobian.math.oriented_matroids._models import (
    B2Obstruction,
    ChirotopeCheckRequest,
    ChirotopeCheckResult,
    ChirotopeCheckStatus,
    UniformRank3Chirotope,
)

type Triple = tuple[int, int, int]


def _permutation_sign(triple: tuple[int, int, int]) -> int:
    """Return the sign of a distinct triple's sorting permutation."""

    inversions = sum(
        triple[left] > triple[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _alternating_value(
    table: dict[Triple, int],
    triple: Triple,
) -> int:
    """Evaluate the table's alternating extension without any coercion."""

    if len(set(triple)) != 3:
        return 0
    increasing = cast(Triple, tuple(sorted(triple)))
    return _permutation_sign(triple) * table[increasing]


def _table(chirotope: UniformRank3Chirotope) -> dict[Triple, int]:
    return {entry.triple: entry.sign for entry in chirotope.entries}


def _ordered_triples(ground_size: int) -> tuple[Triple, ...]:
    return tuple(
        (first, second, third)
        for first, second, third in product(range(ground_size), repeat=3)
    )


def _result(
    chirotope: UniformRank3Chirotope,
    status: ChirotopeCheckStatus,
    b2_exchange_instances_checked: int,
    obstruction: B2Obstruction | None,
) -> ChirotopeCheckResult:
    return ChirotopeCheckResult.model_construct(
        chirotope=chirotope,
        status=status,
        b2_exchange_instances_checked=b2_exchange_instances_checked,
        obstruction=obstruction,
    )


def _expected_result(request: ChirotopeCheckRequest) -> ChirotopeCheckResult:
    """Exhaustively check the complete rank-3 B2 axiom.

    The materialized request pre-bounds the direct enumeration: at most
    ``10**6`` B2 pairs. The first failed instance is deterministic because the
    loop is lexicographic. Alternation is structural in the canonical request
    table.
    """

    chirotope = request.chirotope
    table = _table(chirotope)
    ordered_triples = _ordered_triples(chirotope.ground_size)
    values = {triple: _alternating_value(table, triple) for triple in ordered_triples}

    b2_checked = 0
    for x in ordered_triples:
        x1, x2, x3 = x
        x_value = values[x]
        for y in ordered_triples:
            b2_checked += 1
            y1, y2, y3 = y
            premises = (
                values[(y1, x2, x3)] * values[(x1, y2, y3)],
                values[(y2, x2, x3)] * values[(y1, x1, y3)],
                values[(y3, x2, x3)] * values[(y1, y2, x1)],
            )
            conclusion = x_value * values[y]
            if all(value >= 0 for value in premises) and conclusion < 0:
                premise_factors = (
                    (values[(y1, x2, x3)], values[(x1, y2, y3)]),
                    (values[(y2, x2, x3)], values[(y1, x1, y3)]),
                    (values[(y3, x2, x3)], values[(y1, y2, x1)]),
                )
                conclusion_factors = (x_value, values[y])
                return _result(
                    chirotope,
                    ChirotopeCheckStatus.B2_OBSTRUCTION,
                    b2_checked,
                    B2Obstruction(
                        x=x,
                        y=y,
                        premise_factors=cast(
                            tuple[
                                tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
                                tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
                                tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
                            ],
                            premise_factors,
                        ),
                        premise_products=premises,
                        conclusion_factors=cast(
                            tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]],
                            conclusion_factors,
                        ),
                        conclusion_product=conclusion,
                    ),
                )

    return _result(
        chirotope,
        ChirotopeCheckStatus.VALID,
        b2_checked,
        None,
    )


def check_chirotope(request: ChirotopeCheckRequest) -> ChirotopeCheckResult:
    """Return a result after producer and source-bound replay scans.

    The reported count describes the first mathematical scan. Constructing the
    result through ``model_validate`` performs the mandatory second replay scan;
    admission reserves both scans before this function is entered.
    """

    return ChirotopeCheckResult.model_validate(_expected_result(request).model_dump())


__all__ = ["check_chirotope"]
