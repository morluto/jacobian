"""Exact constant cellular-sheaf construction and composition tests."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafField,
    sections,
)
from jacobian.math.topology.cellular_sheaves._models import (
    SheafCohomologyResult,
    SheafScalar,
)
from jacobian.math.topology.cellular_sheaves.constants import (
    ConstantSheafRequest,
    constant_sheaf,
)
from jacobian.math.topology.cellular_sheaves.constants._tools import TOOLS
from jacobian.math.topology.cellular_sheaves.operations import sheaf_cohomology
from jacobian.math.topology.cohomology.operations import simplicial_cohomology

OPERATION_ID = "cellular_sheaf.constant.compute"

_INTERVAL = canonical_complex(("a", "b"), (("a", "b"),))
_CIRCLE = canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
_TRIANGLE = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
_POINT = canonical_complex(("x",), (("x",),))


def _bettis(result: SheafCohomologyResult) -> tuple[int, ...]:
    return tuple(group.betti_number for group in result.groups)


def _rank_over_qq(matrix: tuple[tuple[SheafScalar, ...], ...]) -> int:
    rows = [
        [
            Fraction(value.num, value.den)
            if isinstance(value, CanonicalRational)
            else Fraction(value)
            for value in row
        ]
        for row in matrix
    ]
    rank = 0
    column_count = len(rows[0]) if rows else 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or rows[row][column] == 0:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def test_constant_interval_preserves_copied_axes_and_composes() -> None:
    sheaf = constant_sheaf(ConstantSheafRequest(complex=_INTERVAL, basis=("u", "v")))

    assert tuple(stalk.simplex for stalk in sheaf.stalks) == (
        ("a",),
        ("b",),
        ("a", "b"),
    )
    assert all(stalk.basis == ("u", "v") for stalk in sheaf.stalks)
    assert len(sheaf.cover_restrictions) == 2
    assert not sheaf.derived_restrictions
    for restriction in sheaf.cover_restrictions:
        assert restriction.row_basis == restriction.column_basis == ("u", "v")
        assert restriction.entries == (
            (CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
            (CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=1)),
        )

    section_space = sections(sheaf)
    assert section_space.dimension == 2
    result = sheaf_cohomology(sheaf)
    assert result.cochain_dimensions == (4, 2)
    assert _bettis(result) == (2, 0)
    differential = result.coboundary_matrices[0]
    assert len(differential) == 2 and all(len(row) == 4 for row in differential)
    assert _rank_over_qq(differential) == 2

    round_trip = FiniteCellularSheaf.model_validate_json(sheaf.model_dump_json())
    assert round_trip == sheaf


@pytest.mark.parametrize(
    ("complex_", "expected"),
    ((_POINT, (1,)), (_INTERVAL, (1, 0)), (_CIRCLE, (1, 1)), (_TRIANGLE, (1, 0, 0))),
)
def test_rank_one_prime_field_constant_sheaf_matches_simplicial_cohomology(
    complex_: FiniteSimplicialComplex, expected: tuple[int, ...]
) -> None:
    prime = 5
    sheaf = constant_sheaf(
        ConstantSheafRequest(
            complex=complex_,
            coefficient_field=SheafField.PRIME_FIELD,
            prime=prime,
            basis=("x",),
        )
    )

    assert _bettis(sheaf_cohomology(sheaf)) == expected
    assert (
        tuple(
            group.betti_number
            for group in simplicial_cohomology(complex_, prime).groups
        )
        == expected
    )
    assert sections(sheaf).dimension == 1


@pytest.mark.parametrize("complex_", (_POINT, _INTERVAL))
def test_zero_vector_space_has_empty_stalks_and_zero_cohomology(
    complex_: FiniteSimplicialComplex,
) -> None:
    sheaf = constant_sheaf(ConstantSheafRequest(complex=complex_, basis=()))

    assert all(stalk.basis == () for stalk in sheaf.stalks)
    assert all(restriction.entries == () for restriction in sheaf.cover_restrictions)
    assert all(restriction.entries == () for restriction in sheaf.derived_restrictions)
    assert sections(sheaf).dimension == 0
    assert all(betti == 0 for betti in _bettis(sheaf_cohomology(sheaf)))


def test_rank_over_stalk_bound_is_admitted_before_restriction_materialization() -> None:
    six_simplex = canonical_complex(
        tuple(f"v{index}" for index in range(7)),
        (tuple(f"v{index}" for index in range(7)),),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        constant_sheaf(ConstantSheafRequest(complex=six_simplex))

    assert error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.constant.simplex_bound"
    )


def test_constant_sheaf_accepts_large_bounded_face_poset() -> None:
    five_simplex = canonical_complex(
        tuple(f"v{index}" for index in range(6)),
        (tuple(f"v{index}" for index in range(6)),),
    )

    sheaf = constant_sheaf(
        ConstantSheafRequest(complex=five_simplex, basis=("a", "b", "c", "d", "e"))
    )

    assert len(sheaf.stalks) == 63
    assert sheaf.comparable_pairs == 602
    assert len(sheaf.derived_restrictions) == 416
    assert len(sheaf.model_dump_json().encode("utf-8")) < 8_000_000


def test_constant_sheaf_manifest_publishes_one_canonical_result() -> None:
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
    result = tool.run(
        tool.request_type.model_validate_json(
            json.dumps(tool.examples[0].input), strict=True
        )
    )
    assert isinstance(result, FiniteCellularSheaf)
    assert len(TOOLS) == 1


def test_constant_sheaf_request_requires_unique_ordered_basis_ids() -> None:
    with pytest.raises(ValidationError, match="basis identifiers must be unique"):
        ConstantSheafRequest(complex=_POINT, basis=("x", "x"))
