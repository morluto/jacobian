"""Exact direct sums of finite cellular sheaves and their source injections."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    direct_sum,
    from_cover_maps,
    sections,
    sheaf_cohomology,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix
from jacobian.math.topology.cellular_sheaves.direct_sum import SheafDirectSumRequest


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_interval(*, field=SheafField.PRIME_FIELD, prime=2):
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    sheaf = from_cover_maps(
        complex_,
        field,
        prime if field is SheafField.PRIME_FIELD else None,
        tuple(SheafStalk(simplex=cell, basis=("x",)) for cell in cells),
        tuple(
            CoverRestrictionMatrix(
                source=vertex,
                target=("a", "b"),
                entries=((1 if field is SheafField.PRIME_FIELD else _q(1),),),
            )
            for vertex in (("a",), ("b",))
        ),
    ).sheaf
    assert sheaf is not None
    return sheaf


def test_direct_sum_block_maps_and_injections_match_independent_gf2_oracle() -> None:
    source = _constant_interval()
    result = direct_sum(source, source)
    summed = result.direct_sum
    assert [stalk.basis for stalk in summed.stalks] == [("L0", "R0")] * 3
    assert result.stalk_inclusions[0].left == ((1,), (0,))
    assert result.stalk_inclusions[0].right == ((0,), (1,))
    assert all(
        item.entries == ((1, 0), (0, 1))
        for item in summed.cover_restrictions + summed.derived_restrictions
    )

    # Enumerate all 2^6 stalk assignments, checking both components against
    # the original face-to-coface equations independently of Jacobian's kernel.
    compatible = set()
    for assignment in product(range(2), repeat=6):
        a, b, edge = assignment[:2], assignment[2:4], assignment[4:]
        if a == edge and b == edge:
            compatible.add(assignment)
    section = sections(summed)
    generated = {
        tuple(
            sum(
                coeff[i] * section.basis_coordinates[i][j]
                for i in range(section.dimension)
            )
            % 2
            for j in range(6)
        )
        for coeff in product(range(2), repeat=section.dimension)
    }
    assert section.dimension == 2
    assert generated == compatible
    groups = sheaf_cohomology(summed).groups
    assert tuple(group.betti_number for group in groups) == (2, 0)


def test_direct_sum_preserves_rational_exact_maps_and_serialization() -> None:
    source = _constant_interval(field=SheafField.RATIONAL)
    result = direct_sum(source, source)
    assert all(
        item.entries == ((_q("1"), _q("0")), (_q("0"), _q("1")))
        for item in result.direct_sum.cover_restrictions
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_direct_sum_rejects_different_field_or_complex() -> None:
    with pytest.raises(
        OperationDomainValidationError, match="same exact coefficient field"
    ):
        direct_sum(_constant_interval(), _constant_interval(field=SheafField.RATIONAL))
    other = from_cover_maps(
        canonical_complex(("x",), (("x",),)),
        SheafField.PRIME_FIELD,
        2,
        (SheafStalk(simplex=("x",), basis=("x",)),),
        (),
    ).sheaf
    assert other is not None
    with pytest.raises(OperationDomainValidationError, match="same simplicial complex"):
        direct_sum(_constant_interval(), other)


def test_direct_sum_doubles_a_stalk_past_public_rank_bound_before_expansion() -> None:
    source = from_cover_maps(
        canonical_complex(("a",), (("a",),)),
        SheafField.PRIME_FIELD,
        2,
        (SheafStalk(simplex=("a",), basis=tuple(f"x{i}" for i in range(5))),),
        (),
    ).sheaf
    assert source is not None
    with pytest.raises(OperationResourceAdmissionError, match="direct-sum stalk"):
        direct_sum(source, source)


def test_direct_sum_catalog_contract_and_example() -> None:
    tool = Catalog(BUILTIN_TOOLS).operation("cellular_sheaf.direct_sum.compute")
    assert tool is not None
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.direct_sum.stalks[0].basis == ("L0", "R0")
    assert SheafDirectSumRequest.model_validate_json(json.dumps(tool.examples[0].input))
