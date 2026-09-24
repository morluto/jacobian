"""Exact Hodge operators for bounded rational cellular sheaves."""

from __future__ import annotations

from fractions import Fraction

import pytest

import jacobian.math.topology.cellular_sheaves.hodge as hodge_module
from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    from_cover_maps,
    hodge_laplacians,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_sheaf(complex_, scalar=None, *, field=SheafField.RATIONAL, prime=None):
    if scalar is None:
        scalar = 1 if field is SheafField.PRIME_FIELD else _q(1)
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in cells
        for face in cells
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        field,
        prime,
        tuple(SheafStalk(simplex=cell, basis=("e",)) for cell in cells),
        tuple(
            CoverRestrictionMatrix(source=a, target=b, entries=((scalar,),))
            for a, b in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _matrix(entries):
    return tuple(
        tuple(Fraction(value.num, value.den) for value in row) for row in entries
    )


def test_interval_hodge_matrices_and_harmonics_are_exact() -> None:
    sheaf = _constant_sheaf(canonical_complex(("a", "b"), (("a", "b"),)))

    result = hodge_laplacians(sheaf)

    assert tuple(coord.simplex for coord in result.cochain_bases[0]) == (
        ("a",),
        ("b",),
    )
    assert _matrix(result.up_laplacians[0]) == ((0, 0), (0, 0))
    assert _matrix(result.down_laplacians[0]) == ((1, -1), (-1, 1))
    assert _matrix(result.up_laplacians[1]) == ((2,),)
    assert _matrix(result.down_laplacians[1]) == ((0,),)
    assert _matrix(result.laplacians[0]) == ((1, -1), (-1, 1))
    assert _matrix(result.laplacians[1]) == ((2,),)
    assert len(result.harmonic_bases[0]) == 1
    assert len(result.harmonic_bases[1]) == 0
    for degree, basis in enumerate(result.harmonic_bases):
        laplacian = _matrix(result.laplacians[degree])
        assert all(
            sum(
                laplacian[i][j] * Fraction(vector[j].num, vector[j].den)
                for j in range(len(vector))
            )
            == 0
            for vector in basis
            for i in range(len(vector))
        )


def test_circle_harmonic_dimensions_match_independent_betti_numbers() -> None:
    circle = canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
    result = hodge_laplacians(_constant_sheaf(circle))

    assert tuple(map(len, result.harmonic_bases)) == (1, 1)
    for matrix in result.laplacians:
        assert all(
            _matrix(matrix)[i][j] == _matrix(matrix)[j][i]
            for i in range(len(matrix))
            for j in range(len(matrix))
        )


def test_hodge_requires_characteristic_zero() -> None:
    sheaf = _constant_sheaf(
        canonical_complex(("a",), (("a",),)),
        field=SheafField.PRIME_FIELD,
        prime=5,
    )

    with pytest.raises(OperationDomainValidationError, match="rational coefficient"):
        hodge_laplacians(sheaf)


def test_hodge_operation_is_published() -> None:
    assert "cellular_sheaf.hodge_laplacians.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }


def test_hodge_work_is_admitted_before_cohomology_expansion(monkeypatch) -> None:
    sheaf = _constant_sheaf(canonical_complex(("a", "b"), (("a", "b"),)))
    monkeypatch.setattr(hodge_module, "_MAX_HODGE_MATRIX_CELLS", 4)

    def forbidden(_sheaf):
        raise AssertionError("cohomology expansion ran before Hodge admission")

    monkeypatch.setattr(hodge_module, "sheaf_cohomology", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        hodge_laplacians(sheaf)
