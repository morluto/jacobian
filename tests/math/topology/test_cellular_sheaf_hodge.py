"""Exact Hodge operators for bounded rational cellular sheaves."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

import jacobian.math.topology.cellular_sheaves.hodge as hodge_module
from jacobian._exact import CanonicalRational
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

if TYPE_CHECKING:
    from jacobian.math.topology._models import FiniteSimplicialComplex
    from jacobian.math.topology.cellular_sheaves import FiniteCellularSheaf


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_sheaf(
    complex_: FiniteSimplicialComplex,
    scalar: CanonicalRational | int | None = None,
    *,
    field: SheafField = SheafField.RATIONAL,
    prime: int | None = None,
) -> FiniteCellularSheaf:
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


def _fraction(value: CanonicalRational | int) -> Fraction:
    if isinstance(value, CanonicalRational):
        return value.as_fraction()
    return Fraction(value)


def _canonical_rational(value: CanonicalRational | int) -> CanonicalRational:
    assert isinstance(value, CanonicalRational)
    return value


def _matrix(
    entries: tuple[tuple[CanonicalRational | int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(_fraction(value) for value in row) for row in entries)


def test_interval_hodge_matrices_and_harmonics_are_exact() -> None:
    sheaf = _constant_sheaf(canonical_complex(("a", "b"), (("a", "b"),)))

    result = hodge_laplacians(sheaf)

    assert tuple(coord.simplex for coord in result.cochain_bases[0]) == (
        ("a",),
        ("b",),
    )
    assert _matrix(result.up_laplacians[0]) == (
        (Fraction(1), Fraction(-1)),
        (Fraction(-1), Fraction(1)),
    )
    assert _matrix(result.down_laplacians[0]) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert _matrix(result.up_laplacians[1]) == ((Fraction(0),),)
    assert _matrix(result.down_laplacians[1]) == ((Fraction(2),),)
    assert _matrix(result.laplacians[0]) == (
        (Fraction(1), Fraction(-1)),
        (Fraction(-1), Fraction(1)),
    )
    assert _matrix(result.laplacians[1]) == ((Fraction(2),),)
    assert len(result.harmonic_bases[0]) == 1
    assert len(result.harmonic_bases[1]) == 0
    for degree, basis in enumerate(result.harmonic_bases):
        laplacian = _matrix(result.laplacians[degree])
        assert all(
            sum(laplacian[i][j] * _fraction(vector[j]) for j in range(len(vector))) == 0
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

    with pytest.raises(OperationDomainValidationError) as exc_info:
        hodge_laplacians(sheaf)
    assert (
        exc_info.value.errors()[0]["type"]
        == "topology.cellular_sheaf.hodge.characteristic_zero_required"
    )


def test_hodge_work_is_admitted_before_cohomology_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sheaf = _constant_sheaf(canonical_complex(("a", "b"), (("a", "b"),)))
    monkeypatch.setattr(hodge_module, "_MAX_HODGE_MATRIX_CELLS", 4)

    def forbidden(_sheaf: FiniteCellularSheaf) -> None:
        raise AssertionError("cohomology expansion ran before Hodge admission")

    monkeypatch.setattr(hodge_module, "sheaf_cohomology", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        hodge_laplacians(sheaf)


def test_hodge_admission_ignores_unused_derived_composites() -> None:
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in cells
        for face in cells
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    large = _q(10**40)
    source = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(SheafStalk(simplex=cell, basis=("e",)) for cell in cells),
        tuple(
            CoverRestrictionMatrix(source=face, target=coface, entries=((large,),))
            for face, coface in covers
        ),
    ).sheaf
    assert source is not None
    assert (
        max(
            len(str(_canonical_rational(value).num))
            for item in source.derived_restrictions
            for value in item.entries[0]
        )
        > 64
    )
    result = hodge_laplacians(source)
    assert result.laplacians
