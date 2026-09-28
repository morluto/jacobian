"""Independent block-matrix checks for cellular sheaf cochain complexes."""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    from_cover_maps,
    sheaf_cochain_complex,
    sheaf_cohomology,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    SheafCochainComplex,
)
from jacobian.math.topology.cellular_sheaves._tools import TOOLS


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_triangle():
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    stalks = tuple(SheafStalk(simplex=face, basis=("x",)) for face in cells)
    covers = tuple(
        CoverRestrictionMatrix(source=face, target=coface, entries=((_q(1),),))
        for coface in cells
        if len(coface) > 1
        for face in (
            coface[:position] + coface[position + 1 :]
            for position in range(len(coface))
        )
    )
    result = from_cover_maps(complex_, SheafField.RATIONAL, None, stalks, covers)
    assert result.sheaf is not None
    return result.sheaf


def _fractions(matrix):
    return tuple(
        tuple(Fraction(value.num, value.den) for value in row) for row in matrix
    )


def test_triangle_matches_independent_oriented_boundary_blocks() -> None:
    sheaf = _constant_triangle()
    result = sheaf_cochain_complex(sheaf)

    # Independent simplicial incidence oracle in the canonical cell order
    # (a,b,c), (ab,ac,bc), (abc).
    delta_0 = (
        (Fraction(-1), Fraction(1), Fraction(0)),
        (Fraction(-1), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1), Fraction(1)),
    )
    delta_1 = ((Fraction(1), Fraction(-1), Fraction(1)),)
    assert result.sheaf == sheaf
    assert result.cochain_dimensions == (3, 3, 1)
    assert tuple(_fractions(matrix) for matrix in result.coboundary_matrices) == (
        delta_0,
        delta_1,
    )
    assert tuple(
        tuple(
            sum(delta_1[row][middle] * delta_0[middle][column] for middle in range(3))
            for column in range(3)
        )
        for row in range(1)
    ) == ((Fraction(0), Fraction(0), Fraction(0)),)

    # Cohomology consumes the same canonical chain matrices.
    assert sheaf_cohomology(sheaf).coboundary_matrices == result.coboundary_matrices
    assert SheafCochainComplex.model_validate_json(result.model_dump_json()) == result


def test_published_operation_example_runs_without_catalog_boot() -> None:
    tool = next(
        declaration
        for declaration in TOOLS
        if declaration.operation_id == "cellular_sheaf.cochain_complex.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, SheafCochainComplex)
    assert result.cochain_dimensions == (2, 1)
    assert _fractions(result.coboundary_matrices[0]) == ((Fraction(-1), Fraction(1)),)
