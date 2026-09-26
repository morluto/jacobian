from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical import (
    TropicalHypersurface,
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    tropical_bivariate_hypersurface,
)
from jacobian.math.polynomials.tropical._tools import TOOLS


def _polynomial(
    convention: str,
    terms: tuple[tuple[tuple[int, int], int | Fraction], ...],
) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base="QQ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x", "y"),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponents,
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_fraction(Fraction(coefficient)),
                ),
            )
            for exponents, coefficient in sorted(terms)
        ),
    )


def _vector(vector: tuple[CanonicalRational, ...]) -> tuple[Fraction, ...]:
    return tuple(component.as_fraction() for component in vector)


def test_min_plus_tropical_line_has_one_vertex_and_three_weight_one_rays() -> None:
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((1, 0), 0), ((0, 1), 0)),
    )
    result = tropical_bivariate_hypersurface(polynomial)

    vertex = next(cell for cell in result.cells if cell.dimension == 0)
    rays = tuple(cell for cell in result.cells if cell.dimension == 1)
    assert _vector(vertex.generators.points[0]) == (Fraction(0), Fraction(0))
    assert vertex.active_term_indices == (0, 1, 2)
    assert len(rays) == 3
    assert {
        (cell.active_term_indices, _vector(cell.generators.rays[0]), cell.weight)
        for cell in rays
    } == {
        ((0, 1), (Fraction(1), Fraction(0)), 1),
        ((0, 2), (Fraction(0), Fraction(1)), 1),
        ((1, 2), (Fraction(-1), Fraction(-1)), 1),
    }
    assert all(cell.incident_cell_ids == (vertex.cell_id,) for cell in rays)
    assert set(vertex.incident_cell_ids) == {cell.cell_id for cell in rays}
    assert all(
        cell.dual_face_id
        in {support.face_id for support in result.subdivision.face_supports}
        for cell in result.cells
    )
    assert TropicalHypersurface.model_validate_json(result.model_dump_json()) == result


def test_square_curve_has_a_bounded_segment_dual_to_the_diagonal() -> None:
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)),
    )
    result = tropical_bivariate_hypersurface(polynomial)
    segments = [
        cell
        for cell in result.cells
        if cell.dimension == 1 and not cell.generators.rays
    ]

    assert len(segments) == 1
    segment = segments[0]
    assert segment.weight == 1
    assert segment.active_term_indices == (1, 2)
    assert {_vector(point) for point in segment.generators.points} == {
        (Fraction(-1), Fraction(-1)),
        (Fraction(0), Fraction(0)),
    }
    assert len(segment.incident_cell_ids) == 2
    assert all(not cell.generators.lineality for cell in result.cells)

    nonincident_ray = next(
        cell
        for cell in result.cells
        if cell.dimension == 1 and cell.active_term_indices == (1, 3)
    )
    first_triangle_vertex = next(
        cell
        for cell in result.cells
        if cell.dimension == 0 and cell.active_term_indices == (0, 1, 2)
    )
    assert set(nonincident_ray.active_term_indices).intersection(
        first_triangle_vertex.active_term_indices
    ) == {1}
    assert first_triangle_vertex.cell_id not in nonincident_ray.incident_cell_ids
    assert nonincident_ray.cell_id not in first_triangle_vertex.incident_cell_ids


def test_inactive_interior_term_does_not_become_corner_provenance() -> None:
    base = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((0, 2), 0), ((2, 0), 0), ((2, 2), 0)),
    )
    inactive = _polynomial(
        "MIN_PLUS",
        (
            ((0, 0), 0),
            ((0, 2), 0),
            ((1, 1), 100),
            ((2, 0), 0),
            ((2, 2), 0),
        ),
    )
    result = tropical_bivariate_hypersurface(inactive)

    assert all(2 not in cell.active_term_indices for cell in result.cells)
    assert tuple(
        (cell.dimension, cell.weight, len(cell.generators.rays))
        for cell in result.cells
    ) == tuple(
        (cell.dimension, cell.weight, len(cell.generators.rays))
        for cell in tropical_bivariate_hypersurface(base).cells
    )


def test_affine_lift_degeneracy_keeps_full_tie_and_three_rays() -> None:
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((0, 1), 1), ((1, 0), 2)),
    )
    result = tropical_bivariate_hypersurface(polynomial)
    vertex = next(cell for cell in result.cells if cell.dimension == 0)

    assert vertex.active_term_indices == (0, 1, 2)
    assert len([cell for cell in result.cells if cell.dimension == 1]) == 3


def test_max_plus_uses_upper_corner_regions() -> None:
    polynomial = _polynomial(
        "MAX_PLUS",
        (((0, 0), 0), ((1, 0), 0), ((0, 1), 0)),
    )
    result = tropical_bivariate_hypersurface(polynomial)
    rays = [cell for cell in result.cells if cell.dimension == 1]
    assert len(rays) == 3
    assert all(cell.weight == 1 for cell in rays)
    assert {_vector(cell.generators.rays[0]) for cell in rays} == {
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
        (Fraction(1), Fraction(1)),
    }


def test_scaled_lattice_dual_edges_report_weight_two() -> None:
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((2, 0), 0), ((0, 2), 0)),
    )
    result = tropical_bivariate_hypersurface(polynomial)

    assert [cell.weight for cell in result.cells if cell.dimension == 1] == [2, 2, 2]


def test_bivariate_hypersurface_catalog_operation_returns_typed_value() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "tropical.polynomial.bivariate_hypersurface.compute"
    )
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((1, 0), 0), ((0, 1), 0)),
    )
    result = tool.run(tool.request_type(polynomial=polynomial))
    assert isinstance(result, TropicalHypersurface)
    assert len(result.cells) == 4


def test_lower_dimensional_support_is_rejected() -> None:
    polynomial = _polynomial(
        "MIN_PLUS",
        (((0, 0), 0), ((1, 0), 1), ((2, 0), 0)),
    )
    with pytest.raises(OperationDomainValidationError, match="affine dimension two"):
        tropical_bivariate_hypersurface(polynomial)


def test_term_count_admission_happens_before_lifted_hull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.tropical.hypersurface as hypersurface

    polynomial = _polynomial(
        "MIN_PLUS",
        tuple([((index, 0), 0) for index in range(10)] + [((0, 1), 0)]),
    )
    monkeypatch.setattr(
        hypersurface,
        "_regular_subdivision_from_admission",
        lambda *_args, **_kwargs: pytest.fail("lifted hull entered before admission"),
    )

    with pytest.raises(OperationResourceAdmissionError, match="at most 10"):
        tropical_bivariate_hypersurface(polynomial)


def test_ten_term_boundary_is_accepted() -> None:
    polynomial = _polynomial(
        "MIN_PLUS", tuple(((index, index * index), 0) for index in range(10))
    )

    result = tropical_bivariate_hypersurface(polynomial)
    assert len(result.cells) <= 40


def test_polynomial_value_rejects_duplicate_exponent_support() -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    scalar = TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(0, 1),
    )
    with pytest.raises(ValidationError, match="unique exponents"):
        TropicalPolynomial(
            semiring=semiring,
            variables=("x", "y"),
            terms=(
                TropicalPolynomialTerm(exponents=(0, 0), coefficient=scalar),
                TropicalPolynomialTerm(exponents=(0, 0), coefficient=scalar),
                TropicalPolynomialTerm(exponents=(1, 1), coefficient=scalar),
            ),
        )
