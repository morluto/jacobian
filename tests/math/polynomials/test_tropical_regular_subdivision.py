from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.tropical import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    regular_subdivision,
    tropical_bivariate_regular_subdivision,
)
from jacobian.math.polynomials.tropical._models import (
    BivariateRegularSubdivisionRequest,
)

_SQUARE_EXPONENTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _polynomial(
    convention: str, coefficients: tuple[Fraction | int, ...]
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
            for exponents, coefficient in zip(
                _SQUARE_EXPONENTS, coefficients, strict=True
            )
        ),
    )


def _supporting_lifted_face_oracle(
    coefficients: tuple[Fraction | int, ...], convention: str
) -> tuple[tuple[int, ...], ...]:
    """Enumerate graph planes through source triples, without a hull backend."""
    points = tuple(
        (Fraction(x), Fraction(y), Fraction(height))
        for (x, y), height in zip(_SQUARE_EXPONENTS, coefficients, strict=True)
    )
    supporting: set[tuple[int, ...]] = set()
    for indices in combinations(range(len(points)), 3):
        first, second, third = (points[index] for index in indices)
        dx1, dy1 = second[0] - first[0], second[1] - first[1]
        dx2, dy2 = third[0] - first[0], third[1] - first[1]
        determinant = dx1 * dy2 - dx2 * dy1
        if not determinant:
            continue
        dz1, dz2 = second[2] - first[2], third[2] - first[2]
        slope_x = (dz1 * dy2 - dz2 * dy1) / determinant
        slope_y = (dx1 * dz2 - dx2 * dz1) / determinant
        intercept = first[2] - slope_x * first[0] - slope_y * first[1]
        residuals = tuple(
            z - slope_x * x - slope_y * y - intercept for x, y, z in points
        )
        supported = (
            all(residual >= 0 for residual in residuals)
            if convention == "MIN_PLUS"
            else all(residual <= 0 for residual in residuals)
        )
        if supported:
            supporting.add(
                tuple(index for index, residual in enumerate(residuals) if not residual)
            )
    return tuple(sorted(supporting))


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_bivariate_lower_and_upper_cells_match_independent_plane_oracle(
    convention: str,
) -> None:
    for coefficients in product((-1, 0, 1), repeat=4):
        subdivision = tropical_bivariate_regular_subdivision(
            _polynomial(convention, coefficients)
        )
        actual = tuple(
            sorted(face.source_term_indices for face in subdivision.lifted_faces)
        )
        assert actual == _supporting_lifted_face_oracle(coefficients, convention)
        assert all(
            (face.normal[2].num < 0) == (convention == "MIN_PLUS")
            for face in subdivision.lifted_faces
        )


def test_min_plus_square_subdivision_has_exact_source_and_diagonal_provenance() -> None:
    subdivision = tropical_bivariate_regular_subdivision(
        _polynomial("MIN_PLUS", (0, 0, 0, 1))
    )
    assert subdivision.cell_complex.f_vector == (1, 4, 5, 2)
    assert tuple(face.source_term_indices for face in subdivision.lifted_faces) == (
        (0, 1, 2),
        (1, 2, 3),
    )
    diagonal = next(
        support
        for support in subdivision.face_supports
        if support.dimension == 1 and support.lifted_face_indices == (0, 1)
    )
    assert diagonal.source_term_indices == (1, 2)
    restored = type(subdivision).model_validate_json(subdivision.model_dump_json())
    assert restored == subdivision


def test_max_plus_selects_the_opposite_regular_subdivision() -> None:
    subdivision = tropical_bivariate_regular_subdivision(
        _polynomial("MAX_PLUS", (0, 0, 0, 1))
    )
    assert tuple(
        sorted(face.source_term_indices for face in subdivision.lifted_faces)
    ) == (
        (0, 1, 3),
        (0, 2, 3),
    )
    diagonal = next(
        support
        for support in subdivision.face_supports
        if support.dimension == 1 and len(support.lifted_face_indices) == 2
    )
    assert diagonal.source_term_indices == (0, 3)


def test_affine_lift_is_one_cell_and_high_interior_term_is_not_lifted_provenance() -> (
    None
):
    affine = tropical_bivariate_regular_subdivision(
        _polynomial("MIN_PLUS", (0, 1, 2, 3))
    )
    assert affine.cell_complex.f_vector == (1, 4, 4, 1)
    assert affine.lifted_faces[0].source_term_indices == (0, 1, 2, 3)

    exponents = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    high_center = TropicalPolynomial(
        semiring=semiring,
        variables=("x", "y"),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponents,
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(height, 1),
                ),
            )
            for exponents, height in zip(exponents, (0, 0, 100, 0, 0), strict=True)
        ),
    )
    inessential = tropical_bivariate_regular_subdivision(high_center)
    assert len(inessential.cell_complex.maximal_cells) == 1
    top_cell = next(face for face in inessential.face_supports if face.dimension == 2)
    assert top_cell.source_term_indices == (0, 1, 3, 4)


def test_rational_lift_heights_remain_exact() -> None:
    coefficients = (Fraction(1, 3), Fraction(2, 3), Fraction(-1, 4), Fraction(5, 4))
    subdivision = tropical_bivariate_regular_subdivision(
        _polynomial("MIN_PLUS", coefficients)
    )
    assert tuple(
        sorted(face.source_term_indices for face in subdivision.lifted_faces)
    ) == _supporting_lifted_face_oracle(coefficients, "MIN_PLUS")
    assert all(
        type(component) is CanonicalRational
        for face in subdivision.lifted_faces
        for component in (*face.normal, face.offset)
    )


@pytest.mark.parametrize("rejection", ["terms", "height"])
def test_subdivision_rejects_before_entering_hull_kernel(
    monkeypatch: pytest.MonkeyPatch, rejection: str
) -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    if rejection == "terms":
        exponents_and_heights = tuple(
            (exponents, 0)
            for exponents in sorted(
                (
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1),
                    (2, 0),
                    (0, 2),
                    (2, 1),
                    (1, 2),
                    (2, 2),
                    (3, 0),
                    (0, 3),
                )
            )
        )
    else:
        exponents_and_heights = (
            ((0, 0), 10**32),
            ((0, 1), 0),
            ((1, 0), 0),
            ((1, 1), 0),
        )
    polynomial = TropicalPolynomial(
        semiring=semiring,
        variables=("x", "y"),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponents,
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(height, 1),
                ),
            )
            for exponents, height in exponents_and_heights
        ),
    )
    monkeypatch.setattr(
        regular_subdivision,
        "points_to_facets",
        lambda *_args, **_kwargs: pytest.fail("hull kernel entered before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError):
        tropical_bivariate_regular_subdivision(polynomial)


def test_request_requires_two_variables() -> None:
    with pytest.raises(ValueError):
        BivariateRegularSubdivisionRequest(
            polynomial=_polynomial("MIN_PLUS", (0, 0, 0, 0)).model_copy(
                update={"variables": ("x",)}
            )
        )
