from collections.abc import Callable

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.blowup_p2._models import BlowupPoint
from jacobian.math.geometry.blowup_p2.operations import (
    adjunction_profile,
    canonical_class,
    construct_divisor_class,
    construct_surface,
    intersect_classes,
)
from jacobian.math.geometry.convex._models import (
    ActiveFacetProfileRequest,
    ConvexHPolytope,
    ConvexInequality,
    ConvexSpace,
    CoverageDirection,
    CoveragePoint,
    DirectionCoverageRequest,
    RationalConvexDirection,
    RationalConvexPoint,
)
from jacobian.math.geometry.convex.operations import (
    active_facet_profile,
    direction_set_coverage,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    PieceAssignment,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    polytopal_complex_closure,
    spline_space,
)
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.geometry.toric._kernel import RecognizedFan
from jacobian.math.geometry.toric._models import (
    NormalToricVariety,
    ToricFanPresentation,
)
from jacobian.math.geometry.toric.operations import (
    check_normal_toric_morphism,
    check_toric_morphism,
    construct_normal_toric_variety,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def q(value: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=value, den=den)


def test_blowup_arithmetic_binds_parent_and_adjunction() -> None:
    point = RationalProjectivePoint(coordinates=(q(1), q(0), q(0)))
    surface = construct_surface((BlowupPoint(label="p", point=point),))
    divisor = construct_divisor_class(surface, 1, (1,))
    assert intersect_classes(divisor, divisor).value == 0
    assert canonical_class(surface).multiplicities == (-1,)
    assert adjunction_profile(divisor).arithmetic_genus == 0


def test_active_facets_and_complete_coverage_matrix() -> None:
    space = ConvexSpace(axes=("x", "y"))
    polytope = ConvexHPolytope(
        space=space,
        inequalities=(
            ConvexInequality(inequality_id="bottom", normal=(q(0), q(-1)), bound=q(0)),
            ConvexInequality(inequality_id="left", normal=(q(-1), q(0)), bound=q(0)),
            ConvexInequality(inequality_id="right", normal=(q(1), q(0)), bound=q(1)),
            ConvexInequality(inequality_id="top", normal=(q(0), q(1)), bound=q(1)),
        ),
    )
    point = RationalConvexPoint(space=space, coordinates=(q(0), q(0)))
    direction = RationalConvexDirection(space=space, components=(q(1), q(1)))
    profile = active_facet_profile(polytope, point)
    assert profile.active_inequality_ids == ("bottom", "left")
    coverage = direction_set_coverage(
        DirectionCoverageRequest(
            polytope=polytope,
            points=(CoveragePoint(point_id="p", point=point),),
            directions=(CoverageDirection(direction_id="d", direction=direction),),
        )
    )
    assert len(coverage.cells) == 1
    assert coverage.strictly_illuminated_point_ids == ("p",)


def test_piecewise_face_compatibility_and_single_cell_spline() -> None:
    space = RationalCoordinateSpace(axes=("x",))
    cell = RationalVPolytope(
        space=space,
        vertices=(
            RationalPolytopeVertex(vertex_id="a", coordinates=(q(0),)),
            RationalPolytopeVertex(vertex_id="b", coordinates=(q(1),)),
        ),
    )
    complex_value = polytopal_complex_closure((cell,))
    polynomial = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=q(1), exponents=(0,)),)
        ),
    )
    function = piecewise_polynomial_from_maximal_pieces(
        complex_value, (PieceAssignment(cell_id="M0", polynomial=polynomial),)
    )
    assert function.status == "COMPATIBLE"
    value = piecewise_polynomial_evaluate(
        function,
        ComplexPoint(coordinates=(q(1, 2),)),
    )
    assert value.value == q(1)
    spline = spline_space(complex_value, 0, 0)
    assert spline.rank == 0 and spline.nullity == 1
    with pytest.raises(OperationDomainValidationError):
        spline_space(complex_value, 13, 0)


def test_piecewise_multivariate_evaluation_and_complete_obstruction_profile() -> None:
    space = RationalCoordinateSpace(axes=("x", "y"))
    square = RationalVPolytope(
        space=space,
        vertices=tuple(
            RationalPolytopeVertex(vertex_id=label, coordinates=coordinates)
            for label, coordinates in (
                ("a", (q(0), q(0))),
                ("b", (q(1), q(0))),
                ("c", (q(1), q(1))),
                ("d", (q(0), q(1))),
            )
        ),
    )
    complex_value = polytopal_complex_closure((square,))
    polynomial = RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(coefficient=q(1), exponents=(1, 0)),
                RationalPolynomialTerm(coefficient=q(1), exponents=(0, 1)),
            )
        ),
    )
    function = piecewise_polynomial_from_maximal_pieces(
        complex_value, (PieceAssignment(cell_id="M0", polynomial=polynomial),)
    )
    value = piecewise_polynomial_evaluate(
        function, ComplexPoint(coordinates=(q(1, 2), q(1, 2)))
    )
    assert value.value == q(1)

    line_space = RationalCoordinateSpace(axes=("x",))
    intervals = tuple(
        RationalVPolytope(
            space=line_space,
            vertices=(
                RationalPolytopeVertex(vertex_id=f"{index}a", coordinates=(q(index),)),
                RationalPolytopeVertex(
                    vertex_id=f"{index}b", coordinates=(q(index + 1),)
                ),
            ),
        )
        for index in range(3)
    )
    line_complex = polytopal_complex_closure(intervals)
    incompatible = piecewise_polynomial_from_maximal_pieces(
        line_complex,
        tuple(
            PieceAssignment(
                cell_id=f"M{index}",
                polynomial=RationalPolynomial(
                    variables=("x",),
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=q(index + 1), exponents=(0,)
                            ),
                        )
                    ),
                ),
            )
            for index in range(3)
        ),
    )
    assert incompatible.status == "INCOMPATIBLE"
    assert len(incompatible.compatibility) == 2


def test_piecewise_evaluation_rechecks_serialized_compatibility_claim() -> None:
    space = RationalCoordinateSpace(axes=("x",))
    intervals = tuple(
        RationalVPolytope(
            space=space,
            vertices=(
                RationalPolytopeVertex(vertex_id=f"{index}a", coordinates=(q(index),)),
                RationalPolytopeVertex(
                    vertex_id=f"{index}b", coordinates=(q(index + 1),)
                ),
            ),
        )
        for index in range(2)
    )
    complex_value = polytopal_complex_closure(intervals)

    def constant(value: int) -> RationalPolynomial:
        return RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(
                terms=(RationalPolynomialTerm(coefficient=q(value), exponents=(0,)),)
            ),
        )

    produced = piecewise_polynomial_from_maximal_pieces(
        complex_value,
        tuple(
            PieceAssignment(cell_id=f"M{index}", polynomial=constant(index + 1))
            for index in range(2)
        ),
    )
    # Replace the producer's truthful status/profile in the serialized payload
    # with a caller-authored compatible claim, as a consumer receives it.
    payload = produced.model_dump(mode="json")
    payload.update(
        {
            "status": "COMPATIBLE",
            "compatibility": [],
            "obstruction_face_id": None,
            "obstruction_difference": None,
        }
    )
    forged = PiecewisePolynomialResult.model_validate_json(encode_strict_json(payload))
    for point in (q(1, 2), q(1)):
        with pytest.raises(OperationDomainValidationError):
            piecewise_polynomial_evaluate(forged, ComplexPoint(coordinates=(point,)))


def test_piecewise_evaluation_admits_output_growth_before_substitution() -> None:
    endpoint = 10**31
    space = RationalCoordinateSpace(axes=("x",))
    interval = RationalVPolytope(
        space=space,
        vertices=(
            RationalPolytopeVertex(vertex_id="a", coordinates=(q(0),)),
            RationalPolytopeVertex(vertex_id="b", coordinates=(q(endpoint),)),
        ),
    )
    complex_value = polytopal_complex_closure((interval,))
    polynomial = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=q(1), exponents=(1100,)),)
        ),
    )
    function = piecewise_polynomial_from_maximal_pieces(
        complex_value, (PieceAssignment(cell_id="M0", polynomial=polynomial),)
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        piecewise_polynomial_evaluate(
            function, ComplexPoint(coordinates=(q(endpoint),))
        )
    assert error.value.errors()[0]["type"] == ("polytopal_complex.evaluation_growth")


def test_convex_profile_rejects_zero_normal_at_owner_boundary() -> None:
    space = ConvexSpace(axes=("x", "y"))
    degenerate = ConvexHPolytope.model_construct(
        space=space,
        inequalities=(
            ConvexInequality(inequality_id="zero", normal=(q(0), q(0)), bound=q(0)),
        ),
    )
    point = RationalConvexPoint(space=space, coordinates=(q(0), q(0)))
    with pytest.raises(ValidationError):
        ActiveFacetProfileRequest(polytope=degenerate, point=point)
    with pytest.raises(OperationDomainValidationError):
        active_facet_profile(degenerate, point)


def test_toric_morphism_native_matrix_admission_does_not_leak_attribute_error() -> None:
    fan = ToricFanPresentation(lattice_rank=1, rays=(), cones=((),))
    with pytest.raises(OperationDomainValidationError):
        check_toric_morphism(fan, fan, None)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        check_toric_morphism(fan, fan, IntegerMatrix.model_construct(entries=((0,),)))


def test_normal_toric_morphism_reconstructs_carriers_and_reuses_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.toric import operations

    fan = ToricFanPresentation(lattice_rank=1, rays=(), cones=((),))
    variety = construct_normal_toric_variety(fan)
    forged = NormalToricVariety.model_construct(
        field=variety.field,
        fan=variety.fan,
        orbit_profile=variety.orbit_profile,
        charts=(),
        gluing=variety.gluing,
        normal=True,
    )
    matrix = IntegerMatrix(row_count=1, column_count=1, entries=((0,),))
    with pytest.raises(OperationDomainValidationError) as error:
        check_normal_toric_morphism(forged, variety, matrix)
    assert error.value.errors()[0]["type"] == "toric.variety_source_mismatch"

    calls = 0
    recognize: Callable[
        [tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int],
        RecognizedFan,
    ] = operations.__dict__["recognize_fan"]

    def counted(
        rays: tuple[tuple[int, ...], ...],
        cones: tuple[tuple[int, ...], ...],
        lattice_rank: int,
    ) -> RecognizedFan:
        nonlocal calls
        calls += 1
        return recognize(rays, cones, lattice_rank)

    monkeypatch.setattr(operations, "recognize_fan", counted)
    result = check_normal_toric_morphism(variety, variety, matrix)
    assert result.map.is_toric_morphism
    assert calls == 2


def test_b5_native_admission_and_derived_growth_boundaries() -> None:
    with pytest.raises(OperationDomainValidationError):
        construct_surface(None)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        construct_surface([None])  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        construct_normal_toric_variety(None)  # type: ignore[arg-type]

    surface = construct_surface(())
    with pytest.raises(OperationResourceAdmissionError):
        construct_divisor_class(surface, 10**300, ())
    left = construct_divisor_class(surface, 10**200, ())
    right = construct_divisor_class(surface, 10**200, ())
    with pytest.raises(OperationResourceAdmissionError):
        intersect_classes(left, right)
