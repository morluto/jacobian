"""Typed declarations for elliptic curve operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.elliptic_curves._models import (
    CurveDiscriminantResult,
    CurvePointRequest,
    EllipticCurvePointAdditionRequest,
    EllipticCurvePointResult,
    EllipticCurveRequest,
    PointOnCurveResult,
    ScalarMultiplicationRequest,
    ScalarMultiplicationResult,
)
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldCardinalityResult,
    FiniteFieldCurveBaseChangeRequest,
    FiniteFieldCurveBaseChangeResult,
    FiniteFieldCurveRequest,
    FiniteFieldDiscriminantRequest,
    FiniteFieldDiscriminantResult,
    FiniteFieldExtensionCountsRequest,
    FiniteFieldExtensionCountsResult,
    FiniteFieldGroupStructureResult,
    FiniteFieldIsogenyClassRequest,
    FiniteFieldIsogenyClassResult,
    FiniteFieldIsomorphismRequest,
    FiniteFieldIsomorphismResult,
    FiniteFieldPointAdditionRequest,
    FiniteFieldPointCheckResult,
    FiniteFieldPointOrderRequest,
    FiniteFieldPointOrderResult,
    FiniteFieldPointRequest,
    FiniteFieldPointResult,
    FiniteFieldPointSet,
    FiniteFieldScalarRequest,
    FiniteFieldShortWeierstrassCurve,
    finite_field_cardinality,
    finite_field_curve_base_change,
    finite_field_discriminant,
    finite_field_extension_counts,
    finite_field_group_structure,
    finite_field_isogeny_class,
    finite_field_isomorphism,
    finite_field_point_add,
    finite_field_point_check,
    finite_field_point_negate,
    finite_field_point_order,
    finite_field_point_scalar,
    finite_field_points,
    finite_field_quadratic_twist,
)
from jacobian.math.number_theory.elliptic_curves.operations import (
    add_points,
    discriminant,
    point_on_curve,
    scalar_multiply,
)


def compute_discriminant(request: EllipticCurveRequest) -> CurveDiscriminantResult:
    """Unpack a wire request for the native discriminant operation."""
    return discriminant(request.curve)


def check_point_on_curve(request: CurvePointRequest) -> PointOnCurveResult:
    """Unpack a wire request for the native point-membership operation."""
    return point_on_curve(request.curve, request.point)


def compute_add_points(
    request: EllipticCurvePointAdditionRequest,
) -> EllipticCurvePointResult:
    """Unpack a wire request for the native point-addition operation."""
    return add_points(request.curve, request.first, request.second)


def compute_scalar_multiply(
    request: ScalarMultiplicationRequest,
) -> ScalarMultiplicationResult:
    """Unpack a wire request for the native scalar-multiplication operation."""
    return scalar_multiply(request.curve, request.point, request.scalar)


def compute_finite_field_discriminant(
    request: FiniteFieldDiscriminantRequest,
) -> FiniteFieldDiscriminantResult:
    """Unpack a wire request for the native finite-field discriminant."""
    return finite_field_discriminant(
        request.field, request.coefficient_a, request.coefficient_b
    )


def compute_finite_field_quadratic_twist(
    request: FiniteFieldCurveRequest,
) -> FiniteFieldShortWeierstrassCurve:
    return finite_field_quadratic_twist(request.curve)


def compute_finite_field_group_structure(
    request: FiniteFieldCurveRequest,
) -> FiniteFieldGroupStructureResult:
    return finite_field_group_structure(request.curve)


_DISCRIMINANT_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
}

_POINT_ON_CURVE_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "point": {
        "x": {"num": "0", "den": "1"},
        "y": {"num": "0", "den": "1"},
    },
}

_POINT_ADDITION_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "first": {
        "curve": {
            "coefficient_a": {"num": "1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "0", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
    "second": {
        "curve": {
            "coefficient_a": {"num": "1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "0", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
}

_SCALAR_MULT_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "-1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "point": {
        "curve": {
            "coefficient_a": {"num": "-1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "1", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
    "scalar": 2,
}


_F5_PRESENTATION: dict[str, Any] = {
    "characteristic": "5",
    "modulus_coefficients": ["0", "1"],
    "generator": "a",
}


def _finite_field_element(coordinate: int) -> dict[str, Any]:
    return {
        "presentation": _F5_PRESENTATION,
        "coordinates": [str(coordinate)],
    }


def _finite_curve() -> dict[str, Any]:
    return {
        "field": _F5_PRESENTATION,
        "coefficient_a": _finite_field_element(1),
        "coefficient_b": _finite_field_element(1),
    }


def _finite_point(x: int, y: int) -> dict[str, Any]:
    return {
        "curve": _finite_curve(),
        "at_infinity": False,
        "x": _finite_field_element(x),
        "y": _finite_field_element(y),
    }


_FINITE_INFINITY = {"curve": _finite_curve(), "at_infinity": True, "x": None, "y": None}

_FINITE_FIELD_DISCRIMINANT_EXAMPLE: dict[str, Any] = {
    "field": _F5_PRESENTATION,
    "coefficient_a": _finite_field_element(1),
    "coefficient_b": _finite_field_element(1),
}

_FINITE_FIELD_ISOGENY_EXAMPLE: dict[str, Any] = {
    "first": _finite_curve(),
    "second": {
        **_finite_curve(),
        "coefficient_a": _finite_field_element(2),
        "coefficient_b": _finite_field_element(1),
    },
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="elliptic_curve.finite_field.point.check",
        title="Check a finite-field elliptic-curve point",
        description="Check whether an infinity or affine point lies on its exact nonsingular short-Weierstrass curve; a well-formed off-curve point is a mathematical negative.",
        request_type=FiniteFieldPointRequest,
        result_type=FiniteFieldPointCheckResult,
        run=lambda request: finite_field_point_check(request.curve, request.point),
        tags=("elliptic-curve", "finite-field", "point", "exact"),
        examples=(
            OperationExample(
                name="check_infinity",
                description="Check the identity point; it must retain the exact nonsingular curve parent.",
                input={"curve": _finite_curve(), "point": _FINITE_INFINITY},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.point.negate.compute",
        title="Negate a finite-field elliptic-curve point",
        description="Negate a projective point on a nonsingular short-Weierstrass curve over an exact finite field; the point and curve must share one field presentation.",
        request_type=FiniteFieldPointRequest,
        result_type=FiniteFieldPointResult,
        run=lambda request: finite_field_point_negate(request.curve, request.point),
        tags=("elliptic-curve", "finite-field", "point", "exact"),
        examples=(
            OperationExample(
                name="negate_infinity",
                description="Negate the identity point; the curve must be nonsingular over an odd finite field.",
                input={"curve": _finite_curve(), "point": _FINITE_INFINITY},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.point.add.compute",
        title="Add finite-field elliptic-curve points",
        description="Compute the exact group sum of two projective points on one nonsingular short-Weierstrass curve; operands must retain the identical curve parent.",
        request_type=FiniteFieldPointAdditionRequest,
        result_type=FiniteFieldPointResult,
        run=lambda request: finite_field_point_add(
            request.curve, request.first, request.second
        ),
        tags=("elliptic-curve", "finite-field", "group-law", "exact"),
        examples=(
            OperationExample(
                name="secant_addition_over_five",
                description="Add (0,1) and (2,1) on y^2=x^3+x+1 over F5; both points retain the exact curve parent.",
                input={
                    "curve": _finite_curve(),
                    "first": _finite_point(0, 1),
                    "second": _finite_point(2, 1),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.point.scalar_multiply.compute",
        title="Multiply a finite-field elliptic-curve point",
        description="Compute an exact signed scalar multiple by double-and-add; the point must be on the supplied nonsingular curve.",
        request_type=FiniteFieldScalarRequest,
        result_type=FiniteFieldPointResult,
        run=lambda request: finite_field_point_scalar(
            request.curve, request.point, request.scalar
        ),
        tags=("elliptic-curve", "finite-field", "scalar", "exact"),
        examples=(
            OperationExample(
                name="zero_times_identity",
                description="Compute zero times the identity; the point must use the curve's exact field presentation.",
                input={
                    "curve": _finite_curve(),
                    "point": _FINITE_INFINITY,
                    "scalar": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.point.order.compute",
        title="Compute the exact order of a finite-field elliptic-curve point",
        description="Compute a point's exact order using the exact curve cardinality and prime-divisor minimality witnesses; the field order must fit the bounded trace computation.",
        request_type=FiniteFieldPointOrderRequest,
        result_type=FiniteFieldPointOrderResult,
        run=lambda request: finite_field_point_order(request.curve, request.point),
        tags=("elliptic-curve", "finite-field", "point-order", "exact"),
        discovery_terms=(
            "finite-field elliptic-curve point order",
            "elliptic point torsion order over finite field",
        ),
        examples=(
            OperationExample(
                name="order_of_point_over_five",
                description="Compute the exact order of (0,1) on y²=x³+x+1 over F5 with annihilator and prime-divisor checks.",
                input={"curve": _finite_curve(), "point": _finite_point(0, 1)},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.points.enumerate",
        title="Enumerate finite-field elliptic-curve points",
        description="Return every projective point on a nonsingular short-Weierstrass curve by exhaustive exact field enumeration; the field order must fit the bounded enumeration envelope.",
        request_type=FiniteFieldCurveRequest,
        result_type=FiniteFieldPointSet,
        run=lambda request: finite_field_points(request.curve),
        tags=("elliptic-curve", "finite-field", "enumeration", "exact"),
        examples=(
            OperationExample(
                name="enumerate_five_field",
                description="Enumerate all points over F5; the curve must be nonsingular and the field order must fit the exhaustive bound.",
                input={"curve": _finite_curve()},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.base_change.compute",
        title="Transport a finite-field elliptic curve along an embedding",
        description="Transport a nonsingular short-Weierstrass curve and optional curve-bound point along an explicit finite-field embedding whose source-generator root relation is checked exactly.",
        request_type=FiniteFieldCurveBaseChangeRequest,
        result_type=FiniteFieldCurveBaseChangeResult,
        run=lambda request: finite_field_curve_base_change(
            request.curve, request.embedding, request.point
        ),
        tags=("elliptic-curve", "finite-field", "base-change", "exact"),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.cardinality.exhaustive.compute",
        title="Count finite-field elliptic-curve points",
        description="Compute the exact cardinality and Frobenius trace from exhaustive projective point enumeration, retaining the field and curve model.",
        request_type=FiniteFieldCurveRequest,
        result_type=FiniteFieldCardinalityResult,
        run=lambda request: finite_field_cardinality(request.curve),
        tags=("elliptic-curve", "finite-field", "cardinality", "exact"),
        examples=(
            OperationExample(
                name="count_five_field",
                description="Count the points over F5; exhaustive enumeration requires a nonsingular curve over a bounded finite field.",
                input={"curve": _finite_curve()},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.extension_counts.compute",
        title="Count finite-field elliptic curve extensions",
        description="Compute exact point counts over F_(q^n) for a bounded degree prefix from the Frobenius recurrence, deriving the base trace by exhaustive exact point enumeration.",
        request_type=FiniteFieldExtensionCountsRequest,
        result_type=FiniteFieldExtensionCountsResult,
        run=lambda request: finite_field_extension_counts(
            request.curve, request.max_degree
        ),
        tags=("elliptic-curve", "finite-field", "extension-counts", "exact"),
        examples=(
            OperationExample(
                name="count_extensions_five_field",
                description="Count the curve over F5 and its first two extension fields using exact Frobenius recurrence.",
                input={"curve": _finite_curve(), "max_degree": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.quadratic_twist.compute",
        title="Construct the canonical nontrivial quadratic twist",
        description=(
            "Return the canonical nontrivial quadratic twist of a nonsingular "
            "short-Weierstrass curve over an admitted finite field. The kernel "
            "chooses the least encoded nonsquare d and returns y^2 = x^3 + "
            "d^2 A x + d^3 B. Its point count has the opposite Frobenius trace."
        ),
        request_type=FiniteFieldCurveRequest,
        result_type=FiniteFieldShortWeierstrassCurve,
        run=compute_finite_field_quadratic_twist,
        tags=("elliptic-curve", "finite-field", "quadratic-twist", "exact"),
        discovery_terms=(
            "quadratic twist over finite fields",
            "nontrivial elliptic curve twist",
        ),
        examples=(
            OperationExample(
                name="nontrivial_twist_over_five",
                description="Return the canonical nontrivial twist over F5.",
                input={"curve": _finite_curve()},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.group_structure.compute",
        title="Compute the finite-field elliptic-curve group structure",
        description=(
            "Return the invariant factors of E(F_q) and exact curve-bound point "
            "generators. The complete point set is enumerated within the "
            "admitted Hasse-order and work bounds; the returned generators "
            "form the full direct product, not only separate cyclic subgroups."
        ),
        request_type=FiniteFieldCurveRequest,
        result_type=FiniteFieldGroupStructureResult,
        run=compute_finite_field_group_structure,
        tags=("elliptic-curve", "finite-field", "group-structure", "exact"),
        discovery_terms=(
            "finite-field elliptic curve group structure",
            "elliptic curve invariant factors and generators",
        ),
        examples=(
            OperationExample(
                name="point_group_over_five",
                description="Compute invariant factors and full point generators for y²=x³+x+1 over F5.",
                input={"curve": _finite_curve()},
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.isogeny_class.decide",
        title="Compare finite-field elliptic-curve isogeny classes",
        description="Decide whether two nonsingular short-Weierstrass curves over the same exact finite-field presentation are isogenous by comparing their exactly computed Frobenius polynomials. This does not construct an isogeny.",
        request_type=FiniteFieldIsogenyClassRequest,
        result_type=FiniteFieldIsogenyClassResult,
        run=lambda request: finite_field_isogeny_class(request.first, request.second),
        tags=("elliptic-curve", "finite-field", "isogeny-class", "exact"),
        discovery_terms=(
            "compare elliptic curve isogeny classes over a finite field",
            "finite-field elliptic isogeny by Frobenius polynomial",
        ),
        examples=(
            OperationExample(
                name="compare_two_curves_over_five",
                description="Compare two nonsingular curves over the same F5 presentation by their exact point counts and Frobenius polynomials.",
                input=_FINITE_FIELD_ISOGENY_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.isomorphism.decide",
        title="Decide finite-field elliptic-curve model isomorphism",
        description=(
            "Decide short-Weierstrass model isomorphism over the same exact "
            "finite-field presentation by a complete bounded search for the "
            "scaling u with (x,y) mapped to (u^2*x,u^3*y)."
        ),
        request_type=FiniteFieldIsomorphismRequest,
        result_type=FiniteFieldIsomorphismResult,
        run=lambda request: finite_field_isomorphism(request.source, request.target),
        tags=("elliptic-curve", "finite-field", "isomorphism", "exact"),
        discovery_terms=(
            "isomorphism of short Weierstrass elliptic curves over finite fields",
            "finite-field elliptic curve model isomorphism",
        ),
        examples=(
            OperationExample(
                name="scaled_models_over_five",
                description="Find the explicit scaling between isomorphic models over F5.",
                input={
                    "source": _finite_curve(),
                    "target": {
                        **_finite_curve(),
                        "coefficient_a": _finite_field_element(1),
                        "coefficient_b": _finite_field_element(4),
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.elliptic_curve.short_weierstrass.discriminant.compute",
        title="Compute the discriminant of a short Weierstrass elliptic curve",
        description="Compute the exact discriminant Δ = -16(4A³ + 27B²) of a short "
        "Weierstrass curve y² = x³ + Ax + B over QQ, together with the "
        "nonsingularity predicate (Δ ≠ 0).",
        request_type=EllipticCurveRequest,
        result_type=CurveDiscriminantResult,
        run=compute_discriminant,
        tags=("elliptic-curve", "discriminant", "exact"),
        examples=(
            OperationExample(
                name="y_squared_equals_x_cubed_plus_x",
                description="Compute the discriminant of y² = x³ + x (A=1, B=0).",
                input=_DISCRIMINANT_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.elliptic_curve.short_weierstrass.point_on_curve.decide",
        title="Check whether a point lies on a short Weierstrass elliptic curve",
        description="Check whether a rational affine point (x, y) lies on the curve "
        "y² = x³ + Ax + B by verifying y² = x³ + Ax + B exactly over QQ.",
        request_type=CurvePointRequest,
        result_type=PointOnCurveResult,
        run=check_point_on_curve,
        tags=("elliptic-curve", "point-on-curve", "exact"),
        examples=(
            OperationExample(
                name="origin_on_x_cubed_plus_x",
                description="Check whether (0, 0) lies on y² = x³ + x.",
                input=_POINT_ON_CURVE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.elliptic_curve.short_weierstrass.point_addition.compute",
        title="Add two points on a short Weierstrass elliptic curve",
        description="Add two rational affine points P₁ + P₂ on y² = x³ + Ax + B using "
        "the exact chord-and-tangent group law over QQ. Returns the point "
        "at infinity when P₁ + P₂ = O.",
        request_type=EllipticCurvePointAdditionRequest,
        result_type=EllipticCurvePointResult,
        run=compute_add_points,
        tags=("elliptic-curve", "point-addition", "group-law", "exact"),
        examples=(
            OperationExample(
                name="double_origin_on_x_cubed_plus_x",
                description="Compute (0,0) + (0,0) on y² = x³ + x; the result is at infinity.",
                input=_POINT_ADDITION_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.elliptic_curve.short_weierstrass.scalar_multiply.compute",
        title="Compute n*P on a short Weierstrass elliptic curve",
        description="Compute the scalar multiple n*P on y² = x³ + Ax + B using the "
        "double-and-add method over QQ. Returns the point at infinity "
        "when n*P = O.",
        request_type=ScalarMultiplicationRequest,
        result_type=ScalarMultiplicationResult,
        run=compute_scalar_multiply,
        tags=("elliptic-curve", "scalar-multiplication", "group-law", "exact"),
        examples=(
            OperationExample(
                name="double_point_on_x_cubed_minus_x",
                description="Compute 2*(1,0) on y² = x³ - x; the result is at infinity.",
                input=_SCALAR_MULT_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="elliptic_curve.finite_field.short_weierstrass.discriminant.compute",
        title="Compute the discriminant of a finite-field short Weierstrass pair",
        description="Compute 4A³, 27B², Δ = -16(4A³ + 27B²), the nonsingularity "
        "predicate, and the j-invariant (present exactly when Δ ≠ 0) for "
        "coefficients A, B over one declared finite field of odd "
        "characteristic above 3. Characteristics 2 and 3 need generalized "
        "models and are rejected before arithmetic.",
        request_type=FiniteFieldDiscriminantRequest,
        result_type=FiniteFieldDiscriminantResult,
        run=compute_finite_field_discriminant,
        tags=("elliptic-curve", "finite-field", "discriminant", "exact"),
        discovery_terms=(
            "finite field Weierstrass discriminant",
            "elliptic curve singularity over a finite field",
            "j-invariant over a finite field",
        ),
        examples=(
            OperationExample(
                name="discriminant_over_f5",
                description="Compute the discriminant data of y² = x³ + x + 1 over F_5; "
                "the coefficients must share the declared field presentation.",
                input=_FINITE_FIELD_DISCRIMINANT_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
