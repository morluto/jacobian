"""Finite function-field operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.function_fields._models import (
    FiniteFunctionFieldElement,
    FunctionFieldBaseEmbeddingApplyRequest,
    FunctionFieldBaseEmbeddingApplyResult,
    FunctionFieldBaseEmbeddingRequest,
    FunctionFieldBaseEmbeddingResult,
    FunctionFieldDivisor,
    FunctionFieldDivisorAddRequest,
    FunctionFieldDivisorDegreeResult,
    FunctionFieldDivisorEffectivePartsResult,
    FunctionFieldDivisorRequest,
    FunctionFieldDivisorScaleRequest,
    FunctionFieldElementAddRequest,
    FunctionFieldElementInverseRequest,
    FunctionFieldElementMultiplyRequest,
    FunctionFieldElementMultiplyResult,
    FunctionFieldFiniteValuation,
    FunctionFieldGenusRequest,
    FunctionFieldGenusResult,
    FunctionFieldPlaceEnumerationRequest,
    FunctionFieldPlaceEnumerationResult,
    FunctionFieldPlaceValuationRequest,
    FunctionFieldPlaceValuationResult,
    FunctionFieldPositiveInfinityValuation,
    FunctionFieldPrincipalDivisorRequest,
    FunctionFieldPrincipalDivisorResult,
    FunctionFieldResidueRequest,
    FunctionFieldResidueResult,
    FunctionFieldRiemannRochSpace,
    FunctionFieldRiemannRochSpaceRequest,
    HyperellipticAffinePlaceValuationRequest,
    HyperellipticAffinePlaceValuationResult,
)
from jacobian.math.function_fields.operations import (
    function_field_base_embedding,
    function_field_base_embedding_apply,
    function_field_divisor_add,
    function_field_divisor_degree,
    function_field_divisor_effective_parts,
    function_field_divisor_negate,
    function_field_divisor_scale,
    function_field_element_add,
    function_field_element_inverse,
    function_field_element_multiply,
    function_field_genus,
    function_field_hyperelliptic_affine_valuation,
    function_field_place_residue,
    function_field_place_valuation,
    function_field_principal_divisor,
    function_field_rational_places_degree_bounded,
    function_field_riemann_roch_space,
)


def _run_element_multiply(
    request: FunctionFieldElementMultiplyRequest,
) -> FunctionFieldElementMultiplyResult:
    return function_field_element_multiply(request.left, request.right)


def _run_element_add(
    request: FunctionFieldElementAddRequest,
) -> FiniteFunctionFieldElement:
    return function_field_element_add(request.left, request.right)


def _run_element_inverse(
    request: FunctionFieldElementInverseRequest,
) -> FiniteFunctionFieldElement:
    return function_field_element_inverse(request.element)


def _rational_function(
    numerator: list[int], denominator: list[int], characteristic: int = 2
) -> dict[str, Any]:
    return {
        "numerator": {"characteristic": characteristic, "coefficients": numerator},
        "denominator": {"characteristic": characteristic, "coefficients": denominator},
    }


def _gf2_elliptic_field() -> dict[str, Any]:
    return {
        "characteristic": 2,
        "variable": "x",
        "generator": "y",
        "defining_polynomial": [
            _rational_function([0, 1], [1]),
            _rational_function([1], [1]),
            _rational_function([1], [1]),
        ],
    }


def _gf2_rational_field() -> dict[str, Any]:
    return {
        "characteristic": 2,
        "variable": "x",
        "generator": "y",
        "defining_polynomial": [_rational_function([1], [1])],
    }


def _gf2_base_embedding_apply() -> dict[str, Any]:
    target = _gf2_elliptic_field()
    return {
        "embedding": {
            "source": _gf2_rational_field(),
            "target": target,
            "variable_image": {
                "field": target,
                "coordinates": [
                    _rational_function([0, 1], [1]),
                    _rational_function([0], [1]),
                ],
            },
        },
        "element": {
            "field": _gf2_rational_field(),
            "coordinates": [_rational_function([1, 1], [1, 1])],
        },
    }


_RATIONAL_FIELD = {
    "characteristic": 5,
    "variable": "x",
    "generator": "y",
    "defining_polynomial": [
        {
            "numerator": {"characteristic": 5, "coefficients": [1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        }
    ],
}
_GF5_HYPERELLIPTIC_FIELD = {
    "characteristic": 5,
    "variable": "x",
    "generator": "y",
    "defining_polynomial": [
        {
            "numerator": {"characteristic": 5, "coefficients": [0, 1, 0, 4]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
        {
            "numerator": {"characteristic": 5, "coefficients": [0]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
        {
            "numerator": {"characteristic": 5, "coefficients": [1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        },
    ],
}
_RATIONAL_X = {
    "field": _RATIONAL_FIELD,
    "coordinates": [
        {
            "numerator": {"characteristic": 5, "coefficients": [0, 1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        }
    ],
}
_X_DIVISOR = {
    "field": _RATIONAL_FIELD,
    "terms": [
        {
            "place": {
                "field": _RATIONAL_FIELD,
                "kind": "FINITE",
                "prime_polynomial": {"characteristic": 5, "coefficients": [0, 1]},
                "degree": 1,
            },
            "multiplicity": "1",
        }
    ],
}


def _run_place_valuation(
    request: FunctionFieldPlaceValuationRequest,
) -> FunctionFieldPlaceValuationResult:
    valuation = function_field_place_valuation(request.place, request.element)
    return FunctionFieldPlaceValuationResult(
        place=request.place,
        element=request.element,
        valuation=(
            FunctionFieldPositiveInfinityValuation(kind="POSITIVE_INFINITY")
            if valuation is None
            else FunctionFieldFiniteValuation(kind="FINITE", value=valuation)
        ),
    )


def _run_hyperelliptic_affine_valuation(
    request: HyperellipticAffinePlaceValuationRequest,
) -> HyperellipticAffinePlaceValuationResult:
    return function_field_hyperelliptic_affine_valuation(
        request.place, request.element
    )


def _run_place_residue(
    request: FunctionFieldResidueRequest,
) -> FunctionFieldResidueResult:
    return function_field_place_residue(request.place, request.element)


def _run_rational_place_enumeration(
    request: FunctionFieldPlaceEnumerationRequest,
) -> FunctionFieldPlaceEnumerationResult:
    return function_field_rational_places_degree_bounded(
        request.field, request.maximum_degree
    )


def _run_principal(
    request: FunctionFieldPrincipalDivisorRequest,
) -> FunctionFieldPrincipalDivisorResult:
    return function_field_principal_divisor(request.field, request.element)


_GF2_Y = {
    "field": _gf2_elliptic_field(),
    "coordinates": [
        _rational_function([0], [1]),
        _rational_function([1], [1]),
    ],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="function_field.base_embedding.apply",
        title="Apply the rational base-field inclusion",
        description=(
            "Apply a validated canonical GF(p)(x) inclusion to one rational "
            "function-field element. The result is a target-parent element "
            "with the same exact rational function in its constant power-basis "
            "coordinate; it composes with target-field arithmetic."
        ),
        request_type=FunctionFieldBaseEmbeddingApplyRequest,
        result_type=FunctionFieldBaseEmbeddingApplyResult,
        run=lambda request: function_field_base_embedding_apply(
            request.embedding, request.element
        ),
        tags=("function-field", "embedding", "transport", "exact"),
        examples=(
            OperationExample(
                name="map_x_plus_one_over_x",
                description=("Apply GF(2)(x) -> GF(2)(x)[y]/(y^2+y+x) to (x+1)/x."),
                input=_gf2_base_embedding_apply(),
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.base_embedding.construct",
        title="Construct the rational base-field inclusion",
        description=(
            "Construct the canonical inclusion GF(p)(x) -> GF(p)(x)[y]/(f) "
            "for one admitted finite separable extension. It fixes GF(p) and "
            "the named rational variable; arbitrary tower and generator maps "
            "are outside this operation."
        ),
        request_type=FunctionFieldBaseEmbeddingRequest,
        result_type=FunctionFieldBaseEmbeddingResult,
        run=lambda request: function_field_base_embedding(request.target),
        tags=("function-field", "embedding", "exact"),
        examples=(
            OperationExample(
                name="rational_base_into_gf2_elliptic_field",
                description="Record the natural GF(2)(x) inclusion into GF(2)(x)[y]/(y^2+y+x).",
                input={"target": _gf2_elliptic_field()},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.add.compute",
        title="Add two function-field divisors",
        description="Add finite divisors over the same rational function field, combining equal canonical places and deleting zero coefficients.",
        request_type=FunctionFieldDivisorAddRequest,
        result_type=FunctionFieldDivisor,
        run=lambda request: function_field_divisor_add(request.left, request.right),
        tags=("function-field", "divisor", "exact"),
        examples=(
            OperationExample(
                name="add_a_finite_place_to_itself",
                description="Add [x] to [x] in the divisor group of GF(5)(x).",
                input={"left": _X_DIVISOR, "right": _X_DIVISOR},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.negate.compute",
        title="Negate a function-field divisor",
        description="Negate every multiplicity in a finite divisor over the rational function field GF(p)(x).",
        request_type=FunctionFieldDivisorRequest,
        result_type=FunctionFieldDivisor,
        run=lambda request: function_field_divisor_negate(request.divisor),
        tags=("function-field", "divisor", "exact"),
        examples=(
            OperationExample(
                name="negate_a_finite_place",
                description="Return -[x] in the divisor group of GF(5)(x).",
                input={"divisor": _X_DIVISOR},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.scale.compute",
        title="Scale a function-field divisor",
        description="Multiply every multiplicity in a finite divisor over GF(p)(x) by a bounded exact integer.",
        request_type=FunctionFieldDivisorScaleRequest,
        result_type=FunctionFieldDivisor,
        run=lambda request: function_field_divisor_scale(
            request.divisor, request.scalar
        ),
        tags=("function-field", "divisor", "exact"),
        examples=(
            OperationExample(
                name="double_a_finite_place",
                description="Return 2[x] in the divisor group of GF(5)(x).",
                input={"divisor": _X_DIVISOR, "scalar": "2"},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.places.degree_bounded.enumerate",
        title="Enumerate rational function-field places through a degree bound",
        description=(
            "Return every place of GF(p)(x) of degree at most the requested "
            "bound: each finite place is represented by its unique monic "
            "irreducible polynomial, and the degree-one infinite place is "
            "included. The finite field must be rational; extensions are not "
            "enumerated by this operation. Candidate and irreducibility work "
            "are admitted before enumeration."
        ),
        request_type=FunctionFieldPlaceEnumerationRequest,
        result_type=FunctionFieldPlaceEnumerationResult,
        run=_run_rational_place_enumeration,
        tags=("function-field", "places", "enumeration", "exact"),
        examples=(
            OperationExample(
                name="rational_places_over_gf2_through_degree_three",
                description=(
                    "Enumerate every place of GF(2)(x) through degree 3, "
                    "including the infinite place."
                ),
                input={
                    "field": {
                        "characteristic": 2,
                        "variable": "x",
                        "generator": "y",
                        "defining_polynomial": [
                            {
                                "numerator": {
                                    "characteristic": 2,
                                    "coefficients": [1],
                                },
                                "denominator": {
                                    "characteristic": 2,
                                    "coefficients": [1],
                                },
                            }
                        ],
                    },
                    "maximum_degree": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.riemann_roch_space.compute",
        title="Compute a rational function-field Riemann-Roch space",
        description=(
            "Return the exact basis and dimension of L(D) for a divisor over "
            "GF(p)(x). The field must use the rational defining polynomial 1. "
            "Support is limited to 256 places and multiplicities to 4096 bits; "
            "positive-dimensional outputs require at most 13 basis elements "
            "and degree-12 canonical rational-function coefficients."
        ),
        request_type=FunctionFieldRiemannRochSpaceRequest,
        result_type=FunctionFieldRiemannRochSpace,
        run=lambda request: function_field_riemann_roch_space(request.divisor),
        tags=("function-field", "riemann-roch", "exact", "basis"),
        examples=(
            OperationExample(
                name="basis_for_twice_infinity_on_projective_line",
                description=(
                    "Compute L(2[∞]) over GF(5)(x), with basis 1, x, x²; "
                    "the divisor must belong to the rational function field."
                ),
                input={
                    "divisor": {
                        "field": _RATIONAL_FIELD,
                        "terms": [
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "INFINITE",
                                    "degree": 1,
                                },
                                "multiplicity": "2",
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.genus.compute",
        title="Compute the genus of a supported function field",
        description=(
            "Return the exact genus of GF(p)(x), or of an odd-characteristic "
            "quadratic extension y^2=f(x) where f is a squarefree polynomial "
            "of degree 3 through 12. The exact field presentation is retained."
        ),
        request_type=FunctionFieldGenusRequest,
        result_type=FunctionFieldGenusResult,
        run=lambda request: function_field_genus(request.field),
        tags=("function-field", "genus", "exact"),
        examples=(
            OperationExample(
                name="genus_of_gf5_rational_field",
                description=(
                    "Compute the genus of GF(5)(x), which is zero; the field "
                    "must be the rational function field encoded by defining "
                    "polynomial 1."
                ),
                input={"field": _RATIONAL_FIELD},
            ),
            OperationExample(
                name="genus_of_y_squared_x_cubed_minus_x",
                description=(
                    "Compute genus 1 for y^2=x^3-x over GF(5)(x); its "
                    "squarefree cubic branch polynomial gives four branch "
                    "points counting infinity."
                ),
                input={"field": _GF5_HYPERELLIPTIC_FIELD},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.place.valuation.compute",
        title="Compute a function-field valuation",
        description=(
            "Compute the exact discrete valuation of a rational-function element "
            "at a finite or infinite place of GF(p)(x). A finite result is tagged "
            "FINITE and carries an integer (including zero); the zero element has "
            "the structural POSITIVE_INFINITY result."
        ),
        request_type=FunctionFieldPlaceValuationRequest,
        result_type=FunctionFieldPlaceValuationResult,
        run=_run_place_valuation,
        tags=("function-field", "place", "valuation", "exact"),
        examples=(
            OperationExample(
                name="valuation_of_x_at_zero",
                description="Compute v_(x)(x); the finite place must be the irreducible polynomial x in the rational field GF(5)(x).",
                input={
                    "place": {
                        "field": _RATIONAL_FIELD,
                        "kind": "FINITE",
                        "prime_polynomial": {
                            "characteristic": 5,
                            "coefficients": [0, 1],
                        },
                        "degree": 1,
                    },
                    "element": _RATIONAL_X,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.hyperelliptic_affine_place.valuation.compute",
        title="Compute a rational affine hyperelliptic valuation",
        description=(
            "Compute an exact valuation at one GF(p)-rational affine point of "
            "an odd-characteristic squarefree model y^2=f(x). The point retains "
            "its curve, coordinates, GF(p) residue parent, and local parameter "
            "(x-x0 off the branch locus, y at a branch point). Infinity and "
            "points over extension residue fields are not represented. Finite "
            "valuations carry an integer; the zero element returns the structural "
            "POSITIVE_INFINITY branch without a numeric value."
        ),
        request_type=HyperellipticAffinePlaceValuationRequest,
        result_type=HyperellipticAffinePlaceValuationResult,
        run=_run_hyperelliptic_affine_valuation,
        tags=("function-field", "hyperelliptic", "affine-place", "valuation", "exact"),
        examples=(
            OperationExample(
                name="branch_uniformizer_at_origin",
                description="On y^2=x^3-x over GF(5), y is the uniformizer at (0,0) and has valuation one.",
                input={
                    "place": {
                        "field": _GF5_HYPERELLIPTIC_FIELD,
                        "x": 0,
                        "y": 0,
                        "local_parameter": "y",
                        "residue_field": {
                            "characteristic": "5",
                            "modulus_coefficients": ["0", "1"],
                            "generator": "z",
                        },
                    },
                    "element": {
                        "field": _GF5_HYPERELLIPTIC_FIELD,
                        "coordinates": [
                            _rational_function([0], [1], characteristic=5),
                            _rational_function([1], [1], characteristic=5),
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.place.residue.compute",
        title="Reduce a regular rational function at a place",
        description=(
            "Compute the exact residue of an element of GF(p)(x) at a finite "
            "or infinite rational place. Elements with a pole are outside the "
            "map domain; finite residue fields use the canonical polynomial "
            "quotient presentation and are bounded by field order 65536."
        ),
        request_type=FunctionFieldResidueRequest,
        result_type=FunctionFieldResidueResult,
        run=_run_place_residue,
        tags=("function-field", "place", "residue", "exact"),
        examples=(
            OperationExample(
                name="reduce_x_squared_at_x_squared_plus_x_plus_one",
                description=(
                    "Reduce x^2 in GF(5)[x]/(x^2+x+1), where the residue "
                    "of x is represented by the finite-field generator z."
                ),
                input={
                    "place": {
                        "field": _RATIONAL_FIELD,
                        "kind": "FINITE",
                        "prime_polynomial": {
                            "characteristic": 5,
                            "coefficients": [1, 1, 1],
                        },
                        "degree": 2,
                    },
                    "element": {
                        "field": _RATIONAL_FIELD,
                        "coordinates": [
                            {
                                "numerator": {
                                    "characteristic": 5,
                                    "coefficients": [0, 0, 1],
                                },
                                "denominator": {
                                    "characteristic": 5,
                                    "coefficients": [1],
                                },
                            }
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.element.principal_divisor.compute",
        title="Compute a principal divisor",
        description="Compute the complete finite zero/pole divisor of a nonzero rational-function element, retaining every place and exact degree-zero parent identity.",
        request_type=FunctionFieldPrincipalDivisorRequest,
        result_type=FunctionFieldPrincipalDivisorResult,
        run=_run_principal,
        tags=("function-field", "divisor", "exact"),
        examples=(
            OperationExample(
                name="principal_divisor_of_x",
                description="Compute div(x); the element must be nonzero in the rational function field GF(5)(x).",
                input={"field": _RATIONAL_FIELD, "element": _RATIONAL_X},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.effective_parts.compute",
        title="Compute effective parts of a function-field divisor",
        description=(
            "Split a finite divisor D over GF(p)(x) into its coefficientwise "
            "effective parts D_+ and D_- with D = D_+ - D_-. Each place must "
            "be a valid place of that rational function field; support is "
            "limited to 256 places and each multiplicity to 4096 bits."
        ),
        request_type=FunctionFieldDivisorRequest,
        result_type=FunctionFieldDivisorEffectivePartsResult,
        run=lambda request: function_field_divisor_effective_parts(request.divisor),
        tags=("function-field", "divisor", "effective-parts", "exact"),
        examples=(
            OperationExample(
                name="split_divisor_on_projective_line",
                description=(
                    "Split 2[x] - 3[∞] into its positive and negative parts; "
                    "both places must be valid places of GF(5)(x)."
                ),
                input={
                    "divisor": {
                        "field": _RATIONAL_FIELD,
                        "terms": [
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "FINITE",
                                    "prime_polynomial": {
                                        "characteristic": 5,
                                        "coefficients": [0, 1],
                                    },
                                    "degree": 1,
                                },
                                "multiplicity": "2",
                            },
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "INFINITE",
                                    "degree": 1,
                                },
                                "multiplicity": "-3",
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.degree.compute",
        title="Compute a divisor degree",
        description="Compute the exact degree of a finite function-field divisor from its place degrees and multiplicities; every place must belong to its declared field.",
        request_type=FunctionFieldDivisorRequest,
        result_type=FunctionFieldDivisorDegreeResult,
        run=lambda request: function_field_divisor_degree(request.divisor),
        tags=("function-field", "divisor", "degree", "exact"),
        examples=(
            OperationExample(
                name="degree_of_principal_x",
                description="Compute the degree of div(x); every place must retain the rational-field parent.",
                input={
                    "divisor": {
                        "field": _RATIONAL_FIELD,
                        "terms": [
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "FINITE",
                                    "prime_polynomial": {
                                        "characteristic": 5,
                                        "coefficients": [0, 1],
                                    },
                                    "degree": 1,
                                },
                                "multiplicity": "1",
                            },
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "INFINITE",
                                    "degree": 1,
                                },
                                "multiplicity": "-1",
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.element.add.compute",
        title="Add two elements of a finite function field",
        description=(
            "Add corresponding reduced generator-power coordinates in "
            "GF(p)(x)[y]/(f), with exact rational-function addition over "
            "GF(p). Both elements must have the identical field parent. "
            "Return the direct canonical reduced element; coefficient degree, "
            "work and serialized output are admitted before coordinate "
            "addition."
        ),
        request_type=FunctionFieldElementAddRequest,
        result_type=FiniteFunctionFieldElement,
        run=_run_element_add,
        tags=("algebra", "function-field", "function-field-element", "exact"),
        discovery_terms=(
            "function field addition",
            "algebraic function field element sum",
            "GF(p)(x)[y] coordinate addition",
        ),
        examples=(
            OperationExample(
                name="characteristic_two_generator_cancellation",
                description=("In GF(2)(x)[y]/(y^2+y+x), add y and y+x to return x."),
                input={
                    "left": _GF2_Y,
                    "right": {
                        "field": _gf2_elliptic_field(),
                        "coordinates": [
                            _rational_function([0, 1], [1]),
                            _rational_function([1], [1]),
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.element.inverse.compute",
        title="Invert a nonzero finite function-field element",
        description=(
            "Compute the unique multiplicative inverse of a nonzero reduced "
            "element in its exact GF(p)(x)[y]/(f) parent using extended "
            "Euclid over GF(p)(x). The field parent is retained. A conservative "
            "coefficient-degree, work, and serialized-output bound is admitted "
            "before Euclidean expansion; zero is a domain error."
        ),
        request_type=FunctionFieldElementInverseRequest,
        result_type=FiniteFunctionFieldElement,
        run=_run_element_inverse,
        tags=("algebra", "function-field", "function-field-element", "exact"),
        discovery_terms=(
            "function field inverse",
            "algebraic function field element reciprocal",
            "GF(p)(x)[y] extended Euclidean algorithm",
        ),
        examples=(
            OperationExample(
                name="quadratic_generator_inverse",
                description=("Invert y in GF(2)(x)[y]/(y^2+y+x)."),
                input={"element": _GF2_Y},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.element.multiply.compute",
        title="Multiply two elements of a finite function field",
        description=(
            "Multiply two reduced elements of GF(p)(x)[y]/(f) by exact "
            "polynomial multiplication in the generator followed by reduction "
            "modulo the monic separable defining polynomial using exact "
            "GF(p)(x) arithmetic. Return the canonical reduced product and the "
            "complete raw-product and reduction ledger. Both elements must be "
            "bound to the identical function field; reducible or inseparable "
            "defining polynomials are rejected."
        ),
        request_type=FunctionFieldElementMultiplyRequest,
        result_type=FunctionFieldElementMultiplyResult,
        run=_run_element_multiply,
        tags=("algebra", "function-field", "function-field-element", "exact"),
        discovery_terms=(
            "function field multiplication",
            "algebraic function field element product",
            "GF(p)(x)[y] reduction",
            "rational function field extension arithmetic",
        ),
        examples=(
            OperationExample(
                name="gf2_y_squared",
                description=(
                    "Multiply the generator y by itself in GF(2)(x)[y]/(y^2+y+x); "
                    "the field must be a monic separable extension and both "
                    "elements must share it. The reduction y^2 = y+x is returned "
                    "in the ledger."
                ),
                input={"left": _GF2_Y, "right": _GF2_Y},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
