"""Public declarations for exact tropical operations."""
# ruff: noqa: F405

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.tropical._models import *  # noqa: F403
from jacobian.math.polynomials.tropical.hypersurface import (
    compute_bivariate_hypersurface,
)
from jacobian.math.polynomials.tropical.operations import *  # noqa: F403
from jacobian.math.polynomials.tropical.regular_subdivision import (
    compute_bivariate_regular_subdivision,
)
from jacobian.math.polynomials.tropical.values import *  # noqa: F403


def _s() -> dict[str, str]:
    return {"convention": "MIN_PLUS", "base": "ZZ"}


def _finite(n: int) -> dict[str, object]:
    return {"semiring": _s(), "kind": "FINITE", "value": {"num": str(n), "den": "1"}}


def _vector(values: tuple[int, ...], axis: tuple[str, ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "axis": list(axis),
        "entries": [_finite(v) for v in values],
    }


def _poly(values: tuple[tuple[tuple[int, ...], int], ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "variables": ["x"],
        "terms": [{"exponents": list(e), "coefficient": _finite(c)} for e, c in values],
    }


def _matrix(values: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "row_axis": ["a", "b"],
        "column_axis": ["a", "b"],
        "entries": [[_finite(v) for v in row] for row in values],
    }


def compute_scalar_add(request: ScalarAddRequest) -> ScalarAddResult:
    result, branch, case = tropical_scalar_add(
        request.semiring, request.left, request.right
    )
    return ScalarAddResult._from_kernel(
        request, result=result, branch=branch, infinity_case=case
    )


def compute_scalar_multiply(request: ScalarBinaryRequest) -> ScalarResult:
    return ScalarResult._from_kernel(
        tropical_scalar_multiply(request.left, request.right)
    )


def compute_scalar_power(request: ScalarPowerRequest) -> ScalarResult:
    return ScalarResult._from_kernel(
        tropical_scalar_power(request.scalar, request.exponent)
    )


def compute_scalar_dual(request: ScalarDualRequest) -> ScalarDualResult:
    return tropical_scalar_dual(request.scalar)


def compute_vector_add(request: VectorBinaryRequest) -> VectorResult:
    return VectorResult._from_kernel(tropical_vector_add(request.left, request.right))


def compute_vector_scale(request: VectorScaleRequest) -> VectorResult:
    return VectorResult._from_kernel(
        tropical_vector_scale(request.scalar, request.vector)
    )


def compute_vector_projectivize(
    request: VectorProjectivizeRequest,
) -> VectorProjectivizeResult:
    kind, representative, translation = tropical_vector_projectivize(request.vector)
    return VectorProjectivizeResult._from_kernel(
        source=request.vector,
        kind=kind,
        representative=representative,
        translation=translation,
    )


def compute_polynomial_add(request: PolynomialBinaryRequest) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_add(request.left, request.right)
    )


def compute_polynomial_multiply(request: PolynomialBinaryRequest) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_multiply(request.left, request.right)
    )


def compute_polynomial_power(request: PolynomialPowerRequest) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_power(request.polynomial, request.exponent)
    )


def compute_polynomial_substitute(
    request: PolynomialSubstituteRequest,
) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_substitute(
            request.polynomial,
            request.target_variables,
            request.images,
        )
    )


def compute_polynomial_evaluate(
    request: PolynomialEvaluateRequest,
) -> PolynomialEvaluateResult:
    value, active = tropical_polynomial_evaluate(request.polynomial, request.point)
    return PolynomialEvaluateResult._from_kernel(
        polynomial=request.polynomial,
        point=request.point,
        value=value,
        active_exponents=active,
    )


def compute_polynomial_active_terms(
    request: PolynomialActiveTermsRequest,
) -> PolynomialActiveTermsResult:
    return tropical_polynomial_active_terms(request.polynomial, request.point)


def compute_univariate_roots(
    request: UnivariateRootsRequest,
) -> TropicalUnivariateRootProfile:
    return tropical_polynomial_univariate_roots(request.polynomial)


def compute_univariate_split_form(
    request: UnivariateSplitFormRequest,
) -> TropicalPolynomial:
    return tropical_polynomial_univariate_split_form(request.polynomial)


def compute_univariate_newton_polygon(
    request: UnivariateNewtonPolygonRequest,
) -> UnivariateNewtonPolygonResult:
    return UnivariateNewtonPolygonResult._from_kernel(
        tropical_polynomial_univariate_newton_polygon(request.polynomial)
    )


def compute_matrix_multiply(request: MatrixMultiplyRequest) -> MatrixResult:
    return MatrixResult._from_kernel(
        tropical_matrix_multiply(request.left, request.right)
    )


def compute_matrix_power(request: MatrixPowerRequest) -> MatrixResult:
    return MatrixResult._from_kernel(
        tropical_matrix_power(request.matrix, request.exponent)
    )


def compute_finite_power_sum(
    request: MatrixFinitePowerSumRequest,
) -> FinitePowerSumResult:
    result, winners = tropical_matrix_finite_power_sum(
        request.matrix, request.max_power
    )
    return FinitePowerSumResult._from_kernel(
        source_matrix=request.matrix,
        max_power=request.max_power,
        matrix=result,
        winning_lengths=winners,
    )


def compute_assignment(request: MatrixAssignmentRequest) -> AssignmentResult:
    value, perms = tropical_assignment_profile(request.matrix)
    return AssignmentResult._from_kernel(
        matrix=request.matrix, value=value, permutations=perms
    )


def compute_minor_assignments(
    request: MatrixMinorAssignmentsRequest,
) -> MatrixMinorAssignmentsResult:
    return MatrixMinorAssignmentsResult._from_kernel(
        matrix=request.matrix,
        sizes=request.sizes,
        minors=tropical_matrix_minor_assignment_profiles(request.matrix, request.sizes),
    )


_EX = OperationExample(
    name="min_plus_finite",
    description="Compute min(3,5)=3; both finite scalars must carry the same MIN_PLUS semiring.",
    input={"semiring": _s(), "left": _finite(3), "right": _finite(5)},
)
TOOLS: MathTools = (
    MathTool(
        operation_id="tropical.scalar.add.compute",
        title="Add tropical scalars",
        description="Compute exact min-plus or max-plus addition.",
        request_type=ScalarAddRequest,
        result_type=ScalarAddResult,
        run=compute_scalar_add,
        tags=("tropical", "semiring", "exact"),
        examples=(_EX,),
    ),
    MathTool(
        operation_id="tropical.scalar.multiply.compute",
        title="Multiply tropical scalars",
        description="Compute exact tropical multiplication, ordinary addition with licensed infinity absorption.",
        request_type=ScalarBinaryRequest,
        result_type=ScalarResult,
        run=compute_scalar_multiply,
        tags=("tropical", "semiring", "exact"),
        examples=(
            OperationExample(
                name="multiply",
                description="Compute 3 times 5 in MIN_PLUS as 8; scalars must share a semiring.",
                input={"semiring": _s(), "left": _finite(3), "right": _finite(5)},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.scalar.power.compute",
        title="Power a tropical scalar",
        description="Compute a bounded exact tropical power.",
        request_type=ScalarPowerRequest,
        result_type=ScalarResult,
        run=compute_scalar_power,
        tags=("tropical", "semiring", "exact"),
        examples=(
            OperationExample(
                name="power",
                description="Compute 3 to tropical power 2 as 6; exponent must be nonnegative.",
                input={"scalar": _finite(3), "exponent": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.scalar.dual.compute",
        title="Dualize a tropical scalar",
        description=(
            "Map a min-plus scalar to max-plus or max-plus to min-plus by exact "
            "negation, swapping the licensed infinity and retaining explicit "
            "source and target semirings."
        ),
        request_type=ScalarDualRequest,
        result_type=ScalarDualResult,
        run=compute_scalar_dual,
        tags=("tropical", "semiring", "exact", "duality"),
        examples=(
            OperationExample(
                name="dual_finite",
                description="Map MIN_PLUS value 3 to MAX_PLUS value -3.",
                input={"scalar": _finite(3)},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.vector.add.compute",
        title="Add tropical vectors",
        description="Compute coordinatewise tropical addition on one labelled axis.",
        request_type=VectorBinaryRequest,
        result_type=VectorResult,
        run=compute_vector_add,
        tags=("tropical", "vector", "exact"),
        examples=(
            OperationExample(
                name="vector_add",
                description="Compute coordinatewise min of two vectors; vectors must share their labelled axis.",
                input={
                    "left": _vector((1, 4), ("x", "y")),
                    "right": _vector((2, 3), ("x", "y")),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.vector.scalar_multiply.compute",
        title="Scale a tropical vector",
        description="Add one finite tropical scalar to every vector coordinate.",
        request_type=VectorScaleRequest,
        result_type=VectorResult,
        run=compute_vector_scale,
        tags=("tropical", "vector", "exact"),
        examples=(
            OperationExample(
                name="vector_scale",
                description="Add scalar 2 to every vector coordinate; scalar and vector must share their semiring.",
                input={"scalar": _finite(2), "vector": _vector((1, 4), ("x", "y"))},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.vector.projectivize.compute",
        title="Projectivize a tropical vector",
        description="Normalize finite coordinates to min 0 or max 0 and return the exact common translation; the all-infinity vector has no projective class.",
        request_type=VectorProjectivizeRequest,
        result_type=VectorProjectivizeResult,
        run=compute_vector_projectivize,
        tags=("tropical", "vector", "projective", "exact"),
        examples=(
            OperationExample(
                name="min_plus_normalization",
                description="Translate a finite min-plus vector so its minimum coordinate is zero.",
                input={"vector": _vector((3, 5), ("x", "y"))},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.add.compute",
        title="Add formal tropical polynomials",
        description="Combine sparse formal tropical polynomials by exponentwise tropical addition.",
        request_type=PolynomialBinaryRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_add,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_add",
                description="Add two sparse tropical polynomials; terms must use one shared variable axis.",
                input={
                    "left": _poly((((0,), 1), ((1,), 3))),
                    "right": _poly((((0,), 2), ((1,), 4))),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.multiply.compute",
        title="Multiply formal tropical polynomials",
        description="Compute sparse tropical convolution exactly.",
        request_type=PolynomialBinaryRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_multiply,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_multiply",
                description="Multiply two sparse tropical polynomials; exponents are nonnegative and share an axis.",
                input={
                    "left": _poly((((0,), 0), ((1,), 1))),
                    "right": _poly((((0,), 0), ((1,), 2))),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.power.compute",
        title="Power a formal tropical polynomial",
        description=(
            "Compute a bounded nonnegative power in the sparse formal tropical "
            "polynomial semiring. The result is the canonical exponent/coefficient "
            "map; it does not compute a functional normal form."
        ),
        request_type=PolynomialPowerRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_power,
        tags=("tropical", "polynomial", "power", "exact"),
        examples=(
            OperationExample(
                name="formal_binomial_cube",
                description=(
                    "Cube 0 plus x in MIN_PLUS: the formal result has coefficients "
                    "0 at exponents 0, 1, 2, and 3."
                ),
                input={
                    "polynomial": _poly((((0,), 0), ((1,), 0))),
                    "exponent": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.substitute.compute",
        title="Substitute tropical polynomials",
        description=(
            "Apply a simultaneous map from every source variable to a sparse "
            "tropical polynomial over one explicit target axis."
        ),
        request_type=PolynomialSubstituteRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_substitute,
        tags=("tropical", "polynomial", "substitution", "exact"),
        examples=(
            OperationExample(
                name="binomial_substitution",
                description=(
                    "Substitute 0 plus x into a MIN_PLUS polynomial; provide one "
                    "image for each source variable in its declared axis order."
                ),
                input={
                    "polynomial": {
                        "semiring": _s(),
                        "variables": ["x"],
                        "terms": [
                            {"exponents": [0], "coefficient": _finite(0)},
                            {"exponents": [1], "coefficient": _finite(0)},
                        ],
                    },
                    "target_variables": ["t"],
                    "images": [
                        {
                            "semiring": _s(),
                            "variables": ["t"],
                            "terms": [
                                {"exponents": [0], "coefficient": _finite(0)},
                                {"exponents": [1], "coefficient": _finite(1)},
                            ],
                        }
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.evaluate.compute",
        title="Evaluate a tropical polynomial",
        description="Evaluate every monomial exactly and return all active exponents.",
        request_type=PolynomialEvaluateRequest,
        result_type=PolynomialEvaluateResult,
        run=compute_polynomial_evaluate,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_eval",
                description="Evaluate a sparse tropical polynomial at a labelled point; point axis must match variables.",
                input={
                    "polynomial": _poly((((0,), 0), ((1,), 1))),
                    "point": _vector((2,), ("x",)),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.active_terms.compute",
        title="Identify all active terms of a tropical polynomial",
        description=(
            "At one exact labelled point, return the tropical value and every "
            "minimizing (min-plus) or maximizing (max-plus) source monomial, "
            "with its canonical term index and exact evaluated value."
        ),
        request_type=PolynomialActiveTermsRequest,
        result_type=PolynomialActiveTermsResult,
        run=compute_polynomial_active_terms,
        tags=("tropical", "polynomial", "active-terms", "exact"),
        discovery_terms=(
            "tropical polynomial active terms",
            "minimizing monomials at a point",
            "maximizing tropical monomials",
            "tropical polynomial argmin or argmax",
        ),
        examples=(
            OperationExample(
                name="min_plus_active_tie",
                description=(
                    "At x=0 both terms of min(0, x) are active; source indices "
                    "and exponents remain explicit."
                ),
                input={
                    "polynomial": _poly((((0,), 0), ((1,), 0))),
                    "point": _vector((0,), ("x",)),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.univariate_roots.compute",
        title="Find exact univariate tropical roots",
        description="Return all finite breakpoints, slope-jump multiplicities, exact active-term ties, and the complete piecewise-linear profile of a univariate tropical polynomial.",
        request_type=UnivariateRootsRequest,
        result_type=TropicalUnivariateRootProfile,
        run=compute_univariate_roots,
        tags=("tropical", "polynomial", "roots", "exact"),
        examples=(
            OperationExample(
                name="min_plus_corner",
                description="The min-plus polynomial min(0, 1+x) has root -1 with multiplicity one.",
                input={"polynomial": _poly((((0,), 0), ((1,), 1)))},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.univariate_split_form.compute",
        title="Compute a univariate tropical split form",
        description=(
            "Return the consecutive-support polynomial with the same exact "
            "univariate tropical function. Its finite roots are encoded as "
            "tropical linear factors with slope-jump multiplicity. An integer "
            "input is promoted to QQ if rational roots require rational split "
            "coefficients; the output is functionally equivalent, not formally "
            "equal, to the source."
        ),
        request_type=UnivariateSplitFormRequest,
        result_type=TropicalPolynomial,
        run=compute_univariate_split_form,
        tags=("tropical", "polynomial", "split-form", "exact"),
        discovery_terms=(
            "tropical polynomial split form",
            "factor univariate tropical polynomial",
            "tropical roots linear factors",
        ),
        examples=(
            OperationExample(
                name="rational_root_over_integer_input",
                description=(
                    "The min-plus polynomial min(0, 1+2x) has root -1/2; its "
                    "split form is represented over QQ."
                ),
                input={
                    "polynomial": {
                        "semiring": {"convention": "MIN_PLUS", "base": "ZZ"},
                        "variables": ["x"],
                        "terms": [
                            {"exponents": [0], "coefficient": _finite(0)},
                            {"exponents": [2], "coefficient": _finite(1)},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.univariate_newton_polygon.compute",
        title="Compute a univariate tropical Newton polygon",
        description="Return the exact lower or upper coefficient hull, source terms on every face, face slopes, and finite tropical roots with horizontal-length multiplicities.",
        request_type=UnivariateNewtonPolygonRequest,
        result_type=UnivariateNewtonPolygonResult,
        run=compute_univariate_newton_polygon,
        tags=("tropical", "polynomial", "newton-polygon", "exact"),
        examples=(
            OperationExample(
                name="min_plus_newton_polygon",
                description="For min-plus terms (0,0), (1,2), (3,0), return the lower coefficient hull and its one exact root of multiplicity 3; the polynomial must be univariate.",
                input={
                    "polynomial": {
                        "semiring": _s(),
                        "variables": ["x"],
                        "terms": [
                            {"exponents": [0], "coefficient": _finite(0)},
                            {"exponents": [1], "coefficient": _finite(2)},
                            {"exponents": [3], "coefficient": _finite(0)},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.bivariate_regular_subdivision.compute",
        title="Compute a bivariate tropical regular subdivision",
        description=(
            "Compute one exact source-bound planar regular subdivision from a "
            "bivariate tropical polynomial. Min-plus uses lower lifted faces; "
            "max-plus uses upper lifted faces. The result includes the complete "
            "face-closed rational cell complex and every lifted face's source-term "
            "incidence. This bounded operation handles one polynomial only; it "
            "does not intersect hypersurfaces or solve a tropical system."
        ),
        request_type=BivariateRegularSubdivisionRequest,
        result_type=TropicalRegularSubdivision,
        run=compute_bivariate_regular_subdivision,
        tags=("tropical", "polynomial", "regular-subdivision", "exact"),
        examples=(
            OperationExample(
                name="min_plus_square_subdivision",
                description=(
                    "Lift the four exponent points of a square with height one at "
                    "(1,1) and compute the two exact lower triangles; the input "
                    "must be one bivariate polynomial with bounded rational heights."
                ),
                input={
                    "polynomial": {
                        "semiring": _s(),
                        "variables": ["x", "y"],
                        "terms": [
                            {"exponents": [0, 0], "coefficient": _finite(0)},
                            {"exponents": [0, 1], "coefficient": _finite(0)},
                            {"exponents": [1, 0], "coefficient": _finite(0)},
                            {"exponents": [1, 1], "coefficient": _finite(1)},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.bivariate_hypersurface.compute",
        title="Compute a bivariate tropical hypersurface",
        description=(
            "Return the complete exact corner locus of one bivariate tropical "
            "polynomial as rational H- and V-presentations. Each corner cell "
            "retains its active source terms, dual regular-subdivision face, "
            "incidence, and lattice edge weight. This bounded operation does "
            "not intersect multiple hypersurfaces or solve a tropical system."
        ),
        request_type=BivariateHypersurfaceRequest,
        result_type=TropicalHypersurface,
        run=compute_bivariate_hypersurface,
        tags=("tropical", "polynomial", "hypersurface", "exact"),
        examples=(
            OperationExample(
                name="min_plus_tropical_line",
                description=(
                    "The corner locus of min(0,x,y) is a vertex with three "
                    "primitive weight-one rays."
                ),
                input={
                    "polynomial": {
                        "semiring": _s(),
                        "variables": ["x", "y"],
                        "terms": [
                            {"exponents": [0, 0], "coefficient": _finite(0)},
                            {"exponents": [0, 1], "coefficient": _finite(0)},
                            {"exponents": [1, 0], "coefficient": _finite(0)},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.multiply.compute",
        title="Multiply tropical matrices",
        description="Compute exact labelled-axis tropical matrix multiplication.",
        request_type=MatrixMultiplyRequest,
        result_type=MatrixResult,
        run=compute_matrix_multiply,
        tags=("tropical", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix_product",
                description="Multiply two 2 by 2 tropical matrices; inner labelled axes must agree.",
                input={
                    "left": _matrix(
                        ((0, 1), (2, 3)),
                    ),
                    "right": _matrix(
                        ((1, 2), (3, 4)),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.power.compute",
        title="Power a tropical matrix",
        description="Compute a finite exact tropical matrix power.",
        request_type=MatrixPowerRequest,
        result_type=MatrixResult,
        run=compute_matrix_power,
        tags=("tropical", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix_power",
                description="Compute a finite power of a square tropical matrix; row and column axes must agree.",
                input={
                    "matrix": _matrix(
                        ((0, 1), (2, 3)),
                    ),
                    "exponent": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.finite_power_sum.compute",
        title="Compute a finite tropical matrix power sum",
        description="Return the exact finite sum I plus A through A^N and winning path lengths; no infinite closure is claimed.",
        request_type=MatrixFinitePowerSumRequest,
        result_type=FinitePowerSumResult,
        run=compute_finite_power_sum,
        tags=("tropical", "matrix", "finite", "exact"),
        examples=(
            OperationExample(
                name="finite_power_sum",
                description="Compute a finite tropical power sum; the matrix must be square and the bound is finite.",
                input={"matrix": _matrix(((0, 1), (2, 3))), "max_power": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.assignment_profile.compute",
        title="Compute a tropical assignment profile",
        description="Return the minimum or maximum assignment value and every tied column-index permutation for a square matrix; row and column labels may differ.",
        request_type=MatrixAssignmentRequest,
        result_type=AssignmentResult,
        run=compute_assignment,
        tags=("tropical", "matrix", "assignment", "exact"),
        examples=(
            OperationExample(
                name="assignment",
                description="Compute the minimum (MIN_PLUS) or maximum (MAX_PLUS) assignment value and all tied permutations for a square matrix; row and column labels may differ.",
                input={
                    "matrix": _matrix(
                        ((0, 4), (3, 1)),
                    )
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.minor_assignment_profiles.compute",
        title="Compute tropical minor assignment profiles",
        description="Compute every min-plus or max-plus assignment profile for selected square-minor sizes; permutations index each profile's selected columns.",
        request_type=MatrixMinorAssignmentsRequest,
        result_type=MatrixMinorAssignmentsResult,
        run=compute_minor_assignments,
        tags=("tropical", "matrix", "minors", "exact"),
        examples=(
            OperationExample(
                name="selected_minor_profiles",
                description="Return assignment profiles for every 1 by 1 and 2 by 2 minor.",
                input={"matrix": _matrix(((0, 4), (3, 1))), "sizes": [1, 2]},
            ),
        ),
    ),
)
__all__ = [
    "TOOLS",
    "compute_assignment",
    "compute_finite_power_sum",
    "compute_matrix_multiply",
    "compute_matrix_power",
    "compute_minor_assignments",
    "compute_polynomial_add",
    "compute_polynomial_evaluate",
    "compute_polynomial_multiply",
    "compute_polynomial_power",
    "compute_scalar_add",
    "compute_scalar_multiply",
    "compute_scalar_power",
    "compute_univariate_split_form",
    "compute_vector_add",
    "compute_vector_projectivize",
    "compute_vector_scale",
]
