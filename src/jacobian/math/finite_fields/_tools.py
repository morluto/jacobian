"""Finite-field catalog projections and immutable tool declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    HomogeneousFixedSubspace,
    OrbitDistribution,
    PaleyTournamentResult,
    PermutationResult,
    ProjectiveLine,
    RankResult,
    analyze_collisions,
    analyze_permutation,
    direction_rank_ledger,
    evaluate_finite_polynomial,
    fiber_partition,
    finite_map_table,
    homogeneous_fixed_subspace,
    linear_map_rank,
    orbit_distribution,
    paley_tournament,
    projective_line,
    restrict_scalars,
)
from jacobian.math.finite_fields._algebraic_set_models import (
    AffineZeroCountRequest,
    AffineZeroCountResult,
    AffineZeroSetRequest,
    AffineZeroSetResult,
    BaseChangeRequest,
    BaseChangeResult,
    ProjectiveZeroCountRequest,
    ProjectiveZeroCountResult,
    ProjectiveZeroSetRequest,
    ProjectiveZeroSetResult,
)
from jacobian.math.finite_fields._jacobian_syzygy_models import (
    JacobianSyzygyCheckRequest,
    JacobianSyzygyCheckResult,
)
from jacobian.math.finite_fields._matrix_rank_models import (
    MatrixRankRequest,
    MatrixRankResult,
)
from jacobian.math.finite_fields._models import (
    CollisionRequest,
    DirectionRankLedgerRequest,
    FiberPartitionRequest,
    FiniteMapTableRequest,
    FinitePolynomialEvaluationRequest,
    HomogeneousFixedSubspaceRequest,
    LinearMapRankRequest,
    OrbitDistributionRequest,
    PaleyTournamentRequest,
    PermutationRequest,
    ProjectiveLineRequest,
    RestrictScalarsRequest,
)

_FIELD_VALUE = FiniteFieldPresentation(
    characteristic=2,
    modulus_coefficients=(1, 1, 1),
    generator="a",
)
_FIELD: dict[str, object] = _FIELD_VALUE.model_dump(mode="json")
_ROWS: dict[str, object] = {"name": "b", "labels": ["b1", "b2"]}
_IMAGE: dict[str, object] = {"name": "image", "labels": ["y1"]}
_BASIS_AXIS: dict[str, object] = {"name": "basis", "labels": ["B1"]}


def _element(first: int, second: int) -> dict[str, object]:
    return FiniteFieldElement(
        presentation=_FIELD_VALUE,
        coordinates=(first, second),
    ).model_dump(mode="json")


def _direction(first: tuple[int, int], second: tuple[int, int]) -> dict[str, object]:
    return {
        "presentation": _FIELD,
        "axis": _ROWS,
        "coordinates": [_element(*first), _element(*second)],
    }


_ZERO = _element(0, 0)
_ONE = _element(1, 0)
_SUBSPACE: dict[str, object] = {
    "presentation": _FIELD,
    "row_axis": _ROWS,
    "column_axis": _IMAGE,
    "basis_axis": _BASIS_AXIS,
    "basis": [
        {
            "presentation": _FIELD,
            "row_axis": _ROWS,
            "column_axis": _IMAGE,
            "entries": [[_ONE], [_ZERO]],
        }
    ],
}
_DIRECTIONS = (
    _direction((0, 0), (1, 0)),
    _direction((1, 0), (0, 0)),
    _direction((1, 0), (1, 0)),
    _direction((1, 0), (0, 1)),
    _direction((1, 0), (1, 1)),
)
_PROJECTIVE_LINE: dict[str, object] = {
    "presentation": _FIELD,
    "axis": _ROWS,
    "points": list(_DIRECTIONS),
}
_F2_VALUE = FiniteFieldPresentation(
    characteristic=2,
    modulus_coefficients=(0, 1),
    generator="a",
)
_F2_ONE = FiniteFieldElement(presentation=_F2_VALUE, coordinates=(1,))
_RANK_MATRIX = AxisBoundMatrix(
    presentation=_F2_VALUE,
    row_axis=Axis(name="rows", labels=("r0", "r1")),
    column_axis=Axis(name="cols", labels=("c0", "c1")),
    entries=((_F2_ONE, _F2_ONE), (_F2_ONE, _F2_ONE)),
)


def _linear_map(rank: int) -> dict[str, object]:
    return {
        "source_axis": _BASIS_AXIS,
        "target_axis": {"name": "Res(image)", "labels": ["y1:1", "y1:a"]},
        "matrix": {"prime": 2, "entries": [[rank], [0]], "columns": 1},
    }


_LINEAR_MAPS = tuple(_linear_map(rank) for rank in (0, 1, 1, 1, 1))
_LEDGER: dict[str, object] = {
    "subspace": _SUBSPACE,
    "entries": [
        {
            "subspace": _SUBSPACE,
            "direction": direction,
            "linear_map": linear_map,
            "rank": rank,
        }
        for direction, linear_map, rank in zip(
            _DIRECTIONS, _LINEAR_MAPS, (0, 1, 1, 1, 1), strict=True
        )
    ],
}
_POLYNOMIAL_MAP: dict[str, object] = {
    "domain": _FIELD,
    "codomain": _FIELD,
    "polynomial": {
        "presentation": _FIELD,
        "variable": "x",
        "coefficients": [_ZERO, _ZERO, _ZERO, _ONE],
    },
}
_TABLE: dict[str, object] = {
    "map": _POLYNOMIAL_MAP,
    "entries": [
        [_element(0, 0), _element(0, 0)],
        [_element(1, 0), _element(1, 0)],
        [_element(0, 1), _element(1, 0)],
        [_element(1, 1), _element(1, 0)],
    ],
}
_FIXED_ACTION: dict[str, object] = {
    "variable_axis": {"name": "variables", "labels": ["x", "y"]},
    "generator_matrices": [
        {
            "prime": 2,
            "entries": [[0, 1], [1, 0]],
            "columns": 2,
        }
    ],
}
_F2_PRIME: dict[str, object] = _F2_VALUE.model_dump(mode="json")
_F2_ZERO: dict[str, object] = FiniteFieldElement(
    presentation=_F2_VALUE, coordinates=(0,)
).model_dump(mode="json")
_F2_ONE_JSON: dict[str, object] = _F2_ONE.model_dump(mode="json")
_AFFINE_AXIS_1: dict[str, object] = {"name": "vars", "labels": ["x"]}
_AFFINE_AXIS_2: dict[str, object] = {"name": "vars", "labels": ["x", "y"]}
_AFFINE_SYSTEM_SPLIT: dict[str, object] = {
    "presentation": _F2_PRIME,
    "variable_axis": _AFFINE_AXIS_1,
    "equations": [
        {
            "presentation": _F2_PRIME,
            "variable_axis": _AFFINE_AXIS_1,
            "terms": [
                {"coefficient": _F2_ONE_JSON, "exponents": [2]},
                {"coefficient": _F2_ONE_JSON, "exponents": [1]},
            ],
        }
    ],
}
_PROJECTIVE_SYSTEM_X: dict[str, object] = {
    "presentation": _F2_PRIME,
    "variable_axis": _AFFINE_AXIS_2,
    "equations": [
        {
            "presentation": _F2_PRIME,
            "variable_axis": _AFFINE_AXIS_2,
            "terms": [{"coefficient": _F2_ONE_JSON, "exponents": [1, 0]}],
        }
    ],
}
_GF4_ZERO: dict[str, object] = _element(0, 0)
_BASE_EMBEDDING_F2_TO_GF4: dict[str, object] = {
    "source": _F2_PRIME,
    "target": _FIELD,
    "generator_image": _GF4_ZERO,
}


_JACOBIAN_SYZYGY_AXIS: dict[str, object] = {
    "name": "vars",
    "labels": ["x", "y", "z"],
}


def _syzygy_term(*exponents: int) -> dict[str, object]:
    return {"coefficient": _F2_ONE_JSON, "exponents": list(exponents)}


def _syzygy_polynomial(*terms: dict[str, object]) -> dict[str, object]:
    return {
        "presentation": _F2_PRIME,
        "variable_axis": _JACOBIAN_SYZYGY_AXIS,
        "terms": list(terms),
    }


def _syzygy_zero() -> dict[str, object]:
    return _syzygy_polynomial({"coefficient": _F2_ZERO, "exponents": [0, 0, 0]})


# Graf's positive-characteristic Lipman-Zariski example: R = F2[x,y,z]/(f),
# f = z^2+x^3+y^5, Der_F2(R) generated by (0,0,1) and (y^4,x^2,0).
_GRAF_F = _syzygy_polynomial(
    _syzygy_term(0, 0, 2), _syzygy_term(3, 0, 0), _syzygy_term(0, 5, 0)
)
_GRAF_SYZYGY_REQUEST: dict[str, object] = {
    "polynomial": _GRAF_F,
    "ideal_generators": [_GRAF_F],
    "reduction_variable": "z",
    "rows": [
        [_syzygy_zero(), _syzygy_zero(), _syzygy_polynomial(_syzygy_term(0, 0, 1))],
        [
            _syzygy_polynomial(_syzygy_term(0, 4, 0)),
            _syzygy_polynomial(_syzygy_term(2, 0, 0)),
            _syzygy_zero(),
        ],
    ],
}


def _enumerate_projective_line(request: ProjectiveLineRequest) -> ProjectiveLine:
    return projective_line(request.presentation, request.axis)


def _restrict(request: RestrictScalarsRequest) -> FiniteLinearMap:
    return restrict_scalars(request.subspace, request.direction)


def _rank(request: LinearMapRankRequest) -> RankResult:
    return linear_map_rank(request.subspace, request.direction)


def _ledger(request: DirectionRankLedgerRequest) -> DirectionRankLedger:
    return direction_rank_ledger(request.subspace, request.directions)


def _orbit_distribution(request: OrbitDistributionRequest) -> OrbitDistribution:
    return orbit_distribution(request.ledger)


def _finite_map_table(request: FiniteMapTableRequest) -> FiniteMapTable:
    return finite_map_table(request.polynomial_map)


def _finite_polynomial_evaluation(
    request: FinitePolynomialEvaluationRequest,
) -> FiniteFieldElement:
    return evaluate_finite_polynomial(request.polynomial, request.value)


def _fiber_partition(request: FiberPartitionRequest) -> FiberPartition:
    return fiber_partition(request.table)


def _analyze_collisions(request: CollisionRequest) -> CollisionResult:
    return analyze_collisions(request.table)


def _analyze_permutation(request: PermutationRequest) -> PermutationResult:
    return analyze_permutation(request.table)


def _paley_tournament(request: PaleyTournamentRequest) -> PaleyTournamentResult:
    return paley_tournament(request.presentation)


def _compute_matrix_rank(request: MatrixRankRequest) -> MatrixRankResult:
    from jacobian.math.finite_fields._matrix_rank import compute_matrix_rank

    return compute_matrix_rank(request.matrix)


def _homogeneous_fixed_subspace(
    request: HomogeneousFixedSubspaceRequest,
) -> HomogeneousFixedSubspace:
    return homogeneous_fixed_subspace(request.action, request.degree)


def _affine_zero_set(request: AffineZeroSetRequest) -> AffineZeroSetResult:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic

    return AffineZeroSetResult._from_kernel(
        system=request.system, points=algebraic.affine_zero_set(request.system)
    )


def _affine_zero_count(request: AffineZeroCountRequest) -> AffineZeroCountResult:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic
    from jacobian.math.finite_fields._algebraic_set_models import AffineZeroCountResult

    return AffineZeroCountResult._from_kernel(
        system=request.system,
        point_count=algebraic.affine_zero_count(request.system),
    )


def _projective_zero_set(request: ProjectiveZeroSetRequest) -> ProjectiveZeroSetResult:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic
    from jacobian.math.finite_fields._algebraic_set_models import (
        ProjectiveZeroSetResult,
    )

    return ProjectiveZeroSetResult._from_kernel(
        system=request.system, points=algebraic.projective_zero_set(request.system)
    )


def _projective_zero_count(
    request: ProjectiveZeroCountRequest,
) -> ProjectiveZeroCountResult:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic
    from jacobian.math.finite_fields._algebraic_set_models import (
        ProjectiveZeroCountResult,
    )

    return ProjectiveZeroCountResult._from_kernel(
        system=request.system,
        point_count=algebraic.projective_zero_count(request.system),
    )


def _jacobian_syzygy_check(
    request: JacobianSyzygyCheckRequest,
) -> JacobianSyzygyCheckResult:
    from jacobian.math.finite_fields._jacobian_syzygy import check_jacobian_syzygy

    return check_jacobian_syzygy(
        polynomial=request.polynomial,
        rows=request.rows,
        ideal_generators=request.ideal_generators,
        reduction_variable=request.reduction_variable,
    )


def _base_change(request: BaseChangeRequest) -> BaseChangeResult:
    from jacobian.math.finite_fields import _algebraic_sets as algebraic
    from jacobian.math.finite_fields._algebraic_set_models import BaseChangeResult

    return BaseChangeResult._from_kernel(
        system=request.system,
        embedding=request.embedding,
        transported=algebraic.base_change_system(request.system, request.embedding),
    )


def _build_tools() -> MathTools:
    projective_line_operation = MathTool(
        operation_id="finite_field.projective_line.enumerate",
        request_type=ProjectiveLineRequest,
        result_type=ProjectiveLine,
        run=_enumerate_projective_line,
        title="Enumerate an exact finite projective line",
        description="Return every normalized direction in deterministic order.",
        tags=("finite-field", "projective"),
        examples=(
            OperationExample(
                name="projective_line_over_gf_four",
                description="Enumerate the projective line on a two-coordinate GF(4) axis.",
                input={"presentation": _FIELD, "axis": _ROWS},
            ),
        ),
    )
    restrict_operation = MathTool(
        operation_id="finite_field.restrict_scalars.compute",
        request_type=RestrictScalarsRequest,
        result_type=FiniteLinearMap,
        run=_restrict,
        title="Restrict a finite-field matrix action to its prime field",
        description="Construct the exact prime-field map B -> B^T b.",
        tags=("finite-field", "linear-map", "restriction-of-scalars"),
        examples=(
            OperationExample(
                name="one_basis_vector",
                description="Restrict a one-vector GF(4) subspace along one projective direction.",
                input={"subspace": _SUBSPACE, "direction": _DIRECTIONS[0]},
            ),
        ),
    )
    rank_operation = MathTool(
        operation_id="finite_field.linear_map.rank.compute",
        request_type=LinearMapRankRequest,
        result_type=RankResult,
        run=_rank,
        title="Compute finite linear-map rank over the prime field",
        description="Return the exact rank bound to its direction and map.",
        tags=("finite-field", "linear-map", "rank", "exact"),
        examples=(
            OperationExample(
                name="restricted_map_rank",
                description="Compute the rank of a restricted GF(4) map over GF(2).",
                input={"subspace": _SUBSPACE, "direction": _DIRECTIONS[0]},
            ),
        ),
    )
    table_operation = MathTool(
        operation_id="finite_field.polynomial_map.table.compute",
        request_type=FiniteMapTableRequest,
        result_type=FiniteMapTable,
        run=_finite_map_table,
        title="Evaluate a polynomial on its complete finite field",
        description="Return the exact domain-bound map table in canonical order.",
        tags=("finite-field", "polynomial", "map-table", "exact"),
        examples=(
            OperationExample(
                name="cubic_map_over_gf_four",
                description="Evaluate x³ on every element of GF(4).",
                input={"polynomial_map": _POLYNOMIAL_MAP},
            ),
        ),
    )
    point_evaluation_operation = MathTool(
        operation_id="finite_field.polynomial.evaluate.compute",
        request_type=FinitePolynomialEvaluationRequest,
        result_type=FiniteFieldElement,
        run=_finite_polynomial_evaluation,
        title="Evaluate a finite-field polynomial at one exact element",
        description=(
            "Return one exact value without enumerating the field. The polynomial "
            "and evaluation point must share one admitted finite-field presentation."
        ),
        tags=("finite-field", "polynomial", "evaluation", "exact"),
        examples=(
            OperationExample(
                name="cubic_map_at_zero",
                description="Evaluate x³ at the zero element of GF(4).",
                input={
                    "polynomial": _POLYNOMIAL_MAP["polynomial"],
                    "value": _ZERO,
                },
            ),
        ),
    )
    ledger_operation = MathTool(
        operation_id="finite_field.direction_rank_ledger.compute",
        request_type=DirectionRankLedgerRequest,
        result_type=DirectionRankLedger,
        run=_ledger,
        title="Compute ranks for a complete finite projective line",
        description="Return every supplied direction with its restricted map and rank.",
        tags=("finite-field", "rank", "exact"),
        examples=(
            OperationExample(
                name="complete_projective_line",
                description="Compute ranks for every direction on a GF(4) projective line.",
                input={"subspace": _SUBSPACE, "directions": _PROJECTIVE_LINE},
            ),
        ),
    )
    orbit_operation = MathTool(
        operation_id="finite_field.orbit_distribution.compute",
        request_type=OrbitDistributionRequest,
        result_type=OrbitDistribution,
        run=_orbit_distribution,
        title="Aggregate a complete direction-rank ledger",
        description="Return exact orbit-size counts bound to the full ledger.",
        tags=("finite-field", "orbit", "exact"),
        examples=(
            OperationExample(
                name="complete_rank_ledger",
                description="Aggregate a complete GF(4) direction-rank ledger.",
                input={"ledger": _LEDGER},
            ),
        ),
    )
    fiber_operation = MathTool(
        operation_id="finite_field.polynomial_map.fibers.compute",
        request_type=FiberPartitionRequest,
        result_type=FiberPartition,
        run=_fiber_partition,
        title="Partition a finite polynomial map into fibers",
        description="Return every nonempty fiber bound to the exact map table.",
        tags=("finite-field", "polynomial", "fibers", "exact"),
        examples=(
            OperationExample(
                name="cubic_map_table",
                description="Partition the table of x^3 over GF(4) into nonempty fibers.",
                input={"table": _TABLE},
            ),
        ),
    )
    collision_operation = MathTool(
        operation_id="finite_field.polynomial_map.collision.analyze",
        request_type=CollisionRequest,
        result_type=CollisionResult,
        run=_analyze_collisions,
        title="Analyze finite polynomial-map collisions",
        description="Return a collision or an exact injectivity result.",
        tags=("finite-field", "polynomial", "collision", "exact"),
        examples=(
            OperationExample(
                name="cubic_map_table",
                description="Find a collision in the table of x^3 over GF(4).",
                input={"table": _TABLE},
            ),
        ),
    )
    permutation_operation = MathTool(
        operation_id="finite_field.polynomial_map.permutation.analyze",
        request_type=PermutationRequest,
        result_type=PermutationResult,
        run=_analyze_permutation,
        title="Analyze a finite polynomial permutation",
        description="Return an inverse table or an exact non-permutation result.",
        tags=("finite-field", "polynomial", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cubic_map_table",
                description="Determine whether x^3 permutes GF(4).",
                input={"table": _TABLE},
            ),
        ),
    )
    paley_tournament_operation = MathTool(
        operation_id="finite_field.paley_tournament.construct",
        request_type=PaleyTournamentRequest,
        result_type=PaleyTournamentResult,
        run=_paley_tournament,
        title="Construct a finite-field Paley tournament",
        description=(
            "Return the complete directed tournament on the presentation's "
            "power-basis encoding, with x -> y exactly when y - x is a nonzero square."
        ),
        tags=("finite-field", "graph", "tournament", "quadratic-residue", "exact"),
        examples=(
            OperationExample(
                name="paley_tournament_over_f3",
                description="Construct the directed three-cycle from the canonical F_3 presentation.",
                input={
                    "presentation": FiniteFieldPresentation(
                        characteristic=3,
                        modulus_coefficients=(0, 1),
                        generator="a",
                    ).model_dump(mode="json")
                },
            ),
        ),
    )
    matrix_rank_operation = MathTool(
        operation_id="finite_field.matrix.rank.compute",
        title="Compute exact rank of a labelled matrix over its presented field",
        description=(
            "Given one AxisBoundMatrix bound to a FiniteFieldPresentation, return its "
            "exact rank over that field with deterministic row and column pivot labels. "
            "Supports both prime and extension fields."
        ),
        request_type=MatrixRankRequest,
        result_type=MatrixRankResult,
        run=_compute_matrix_rank,
        tags=("finite-field", "matrix", "rank", "exact"),
        examples=(
            OperationExample(
                name="rank_one_over_f2",
                description="Rank [[1,1],[1,1]] over F_2 is 1; the matrix must use one consistent field presentation.",
                input={"matrix": _RANK_MATRIX.model_dump(mode="json")},
            ),
        ),
    )
    fixed_subspace_operation = MathTool(
        operation_id="finite_field.prime_linear_action.homogeneous_fixed_subspace.compute",
        title="Compute a homogeneous fixed subspace of a prime-field linear action",
        description=(
            "Given explicit invertible generator matrices on a variable axis and "
            "a homogeneous degree, return the simultaneous fixed subspace in "
            "canonical monomial coordinates."
        ),
        request_type=HomogeneousFixedSubspaceRequest,
        result_type=HomogeneousFixedSubspace,
        run=_homogeneous_fixed_subspace,
        tags=("finite-field", "linear-action", "fixed-subspace", "exact"),
        examples=(
            OperationExample(
                name="quadratic_swap_fixed_subspace",
                description="Compute the quadratic fixed subspace for the coordinate-swap action over F_2.",
                input={"action": _FIXED_ACTION, "degree": 2},
            ),
        ),
    )
    affine_zero_operation = MathTool(
        operation_id="finite_field.affine_zero_set.compute",
        title="Compute an affine zero set over a finite field",
        description=(
            "Return all and only simultaneous zeros of a supplied affine polynomial "
            "system in canonical coordinate order with an agreeing count; every "
            "ambient point, evaluation, and output is preflight-bounded."
        ),
        request_type=AffineZeroSetRequest,
        result_type=AffineZeroSetResult,
        run=_affine_zero_set,
        tags=("finite-field", "algebraic-set", "affine", "exact", "complete"),
        examples=(
            OperationExample(
                name="split_quadratic_over_f2",
                description=(
                    "Zeros of x^2+x over F_2 are 0 and 1; the system must share one "
                    "presentation and variable axis."
                ),
                input={"system": _AFFINE_SYSTEM_SPLIT},
            ),
        ),
    )
    affine_count_operation = MathTool(
        operation_id="finite_field.affine_zero_count.compute",
        title="Count an affine zero set over a finite field",
        description=(
            "Return the compact point count agreeing exactly with the complete affine "
            "enumeration; the system must share one presentation and variable axis."
        ),
        request_type=AffineZeroCountRequest,
        result_type=AffineZeroCountResult,
        run=_affine_zero_count,
        tags=("finite-field", "algebraic-set", "affine", "count", "exact"),
        examples=(
            OperationExample(
                name="split_quadratic_count_over_f2",
                description=(
                    "Count zeros of x^2+x over F_2; the system must share one "
                    "presentation and variable axis."
                ),
                input={"system": _AFFINE_SYSTEM_SPLIT},
            ),
        ),
    )
    projective_zero_operation = MathTool(
        operation_id="finite_field.projective_zero_set.compute",
        title="Compute a projective zero set over a finite field",
        description=(
            "Return canonical scalar-class representatives of the zeros of a "
            "homogeneous system; only homogeneous systems are admitted and every "
            "ambient state is preflight-bounded."
        ),
        request_type=ProjectiveZeroSetRequest,
        result_type=ProjectiveZeroSetResult,
        run=_projective_zero_set,
        tags=("finite-field", "algebraic-set", "projective", "exact", "complete"),
        examples=(
            OperationExample(
                name="vanishing_x_over_f2",
                description=(
                    "Projective zeros of x over F_2 are the single class [0:1]; the "
                    "system must be homogeneous."
                ),
                input={"system": _PROJECTIVE_SYSTEM_X},
            ),
        ),
    )
    projective_count_operation = MathTool(
        operation_id="finite_field.projective_zero_count.compute",
        title="Count a projective zero set over a finite field",
        description=(
            "Return the compact projective count agreeing exactly with the complete "
            "scalar-class enumeration; the system must be homogeneous."
        ),
        request_type=ProjectiveZeroCountRequest,
        result_type=ProjectiveZeroCountResult,
        run=_projective_zero_count,
        tags=("finite-field", "algebraic-set", "projective", "count", "exact"),
        examples=(
            OperationExample(
                name="vanishing_x_count_over_f2",
                description=(
                    "Count projective zeros of x over F_2; the system must be "
                    "homogeneous."
                ),
                input={"system": _PROJECTIVE_SYSTEM_X},
            ),
        ),
    )
    base_change_operation = MathTool(
        operation_id="finite_field.algebraic_set.base_change.compute",
        title="Transport an algebraic set along a field embedding",
        description=(
            "Transport a polynomial system along an explicit exact field embedding, "
            "checking the generator root relation; the system must use the embedding "
            "source and characteristics must agree."
        ),
        request_type=BaseChangeRequest,
        result_type=BaseChangeResult,
        run=_base_change,
        tags=("finite-field", "algebraic-set", "base-change", "exact"),
        examples=(
            OperationExample(
                name="split_quadratic_f2_to_gf4",
                description=(
                    "Transport x^2+x from F_2 to GF(4); the embedding must carry an "
                    "explicit generator image satisfying the source modulus."
                ),
                input={
                    "system": _AFFINE_SYSTEM_SPLIT,
                    "embedding": _BASE_EMBEDDING_F2_TO_GF4,
                },
            ),
        ),
    )
    jacobian_syzygy_operation = MathTool(
        operation_id="finite_field.jacobian_syzygy.check",
        title="Check finite-field Jacobian syzygy rows in a quotient ring",
        description=(
            "Derive the characteristic-p Jacobian of f in GF(p)[x_1,...,x_n] "
            "itself and check candidate syzygy rows: VERIFIED exactly when every "
            "row r satisfies sum_i r_i * (df/dx_i) == 0 in GF(p)[x]/I, with a "
            "per-row normal-form remainder and reduction-step ledger; otherwise "
            "REJECTED at the first failing row with its nonzero remainder. "
            "Callers supply rows only, never a gradient claim, so a "
            "wrong-characteristic Jacobian cannot slip through. Admitted "
            "reduction regime (V1, no general Groebner engine): I is empty (the "
            "plain polynomial ring) or principal with a generator monic in one "
            "declared reduction variable, giving unique normal forms by exact "
            "division; presentations outside this regime are rejected as typed "
            "admission errors. Envelope: p <= 65536, n <= 4 variables, 64 terms "
            "per polynomial, degree 32 per variable, 16 rows, and a bounded "
            "reduction work budget checked before any reduction runs."
        ),
        request_type=JacobianSyzygyCheckRequest,
        result_type=JacobianSyzygyCheckResult,
        run=_jacobian_syzygy_check,
        tags=(
            "finite-field",
            "jacobian",
            "syzygy",
            "quotient-ring",
            "characteristic-p",
            "exact",
        ),
        discovery_terms=(
            "Jacobian syzygy",
            "finite field syzygy module",
            "characteristic p differentiation",
            "quotient ring normal form",
            "Lipman-Zariski",
            "Der module generators",
            "monic principal reduction",
        ),
        examples=(
            OperationExample(
                name="graf_char_two_syzygy_generators",
                description=(
                    "Graf's F2 example: R=F2[x,y,z]/(z^2+x^3+y^5) with char-2 "
                    "Jacobian (x^2,y^4,0); Der rows (0,0,1) and (y^4,x^2,0) both "
                    "verify as syzygies."
                ),
                input=_GRAF_SYZYGY_REQUEST,
            ),
        ),
    )
    return (
        projective_line_operation,
        matrix_rank_operation,
        restrict_operation,
        rank_operation,
        ledger_operation,
        orbit_operation,
        point_evaluation_operation,
        table_operation,
        fiber_operation,
        collision_operation,
        permutation_operation,
        paley_tournament_operation,
        fixed_subspace_operation,
        affine_zero_operation,
        affine_count_operation,
        projective_zero_operation,
        projective_count_operation,
        base_change_operation,
        jacobian_syzygy_operation,
    )


TOOLS: MathTools = _build_tools()

__all__ = ["TOOLS"]
