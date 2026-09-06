"""Finite-field catalog projections and immutable tool declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.finite_fields import (
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
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
    fiber_partition,
    finite_map_table,
    homogeneous_fixed_subspace,
    linear_map_rank,
    orbit_distribution,
    paley_tournament,
    projective_line,
    restrict_scalars,
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
    HomogeneousFixedSubspaceRequest,
    LinearMapRankRequest,
    OrbitDistributionRequest,
    PaleyTournamentRequest,
    PermutationRequest,
    ProjectiveLineRequest,
    RestrictScalarsRequest,
)

_FIELD: dict[str, object] = {
    "characteristic": 2,
    "modulus_coefficients": [1, 1, 1],
    "generator": "a",
}
_ROWS: dict[str, object] = {"name": "b", "labels": ["b1", "b2"]}
_IMAGE: dict[str, object] = {"name": "image", "labels": ["y1"]}
_BASIS_AXIS: dict[str, object] = {"name": "basis", "labels": ["B1"]}


def _element(first: int, second: int) -> dict[str, object]:
    return {"presentation": _FIELD, "coordinates": [first, second]}


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
                    "presentation": {
                        "characteristic": 3,
                        "modulus_coefficients": [0, 1],
                        "generator": "a",
                    }
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
                input={
                    "matrix": {
                        "presentation": {
                            "characteristic": 2,
                            "modulus_coefficients": [0, 1],
                            "generator": "a",
                        },
                        "row_axis": {"name": "rows", "labels": ["r0", "r1"]},
                        "column_axis": {"name": "cols", "labels": ["c0", "c1"]},
                        "entries": [
                            [
                                {
                                    "presentation": {
                                        "characteristic": 2,
                                        "modulus_coefficients": [0, 1],
                                        "generator": "a",
                                    },
                                    "coordinates": [1],
                                },
                                {
                                    "presentation": {
                                        "characteristic": 2,
                                        "modulus_coefficients": [0, 1],
                                        "generator": "a",
                                    },
                                    "coordinates": [1],
                                },
                            ],
                            [
                                {
                                    "presentation": {
                                        "characteristic": 2,
                                        "modulus_coefficients": [0, 1],
                                        "generator": "a",
                                    },
                                    "coordinates": [1],
                                },
                                {
                                    "presentation": {
                                        "characteristic": 2,
                                        "modulus_coefficients": [0, 1],
                                        "generator": "a",
                                    },
                                    "coordinates": [1],
                                },
                            ],
                        ],
                    }
                },
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
    return (
        projective_line_operation,
        matrix_rank_operation,
        restrict_operation,
        rank_operation,
        ledger_operation,
        orbit_operation,
        table_operation,
        fiber_operation,
        collision_operation,
        permutation_operation,
        paley_tournament_operation,
        fixed_subspace_operation,
    )


TOOLS: MathTools = _build_tools()

__all__ = ["TOOLS"]
