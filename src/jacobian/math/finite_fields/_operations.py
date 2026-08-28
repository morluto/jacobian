"""Finite-field operation declarations over authoritative native values."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.finite_fields import (
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
    FiniteLinearMap,
    FiniteMapTable,
    OrbitDistribution,
    PermutationResult,
    ProjectiveLine,
    RankResult,
    analyze_collisions,
    analyze_permutation,
    direction_rank_ledger,
    fiber_partition,
    finite_map_table,
    linear_map_rank,
    orbit_distribution,
    projective_line,
    restrict_scalars,
)
from jacobian.math.finite_fields._models import (
    CollisionRequest,
    DirectionRankLedgerRequest,
    FiberPartitionRequest,
    FiniteMapTableRequest,
    LinearMapRankRequest,
    OrbitDistributionRequest,
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


def finite_field_operations() -> MathTools:
    projective_line_operation = MathTool(
        operation_id="finite_field.projective_line.enumerate",
        request_type=ProjectiveLineRequest,
        result_type=ProjectiveLine,
        run=_enumerate_projective_line,
        title="Enumerate an exact finite projective line",
        description="Return every normalized direction in deterministic order.",
        tags=("finite-field", "projective"),
        examples=(
            example(
                "projective_line_over_gf_four",
                "Enumerate the projective line on a two-coordinate GF(4) axis.",
                {"presentation": _FIELD, "axis": _ROWS},
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
            example(
                "one_basis_vector",
                "Restrict a one-vector GF(4) subspace along one projective direction.",
                {"subspace": _SUBSPACE, "direction": _DIRECTIONS[0]},
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
            example(
                "restricted_map_rank",
                "Compute the rank of a restricted GF(4) map over GF(2).",
                {"subspace": _SUBSPACE, "direction": _DIRECTIONS[0]},
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
            example(
                "cubic_map_over_gf_four",
                "Evaluate x³ on every element of GF(4).",
                {"polynomial_map": _POLYNOMIAL_MAP},
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
            example(
                "complete_projective_line",
                "Compute ranks for every direction on a GF(4) projective line.",
                {"subspace": _SUBSPACE, "directions": _PROJECTIVE_LINE},
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
            example(
                "complete_rank_ledger",
                "Aggregate a complete GF(4) direction-rank ledger.",
                {"ledger": _LEDGER},
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
            example(
                "cubic_map_table",
                "Partition the table of x^3 over GF(4) into nonempty fibers.",
                {"table": _TABLE},
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
            example(
                "cubic_map_table",
                "Find a collision in the table of x^3 over GF(4).",
                {"table": _TABLE},
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
            example(
                "cubic_map_table",
                "Determine whether x^3 permutes GF(4).",
                {"table": _TABLE},
            ),
        ),
    )
    return (
        projective_line_operation,
        restrict_operation,
        rank_operation,
        ledger_operation,
        orbit_operation,
        table_operation,
        fiber_operation,
        collision_operation,
        permutation_operation,
    )


__all__ = ["finite_field_operations"]
