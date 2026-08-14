"""Finite-field operation declarations over authoritative native values."""

from jacobian.domains.finite_fields.contracts import (
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
from jacobian.math.finite_fields import (
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
    FiniteDimensionalSubspace,
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
from jacobian.math_tools import MathTool, MathTools

_MAX_PROJECTIVE_POINTS = 4096
_MAX_FINITE_MAP_ELEMENTS = 4096
_MAX_FINITE_MAP_WORK = 1_000_000
_MAX_FINITE_MAP_REPLAY_WORK = 1_000_000
_MAX_DIRECTION_RANK_WORK = 1_000_000


def _enumerate_projective_line(request: ProjectiveLineRequest) -> ProjectiveLine:
    count = (request.presentation.order ** len(request.axis.labels) - 1) // (
        request.presentation.order - 1
    )
    if count > _MAX_PROJECTIVE_POINTS:
        raise ValueError(
            f"projective line has {count} directions; limit is {_MAX_PROJECTIVE_POINTS}"
        )
    return projective_line(request.presentation, request.axis)


def _restrict(request: RestrictScalarsRequest) -> FiniteLinearMap:
    return restrict_scalars(request.subspace, request.direction)


def _rank(request: LinearMapRankRequest) -> RankResult:
    return linear_map_rank(request.direction, request.linear_map)


def _ledger(request: DirectionRankLedgerRequest) -> DirectionRankLedger:
    _require_direction_rank_work(request.subspace, len(request.directions.points))
    return direction_rank_ledger(request.subspace, request.directions)


def _direction_rank_work(
    subspace: FiniteDimensionalSubspace,
    direction_count: int,
) -> int:
    source_dimension = len(subspace.basis)
    target_dimension = len(subspace.column_axis.labels) * subspace.presentation.degree
    restriction_work = (
        source_dimension * len(subspace.row_axis.labels) * target_dimension
    )
    rank_work = (
        target_dimension * source_dimension * min(target_dimension, source_dimension)
    )
    return direction_count * (restriction_work + rank_work)


def _require_direction_rank_work(
    subspace: FiniteDimensionalSubspace,
    direction_count: int,
) -> None:
    work = _direction_rank_work(subspace, direction_count)
    if work > _MAX_DIRECTION_RANK_WORK:
        raise ValueError(
            f"direction-rank work is {work}; limit is {_MAX_DIRECTION_RANK_WORK}",
        )


def _orbit_distribution(request: OrbitDistributionRequest) -> OrbitDistribution:
    _require_direction_rank_work(request.ledger.subspace, len(request.ledger.entries))
    return orbit_distribution(request.ledger)


def _finite_map_table(request: FiniteMapTableRequest) -> FiniteMapTable:
    polynomial_map = request.polynomial_map
    count = polynomial_map.domain.order
    if count > _MAX_FINITE_MAP_ELEMENTS:
        raise ValueError(
            f"finite map has {count} inputs; limit is {_MAX_FINITE_MAP_ELEMENTS}"
        )
    work = (
        count
        * len(polynomial_map.polynomial.coefficients)
        * polynomial_map.domain.degree
    )
    if work > _MAX_FINITE_MAP_WORK:
        raise ValueError(f"finite map work is {work}; limit is {_MAX_FINITE_MAP_WORK}")
    return finite_map_table(request.polynomial_map)


def _fiber_partition(request: FiberPartitionRequest) -> FiberPartition:
    _require_finite_map_replay_work(request)
    return fiber_partition(request.table)


def _analyze_collisions(request: CollisionRequest) -> CollisionResult:
    _require_finite_map_replay_work(request)
    return analyze_collisions(request.table)


def _analyze_permutation(request: PermutationRequest) -> PermutationResult:
    _require_finite_map_replay_work(request)
    return analyze_permutation(request.table)


def _require_finite_map_replay_work(
    request: FiberPartitionRequest | CollisionRequest | PermutationRequest,
) -> None:
    table = request.table
    work = (
        len(table.entries)
        * len(table.map.polynomial.coefficients)
        * table.map.domain.degree
    )
    if work > _MAX_FINITE_MAP_REPLAY_WORK:
        raise ValueError(
            f"finite map replay work is {work}; limit is {_MAX_FINITE_MAP_REPLAY_WORK}",
        )


def finite_field_operations() -> MathTools:
    projective_line_operation = MathTool(
        operation_id="finite_field.projective_line.enumerate",
        version="1",
        request_type=ProjectiveLineRequest,
        result_type=ProjectiveLine,
        run=_enumerate_projective_line,
        title="Enumerate an exact finite projective line",
        description="Return every normalized direction in deterministic order.",
        tags=("finite-field", "projective"),
    )
    restrict_operation = MathTool(
        operation_id="finite_field.restrict_scalars.compute",
        version="1",
        request_type=RestrictScalarsRequest,
        result_type=FiniteLinearMap,
        run=_restrict,
        title="Restrict a finite-field matrix action to its prime field",
        description="Construct the exact prime-field map B -> B^T b.",
        tags=("finite-field", "linear-map", "restriction-of-scalars"),
    )
    rank_operation = MathTool(
        operation_id="finite_field.linear_map.rank.compute",
        version="1",
        request_type=LinearMapRankRequest,
        result_type=RankResult,
        run=_rank,
        title="Compute finite linear-map rank over the prime field",
        description="Return the exact rank bound to its direction and map.",
        tags=("finite-field", "linear-map", "rank", "exact"),
    )
    ledger_operation = MathTool(
        operation_id="finite_field.direction_rank_ledger.compute",
        version="1",
        request_type=DirectionRankLedgerRequest,
        result_type=DirectionRankLedger,
        run=_ledger,
        title="Compute ranks for a complete finite projective line",
        description="Return every direction with its restricted map and rank.",
        tags=("finite-field", "rank"),
    )
    orbit_operation = MathTool(
        operation_id="finite_field.orbit_distribution.compute",
        version="1",
        request_type=OrbitDistributionRequest,
        result_type=OrbitDistribution,
        run=_orbit_distribution,
        title="Aggregate a complete direction-rank ledger",
        description="Return exact orbit-size counts bound to the full ledger.",
        tags=("finite-field", "orbit"),
    )
    table_operation = MathTool(
        operation_id="finite_field.polynomial_map.table.compute",
        version="1",
        request_type=FiniteMapTableRequest,
        result_type=FiniteMapTable,
        run=_finite_map_table,
        title="Evaluate a polynomial on its complete finite field",
        description="Return the exact domain-bound map table in canonical order.",
        tags=("finite-field", "polynomial", "map-table", "exact"),
    )
    fiber_operation = MathTool(
        operation_id="finite_field.polynomial_map.fibers.compute",
        version="1",
        request_type=FiberPartitionRequest,
        result_type=FiberPartition,
        run=_fiber_partition,
        title="Partition a finite polynomial map into fibers",
        description="Return every nonempty fiber bound to the exact map table.",
        tags=("finite-field", "polynomial", "fibers", "exact"),
    )
    collision_operation = MathTool(
        operation_id="finite_field.polynomial_map.collision.analyze",
        version="1",
        request_type=CollisionRequest,
        result_type=CollisionResult,
        run=_analyze_collisions,
        title="Analyze finite polynomial-map collisions",
        description="Return a collision or an exact injectivity result.",
        tags=("finite-field", "polynomial", "collision"),
    )
    permutation_operation = MathTool(
        operation_id="finite_field.polynomial_map.permutation.analyze",
        version="1",
        request_type=PermutationRequest,
        result_type=PermutationResult,
        run=_analyze_permutation,
        title="Analyze a finite polynomial permutation",
        description="Return an inverse table or an exact non-permutation result.",
        tags=("finite-field", "polynomial", "permutation"),
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
