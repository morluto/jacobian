"""Installed finite-field operations over the authoritative native values."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.finite_fields.checkers import (
    FINITE_FIELD_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.finite_fields.contracts import (
    CollisionCertificateRequest,
    DirectionRankLedgerRequest,
    FiberPartitionRequest,
    FiniteMapTableRequest,
    LinearMapRankRequest,
    OrbitDistributionRequest,
    PermutationCertificateRequest,
    ProjectiveLineRequest,
    RestrictScalarsRequest,
)
from jacobian.math.finite_fields import (
    Axis,
    CollisionCertificate,
    DirectionRankLedger,
    FiberPartition,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    FinitePolynomialMap,
    OrbitDistribution,
    PermutationCertificate,
    ProjectiveLine,
    ProjectivePoint,
    RankResult,
    collision_certificate,
    direction_rank_ledger,
    fiber_partition,
    finite_map_table,
    linear_map_rank,
    orbit_distribution,
    permutation_certificate,
    projective_line,
    restrict_scalars,
)
from jacobian.math.finite_fields.operations import (
    _InvalidDirectionRankLedgerError,
    _InvalidFiniteMapTableError,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operation_ports import InputPort, OutputPort
from jacobian.operations import (
    SUPPORTED,
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
    OperationRefusalError,
    OperationSpec,
    PreflightResult,
    PreflightStatus,
)
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.providers.flint_runtime import python_flint_finite_field_provider_runtime

_MAX_PROJECTIVE_POINTS = 4096
_MAX_FINITE_MAP_ELEMENTS = 4096
_MAX_FINITE_MAP_WORK = 1_000_000
_MAX_FINITE_MAP_REPLAY_WORK = 1_000_000
_MAX_DIRECTION_RANK_WORK = 1_000_000


def _enumerate_projective_line(request: ProjectiveLineRequest) -> ProjectiveLine:
    return projective_line(request.presentation, request.axis)


def _projective_line_preflight(request: ProjectiveLineRequest) -> PreflightResult:
    count = (request.presentation.order ** len(request.axis.labels) - 1) // (
        request.presentation.order - 1
    )
    if count > _MAX_PROJECTIVE_POINTS:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"projective line has {count} directions; limit is {_MAX_PROJECTIVE_POINTS}",
        )
    return SUPPORTED


def _restrict(request: RestrictScalarsRequest) -> FiniteLinearMap:
    return restrict_scalars(request.subspace, request.direction)


def _rank(request: LinearMapRankRequest) -> RankResult:
    return linear_map_rank(request.direction, request.linear_map)


def _ledger(request: DirectionRankLedgerRequest) -> DirectionRankLedger:
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


def _direction_rank_preflight(
    request: DirectionRankLedgerRequest,
) -> PreflightResult:
    work = _direction_rank_work(request.subspace, len(request.directions.points))
    if work > _MAX_DIRECTION_RANK_WORK:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"direction-rank work is {work}; limit is {_MAX_DIRECTION_RANK_WORK}",
        )
    return SUPPORTED


def _orbit_distribution(request: OrbitDistributionRequest) -> OrbitDistribution:
    try:
        return orbit_distribution(request.ledger)
    except _InvalidDirectionRankLedgerError as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="INVALID_DIRECTION_RANK_LEDGER",
                stage="direction_rank_ledger_validation",
                message=str(exc),
                hint="Use a ledger computed from the exact bound subspace.",
            )
        ) from exc


def _orbit_replay_preflight(request: OrbitDistributionRequest) -> PreflightResult:
    ledger = request.ledger
    work = _direction_rank_work(ledger.subspace, len(ledger.entries))
    if work > _MAX_DIRECTION_RANK_WORK:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"orbit replay work is {work}; limit is {_MAX_DIRECTION_RANK_WORK}",
        )
    return SUPPORTED


def _finite_map_table(request: FiniteMapTableRequest) -> FiniteMapTable:
    return finite_map_table(request.polynomial_map)


def _finite_map_preflight(request: FiniteMapTableRequest) -> PreflightResult:
    polynomial_map = request.polynomial_map
    count = polynomial_map.domain.order
    if count > _MAX_FINITE_MAP_ELEMENTS:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"finite map has {count} inputs; limit is {_MAX_FINITE_MAP_ELEMENTS}",
        )
    work = (
        count
        * len(polynomial_map.polynomial.coefficients)
        * polynomial_map.domain.degree
    )
    if work > _MAX_FINITE_MAP_WORK:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"finite map work is {work}; limit is {_MAX_FINITE_MAP_WORK}",
        )
    return SUPPORTED


def _fiber_partition(request: FiberPartitionRequest) -> FiberPartition:
    try:
        return fiber_partition(request.table)
    except _InvalidFiniteMapTableError as exc:
        raise _finite_map_table_refusal(exc) from exc


def _collision_certificate(
    request: CollisionCertificateRequest,
) -> CollisionCertificate:
    try:
        return collision_certificate(request.table)
    except _InvalidFiniteMapTableError as exc:
        raise _finite_map_table_refusal(exc) from exc
    except ValueError as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="FINITE_MAP_HAS_NO_COLLISION",
                stage="finite_map_collision",
                message=str(exc),
                hint="Use a non-injective polynomial map.",
            )
        ) from exc


def _permutation_certificate(
    request: PermutationCertificateRequest,
) -> PermutationCertificate:
    try:
        return permutation_certificate(request.table)
    except _InvalidFiniteMapTableError as exc:
        raise _finite_map_table_refusal(exc) from exc
    except ValueError as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="FINITE_MAP_NOT_PERMUTATION",
                stage="finite_map_permutation",
                message=str(exc),
                hint="Use an injective polynomial map.",
            )
        ) from exc


def _finite_map_table_refusal(error: ValueError) -> OperationRefusalError:
    return OperationRefusalError(
        CapabilityDiagnostic(
            code="INVALID_FINITE_MAP_TABLE",
            stage="finite_map_table_validation",
            message=str(error),
            hint="Use the complete table produced by the bound polynomial map.",
        )
    )


def _finite_map_replay_preflight(
    request: FiberPartitionRequest
    | CollisionCertificateRequest
    | PermutationCertificateRequest,
) -> PreflightResult:
    table = request.table
    work = (
        len(table.entries)
        * len(table.map.polynomial.coefficients)
        * table.map.domain.degree
    )
    if work > _MAX_FINITE_MAP_REPLAY_WORK:
        return PreflightResult(
            PreflightStatus.RESOURCE_LIMIT_EXCEEDED,
            f"finite map replay work is {work}; limit is {_MAX_FINITE_MAP_REPLAY_WORK}",
        )
    return SUPPORTED


def build_finite_field_bundle() -> DomainBundle:
    provider = known_provider_runtime(
        "jacobian.sympy",
        features=(
            "finite-field-presentation",
            "finite-map-replay",
            "projective-normalization",
            "projective-enumeration",
        ),
    )
    flint_provider = python_flint_finite_field_provider_runtime()
    projective_line_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.projective_line.enumerate",
            version="1",
            request_type=ProjectiveLineRequest,
            result_type=ProjectiveLine,
            execute=_enumerate_projective_line,
            preflight=_projective_line_preflight,
            title="Enumerate an exact finite projective line",
            description="Return every normalized direction in deterministic order.",
            tags=("finite-field", "projective"),
        ),
        input_ports=(
            InputPort(
                name="presentation",
                value_type=FiniteFieldPresentation,
                request_field="presentation",
            ),
            InputPort(name="axis", value_type=Axis, request_field="axis"),
        ),
        output_ports=(OutputPort(name="directions", value_type=ProjectiveLine),),
    )
    restrict_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.restrict_scalars.compute",
            version="1",
            request_type=RestrictScalarsRequest,
            result_type=FiniteLinearMap,
            execute=_restrict,
            title="Restrict a finite-field matrix action to its prime field",
            description="Construct the exact prime-field map B -> B^T b.",
            tags=("finite-field", "linear-map", "restriction-of-scalars"),
        ),
        provider_runtime=flint_provider,
        input_ports=(
            InputPort(
                name="subspace",
                value_type=FiniteDimensionalSubspace,
                request_field="subspace",
            ),
            InputPort(
                name="direction",
                value_type=ProjectivePoint,
                request_field="direction",
            ),
        ),
        output_ports=(OutputPort(name="linear_map", value_type=FiniteLinearMap),),
    )
    rank_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.linear_map.rank.compute",
            version="1",
            request_type=LinearMapRankRequest,
            result_type=RankResult,
            execute=_rank,
            title="Compute finite linear-map rank over the prime field",
            description="Return the exact rank bound to its direction and map.",
            tags=("finite-field", "linear-map", "rank", "exact"),
        ),
        provider_runtime=flint_provider,
        input_ports=(
            InputPort(
                name="direction",
                value_type=ProjectivePoint,
                request_field="direction",
            ),
            InputPort(
                name="linear_map",
                value_type=FiniteLinearMap,
                request_field="linear_map",
            ),
        ),
        output_ports=(OutputPort(name="rank", value_type=RankResult),),
    )
    ledger_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.direction_rank_ledger.compute",
            version="1",
            request_type=DirectionRankLedgerRequest,
            result_type=DirectionRankLedger,
            execute=_ledger,
            preflight=_direction_rank_preflight,
            title="Compute ranks for a complete finite projective line",
            description="Return every direction with its restricted map and rank.",
            tags=("finite-field", "rank"),
        ),
        provider_runtime=flint_provider,
        input_ports=(
            InputPort(
                name="subspace",
                value_type=FiniteDimensionalSubspace,
                request_field="subspace",
            ),
            InputPort(
                name="directions",
                value_type=ProjectiveLine,
                request_field="directions",
            ),
        ),
        output_ports=(OutputPort(name="ledger", value_type=DirectionRankLedger),),
    )
    orbit_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.orbit_distribution.compute",
            version="1",
            request_type=OrbitDistributionRequest,
            result_type=OrbitDistribution,
            execute=_orbit_distribution,
            preflight=_orbit_replay_preflight,
            title="Aggregate a complete direction-rank ledger",
            description="Return exact orbit-size counts bound to the full ledger.",
            tags=("finite-field", "orbit"),
        ),
        provider_runtime=flint_provider,
        input_ports=(
            InputPort(
                name="ledger",
                value_type=DirectionRankLedger,
                request_field="ledger",
            ),
        ),
        output_ports=(OutputPort(name="distribution", value_type=OrbitDistribution),),
    )
    table_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.polynomial_map.table.compute",
            version="1",
            request_type=FiniteMapTableRequest,
            result_type=FiniteMapTable,
            execute=_finite_map_table,
            preflight=_finite_map_preflight,
            title="Evaluate a polynomial on its complete finite field",
            description="Return the exact domain-bound map table in canonical order.",
            tags=("finite-field", "polynomial", "map-table", "exact"),
        ),
        provider_runtime=flint_provider,
        input_ports=(
            InputPort(
                name="polynomial_map",
                value_type=FinitePolynomialMap,
                request_field="polynomial_map",
            ),
        ),
        output_ports=(OutputPort(name="table", value_type=FiniteMapTable),),
    )
    fiber_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.polynomial_map.fibers.compute",
            version="1",
            request_type=FiberPartitionRequest,
            result_type=FiberPartition,
            execute=_fiber_partition,
            preflight=_finite_map_replay_preflight,
            title="Partition a finite polynomial map into fibers",
            description="Return every nonempty fiber bound to the exact map table.",
            tags=("finite-field", "polynomial", "fibers", "exact"),
        ),
        provider_runtime=provider,
        input_ports=(
            InputPort(name="table", value_type=FiniteMapTable, request_field="table"),
        ),
        output_ports=(OutputPort(name="fibers", value_type=FiberPartition),),
    )
    collision_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.polynomial_map.collision.compute",
            version="1",
            request_type=CollisionCertificateRequest,
            result_type=CollisionCertificate,
            execute=_collision_certificate,
            preflight=_finite_map_replay_preflight,
            title="Extract a finite polynomial-map collision",
            description="Return two distinct inputs with the same exact table image.",
            tags=("finite-field", "polynomial", "collision", "certificate"),
        ),
        provider_runtime=provider,
        input_ports=(
            InputPort(name="table", value_type=FiniteMapTable, request_field="table"),
        ),
        output_ports=(OutputPort(name="collision", value_type=CollisionCertificate),),
    )
    permutation_operation = inline_operation(
        OperationSpec(
            operation_id="finite_field.polynomial_map.permutation.compute",
            version="1",
            request_type=PermutationCertificateRequest,
            result_type=PermutationCertificate,
            execute=_permutation_certificate,
            preflight=_finite_map_replay_preflight,
            title="Certify a finite polynomial permutation",
            description="Return the exact inverse table of an injective finite map.",
            tags=("finite-field", "polynomial", "permutation", "certificate"),
        ),
        provider_runtime=provider,
        input_ports=(
            InputPort(name="table", value_type=FiniteMapTable, request_field="table"),
        ),
        output_ports=(
            OutputPort(name="permutation", value_type=PermutationCertificate),
        ),
    )
    return DomainBundle(
        domain_id="finite_fields",
        schema_namespace="jacobian.finite-fields",
        semantics=DomainSemantics(
            name="jacobian.exact-finite-field-linear-algebra",
            version="1",
            definition={
                "field_identity": "exact modulus, generator, and ordered power basis",
                "linear_map": "explicit restriction of scalars to the prime field",
            },
        ),
        provider_runtime=provider,
        backend_version=f"sympy-{SYMPY_VERSION}",
        capabilities=(
            projective_line_operation,
            restrict_operation,
            rank_operation,
            ledger_operation,
            orbit_operation,
            table_operation,
            fiber_operation,
            collision_operation,
            permutation_operation,
        ),
        checker_declarations=FINITE_FIELD_EXACT_REPLAY_CHECKERS,
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_FINITE_FIELD_REQUEST",
                stage="finite_field_input_validation",
                message="Input does not satisfy the exact finite-field contract.",
                hint="Use values with identical presentations, axes, and bases.",
            )
        ),
    )


__all__ = ["build_finite_field_bundle"]
