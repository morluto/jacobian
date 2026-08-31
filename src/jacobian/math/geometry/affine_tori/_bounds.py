"""One admission plan and deadline for affine-torus fixed loci."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from time import monotonic
from typing import Literal, NoReturn

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.affine_tori.values import (
    MAX_AFFINE_TORUS_POINT_DIGITS,
    RationalAffineTorusMap,
)

# The dense upper-triangular 16-by-16, 32-digit boundary fixture completes in
# well under one second on an ordinary development host. Two minutes leaves a
# generous platform margin while retaining one finite owner deadline.
AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS = 120.0

type AffineTorusBackendEnvelopeCategory = Literal[
    "integer_rank",
    "integer_hnf",
    "rational_solve",
    "integer_snf",
    "rational_inverse",
    "integer_multiply",
]


@dataclass(frozen=True, slots=True)
class AffineTorusRankBounds:
    """Exact height envelope conditional on one possible rank of ``A-I``."""

    rank: int
    nullity: int
    source_minor_height: int
    source_hnf_transform_height: int
    character_hnf_transform_height: int
    image_saturation_height: int
    leading_solution_height: int
    integral_lift_height: int
    image_coordinate_height: int
    rational_intermediate_height: int
    base_point_component_height: int


@dataclass(frozen=True, slots=True)
class AffineTorusBackendEnvelope:
    """Structural bounds for the maintained FLINT primitive schedule.

    Counts are direct backend invocations, not a proxy for FLINT's private
    scalar instruction count. Operand dimensions and exact heights are bounded
    independently by the admitted source and every possible rank of ``A-I``.
    """

    primitive_call_limits: tuple[tuple[AffineTorusBackendEnvelopeCategory, int], ...]
    maximum_integer_rows: int
    maximum_integer_columns: int
    maximum_integer_height: int
    maximum_rational_rows: int
    maximum_rational_columns: int
    maximum_rational_height: int


@dataclass(frozen=True, slots=True)
class AffineTorusFixedLocusPlan:
    """Source-derived structural admission and delivery bounds."""

    dimension: int
    deadline: float
    displacement_height: int
    translation_common_denominator: int
    rank_bounds: tuple[AffineTorusRankBounds, ...]
    worker_input_bytes_upper_bound: int
    result_bytes_upper_bound: int
    backend_envelope: AffineTorusBackendEnvelope

    def bounds_for_rank(self, rank: int) -> AffineTorusRankBounds:
        """Return the precomputed envelope for the kernel's exact source rank."""

        for bounds in self.rank_bounds:
            if bounds.rank == rank:
                return bounds
        raise AssertionError("affine-torus rank is outside its admitted plan")


def _reject(reason: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("affine_map",),
        code=f"affine_torus.fixed_locus.{reason}",
        message=message,
    )


def begin_affine_torus_deadline() -> float:
    """Bind the one deadline covering admission, FLINT, result, and projection."""

    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    require_affine_torus_deadline(deadline, "before semantic admission")
    return deadline


def require_affine_torus_deadline(deadline: float, stage: str) -> None:
    """Stop without a mathematical conclusion when the request envelope expires."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"affine-torus fixed-locus computation cancelled {stage}"
        )
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"affine-torus fixed-locus deadline expired {stage}"
        )


def _decimal_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _rank_minor_height(rank: int, source_height: int) -> int:
    """Hadamard-bound every rank minor by ``(r ||M||_inf)^r``."""

    if rank == 0:
        return 1
    return int((rank * max(1, source_height)) ** rank)


def _augmented_hnf_transform_height(rows: int, minor_height: int) -> int:
    """Bound the transform extracted from ``HNF([B | I])``.

    Every full-row minor of ``[B | I]`` is a minor of ``B``. For a pivot
    minor ``P``, HNF pivot entries are at most its determinant. Cramer's rule
    for each remaining column then bounds every HNF entry, including the
    right-hand transform block, by ``rows * Delta^2``.
    """

    if rows == 0:
        return 1
    return rows * minor_height * minor_height


def _exact_integer_rank(matrix: tuple[tuple[int, ...], ...]) -> int:
    """Return the exact rational rank of a square integer matrix.

    Fraction-free (Bareiss-style) Gaussian elimination is exact over the
    integers, so it agrees with the tightened FLINT kernel's ``rank`` without
    requiring the isolated backend here.  The matrix is bounded by the admitted
    dimension and entry digit envelope, keeping admission one bounded pass.
    """

    dimension = len(matrix)
    if dimension == 0:
        return 0
    pivot_rows = [list(row) for row in matrix]
    rank = 0
    previous_pivot = 1
    for column in range(dimension):
        pivot_row = -1
        for row in range(rank, dimension):
            if pivot_rows[row][column] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        pivot_rows[rank], pivot_rows[pivot_row] = (
            pivot_rows[pivot_row],
            pivot_rows[rank],
        )
        pivot = pivot_rows[rank][column]
        for row in range(rank + 1, dimension):
            factor = pivot_rows[row][column]
            if factor == 0:
                continue
            for inner in range(column + 1, dimension):
                pivot_rows[row][inner] = (
                    pivot_rows[row][inner] * pivot - pivot_rows[rank][inner] * factor
                )
                if rank > 0:
                    pivot_rows[row][inner] //= previous_pivot
        previous_pivot = pivot
        rank += 1
        if rank == dimension:
            break
    return rank


def _rank_bounds(
    *,
    dimension: int,
    rank: int,
    displacement_height: int,
    common_denominator: int,
    translation_is_zero: bool,
) -> AffineTorusRankBounds:
    nullity = dimension - rank
    minor_height = _rank_minor_height(rank, displacement_height)
    source_transform_height = _augmented_hnf_transform_height(dimension, minor_height)

    # The bottom rows of HNF([M^t | I]) are the canonical HNF basis of the
    # saturated kernel. Their primitive Plucker coordinates are the rank
    # minors of M divided by their gcd, so their entries are <= minor_height.
    # The next augmented HNF sees at most ``nullity`` such columns.
    character_augmented_minor = (
        1 if nullity == 0 else (nullity * minor_height) ** nullity
    )
    character_transform_height = _augmented_hnf_transform_height(
        dimension, character_augmented_minor
    )
    # The kernel of the character matrix is the saturated image of M. Its
    # canonical HNF basis has the same primitive Plucker coordinates as M's
    # rational image, hence is bounded directly by M's rank minors even though
    # the transformation used to recover it has the coarser bound above.
    image_saturation_height = minor_height

    pairing_height = max(1, dimension * minor_height)
    # The character lattice is primitive. Its HNF leading block is therefore
    # the identity, so the leading solution is the integral pairing vector.
    leading_solution_height = pairing_height
    integral_lift_height = max(
        1,
        nullity * character_transform_height * leading_solution_height,
    )

    if rank == 0:
        image_coordinate_height = 1
        rational_intermediate_height = 1
    else:
        cofactor_height = (rank * minor_height) ** (rank - 1)
        image_coordinate_height = max(
            1,
            rank * cofactor_height * max(1, displacement_height),
        )
        component_solve_height = max(
            1,
            rank * cofactor_height * image_saturation_height,
        )
        base_rhs_height = common_denominator * (integral_lift_height + 1)
        base_solve_height = max(
            1,
            rank * cofactor_height * base_rhs_height,
            minor_height * common_denominator,
        )
        rational_intermediate_height = max(
            image_coordinate_height,
            component_solve_height,
            base_solve_height,
        )

    # After reduction modulo one, generator denominators divide a rank minor;
    # the base-point denominator additionally divides the translation lcm.
    # A rank-zero displacement has no such base point: the kernel's rank-zero
    # base solution is the zero point, and a translated identity resolves as an
    # empty locus.  Charging the translation lcm to that branch would fabricate
    # a denominator the result never carries, so leave it out for rank zero.
    base_point_component_height = (
        1
        if rank == 0 or translation_is_zero
        else max(1, minor_height * common_denominator)
    )
    return AffineTorusRankBounds(
        rank=rank,
        nullity=nullity,
        source_minor_height=minor_height,
        source_hnf_transform_height=source_transform_height,
        character_hnf_transform_height=character_transform_height,
        image_saturation_height=image_saturation_height,
        leading_solution_height=leading_solution_height,
        integral_lift_height=integral_lift_height,
        image_coordinate_height=image_coordinate_height,
        rational_intermediate_height=rational_intermediate_height,
        base_point_component_height=base_point_component_height,
    )


def _array_wire_bytes(count: int, item_bytes: int) -> int:
    """Return the exact compact-JSON size of ``count`` equal-width items."""

    return 2 + max(count - 1, 0) + count * item_bytes


def _array_item_wire_bytes(item_bytes: tuple[int, ...]) -> int:
    """Return the exact compact-JSON size of a heterogeneous array."""

    return 2 + max(len(item_bytes) - 1, 0) + sum(item_bytes)


def _nonnegative_integer_wire_bytes(value: int) -> int:
    return _decimal_digits(value)


def _integer_string_wire_bytes(height: int) -> int:
    """Bound a signed canonical integer string, including JSON quotes."""

    return _decimal_digits(height) + 3


def _rational_wire_bytes(height: int) -> int:
    """Bound a rational object with a signed numerator and positive denominator."""

    digits = _decimal_digits(height)
    return strict_json_object_size(
        (
            ("num", digits + 3),
            ("den", digits + 2),
        )
    )


def _torus_wire_bytes(dimension: int) -> int:
    return strict_json_object_size(
        (("dimension", _nonnegative_integer_wire_bytes(dimension)),)
    )


def _integer_matrix_wire_bytes(*, rows: int, columns: int, height: int) -> int:
    entry_bytes = _integer_string_wire_bytes(height)
    row_bytes = _array_wire_bytes(columns, entry_bytes)
    entries_bytes = _array_wire_bytes(rows, row_bytes)
    return strict_json_object_size(
        (
            ("domain", 4),  # canonical JSON string ``"ZZ"``
            ("row_count", _nonnegative_integer_wire_bytes(rows)),
            ("column_count", _nonnegative_integer_wire_bytes(columns)),
            ("entries", entries_bytes),
        )
    )


def _point_wire_bytes(*, dimension: int, height: int) -> int:
    return strict_json_object_size(
        (
            ("torus", _torus_wire_bytes(dimension)),
            (
                "coordinates",
                _array_wire_bytes(dimension, _rational_wire_bytes(height)),
            ),
        )
    )


def _source_wire_bytes(source: RationalAffineTorusMap) -> int:
    """Return the exact compact-JSON size of the retained canonical source."""

    dimension = source.torus.dimension
    entry_rows = tuple(
        _array_item_wire_bytes(tuple(len(entry) + 2 for entry in row))
        for row in source.linear_part.entries
    )
    linear_part_bytes = strict_json_object_size(
        (
            ("domain", 4),  # canonical JSON string ``"ZZ"``
            ("row_count", _nonnegative_integer_wire_bytes(dimension)),
            ("column_count", _nonnegative_integer_wire_bytes(dimension)),
            ("entries", _array_item_wire_bytes(entry_rows)),
        )
    )
    coordinate_bytes = tuple(
        strict_json_object_size(
            (
                ("num", len(coordinate.num) + 2),
                ("den", len(coordinate.den) + 2),
            )
        )
        for coordinate in source.translation.coordinates
    )
    torus_bytes = _torus_wire_bytes(dimension)
    translation_bytes = strict_json_object_size(
        (
            ("torus", torus_bytes),
            ("coordinates", _array_item_wire_bytes(coordinate_bytes)),
        )
    )
    return strict_json_object_size(
        (
            ("torus", torus_bytes),
            ("linear_part", linear_part_bytes),
            ("translation", translation_bytes),
        )
    )


def _result_bytes_for_rank(
    *,
    dimension: int,
    source_wire_bytes: int,
    bounds: AffineTorusRankBounds,
) -> int:
    """Bound both branches using the transport's exact compact-JSON grammar."""

    rank = bounds.rank
    nullity = bounds.nullity
    torus_bytes = _torus_wire_bytes(dimension)
    base_point_bytes = _point_wire_bytes(
        dimension=dimension,
        height=bounds.base_point_component_height,
    )
    generator_bytes = _point_wire_bytes(
        dimension=dimension,
        height=bounds.source_minor_height,
    )
    integer_bytes = _integer_string_wire_bytes(bounds.source_minor_height)
    identity_component_bytes = strict_json_object_size(
        (
            ("ambient_torus", torus_bytes),
            (
                "parameter_dimension",
                _nonnegative_integer_wire_bytes(nullity),
            ),
            (
                "embedding",
                _integer_matrix_wire_bytes(
                    rows=dimension,
                    columns=nullity,
                    height=bounds.source_minor_height,
                ),
            ),
        )
    )
    finite_components_bytes = strict_json_object_size(
        (
            ("generator_count", _nonnegative_integer_wire_bytes(rank)),
            (
                "relation_matrix",
                _integer_matrix_wire_bytes(
                    rows=rank,
                    columns=rank,
                    height=bounds.source_minor_height,
                ),
            ),
            ("generator_orders", _array_wire_bytes(rank, integer_bytes)),
            ("invariant_factors", _array_wire_bytes(rank, integer_bytes)),
            ("component_count", integer_bytes),
        )
    )
    fixed_locus_bytes = strict_json_object_size(
        (
            ("ambient_torus", torus_bytes),
            ("base_point", base_point_bytes),
            ("identity_component", identity_component_bytes),
            (
                "component_generators",
                _array_wire_bytes(rank, generator_bytes),
            ),
            ("finite_components", finite_components_bytes),
        )
    )
    nonempty_outcome_bytes = strict_json_object_size(
        (
            ("status", 10),  # canonical JSON string ``"NONEMPTY"``
            ("fixed_locus", fixed_locus_bytes),
        )
    )
    nonempty_bytes = strict_json_object_size(
        (
            ("source", source_wire_bytes),
            ("outcome", nonempty_outcome_bytes),
        )
    )
    branch_bytes = [nonempty_bytes]
    if nullity:
        obstruction_bytes = strict_json_object_size(
            (
                ("torus", torus_bytes),
                (
                    "coefficients",
                    _array_wire_bytes(dimension, integer_bytes),
                ),
            )
        )
        empty_outcome_bytes = strict_json_object_size(
            (
                ("status", 7),  # canonical JSON string ``"EMPTY"``
                ("obstruction", obstruction_bytes),
                (
                    "obstruction_pairing",
                    _rational_wire_bytes(bounds.base_point_component_height),
                ),
            )
        )
        branch_bytes.append(
            strict_json_object_size(
                (
                    ("source", source_wire_bytes),
                    ("outcome", empty_outcome_bytes),
                )
            )
        )
    return max(branch_bytes)


def _backend_envelope(
    *,
    dimension: int,
    displacement_height: int,
    rank_bounds: tuple[AffineTorusRankBounds, ...],
) -> AffineTorusBackendEnvelope:
    """Bound the exact maintained primitive schedule and its operand shapes."""

    positive_dimension = int(dimension > 0)
    integer_height = max(
        1,
        displacement_height,
        *(
            height
            for bounds in rank_bounds
            for height in (
                bounds.source_minor_height,
                bounds.source_hnf_transform_height,
                bounds.character_hnf_transform_height,
                bounds.image_saturation_height,
                bounds.leading_solution_height,
                bounds.integral_lift_height,
                bounds.image_coordinate_height,
            )
        ),
    )
    rational_height = max(
        1,
        *(bounds.rational_intermediate_height for bounds in rank_bounds),
    )
    return AffineTorusBackendEnvelope(
        primitive_call_limits=(
            ("integer_rank", 1 + 2 * dimension),
            ("integer_hnf", 3 + positive_dimension),
            ("rational_solve", 3 * positive_dimension),
            ("integer_snf", 1),
            ("rational_inverse", positive_dimension),
            ("integer_multiply", positive_dimension),
        ),
        maximum_integer_rows=dimension,
        maximum_integer_columns=2 * dimension,
        maximum_integer_height=integer_height,
        maximum_rational_rows=dimension,
        maximum_rational_columns=dimension,
        maximum_rational_height=rational_height,
    )


def build_affine_torus_plan(
    source: RationalAffineTorusMap, *, deadline: float
) -> AffineTorusFixedLocusPlan:
    """Build structural and exact-result envelopes before the first FLINT call."""

    require_affine_torus_deadline(deadline, "before work accounting")
    dimension = source.torus.dimension
    displacement_height = max(
        (
            abs(
                parse_canonical_integer(source.linear_part.entries[row][column])
                - int(row == column)
            )
            for row in range(dimension)
            for column in range(dimension)
        ),
        default=0,
    )
    common_denominator = lcm(
        *(
            coordinate.as_integer_ratio()[1]
            for coordinate in source.translation.coordinates
        )
    )
    translation_is_zero = all(
        coordinate.as_integer_ratio()[0] == 0
        for coordinate in source.translation.coordinates
    )
    # Admit against the source's actually attainable rank.  The exact rank of
    # the displacement A - I bounds every unreachable-rank branch that could
    # otherwise fabricate a too-large point-height or transport rejection (for
    # example dependent non-zero rows in a low-rank displacement).  Computing
    # it once here lets the admission gates look only at the result the kernel
    # will build at that rank rather than an over-large worst case.
    displacement = tuple(
        tuple(
            parse_canonical_integer(source.linear_part.entries[row][column])
            - int(row == column)
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    attained_rank = _exact_integer_rank(displacement)
    rank_bounds = (
        _rank_bounds(
            dimension=dimension,
            rank=attained_rank,
            displacement_height=displacement_height,
            common_denominator=common_denominator,
            translation_is_zero=translation_is_zero,
        ),
    )
    if any(
        _decimal_digits(bounds.base_point_component_height)
        > MAX_AFFINE_TORUS_POINT_DIGITS
        for bounds in rank_bounds
    ):
        _reject(
            "point_height",
            "the exact fixed-locus point bound exceeds the canonical torus-point "
            f"carrier's {MAX_AFFINE_TORUS_POINT_DIGITS}-digit envelope",
        )

    source_wire_bytes = _source_wire_bytes(source)
    result_bytes = max(
        _result_bytes_for_rank(
            dimension=dimension,
            source_wire_bytes=source_wire_bytes,
            bounds=bounds,
        )
        for bounds in rank_bounds
    )
    transport_limit = CanonicalLimits().max_output_bytes
    if result_bytes > transport_limit:
        _reject(
            "canonical_output",
            f"predicted exact result of {result_bytes} bytes exceeds the actual "
            f"{transport_limit}-byte canonical transport limit",
        )

    require_affine_torus_deadline(deadline, "after semantic admission")
    return AffineTorusFixedLocusPlan(
        dimension=dimension,
        deadline=deadline,
        displacement_height=displacement_height,
        translation_common_denominator=common_denominator,
        rank_bounds=rank_bounds,
        worker_input_bytes_upper_bound=source_wire_bytes,
        result_bytes_upper_bound=result_bytes,
        backend_envelope=_backend_envelope(
            dimension=dimension,
            displacement_height=displacement_height,
            rank_bounds=rank_bounds,
        ),
    )


__all__ = [
    "AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS",
    "AffineTorusBackendEnvelope",
    "AffineTorusFixedLocusPlan",
    "AffineTorusRankBounds",
    "begin_affine_torus_deadline",
    "build_affine_torus_plan",
    "require_affine_torus_deadline",
]
