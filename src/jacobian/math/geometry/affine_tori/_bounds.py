"""One admission plan, deadline, and work ledger for affine-torus fixed loci."""

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

# This release uses a conservative finite backend envelope: at most 16 axes,
# 32 source digits, a fixed number of dense FLINT calls, and the source-derived
# operand heights below. Envelope units combine dense cells, elimination
# dimension, and 32-digit height chunks. They are an admission proxy for that
# finite representation, not observed FLINT scalar-operation counts. Evaluating
# these formulas at n=16, |A-I|<=10^32, and denominator lcm <10^512 gives at
# most 1,656,332,960 units across the possible ranks; two billion retains margin.
MAX_AFFINE_TORUS_BACKEND_ENVELOPE_UNITS = 2_000_000_000

type AffineTorusBackendEnvelopeCategory = Literal[
    "source_conversion",
    "source_hnf",
    "character_hnf",
    "rank_minor_selection",
    "rational_linear_algebra",
    "relation_hnf",
    "smith",
    "integral_lift",
    "result_construction",
]

_BACKEND_ENVELOPE_CATEGORIES: tuple[AffineTorusBackendEnvelopeCategory, ...] = (
    "source_conversion",
    "source_hnf",
    "character_hnf",
    "rank_minor_selection",
    "rational_linear_algebra",
    "relation_hnf",
    "smith",
    "integral_lift",
    "result_construction",
)


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
class AffineTorusFixedLocusPlan:
    """The conservative envelope reused by every mandatory kernel phase."""

    dimension: int
    deadline: float
    displacement_height: int
    translation_common_denominator: int
    rank_bounds: tuple[AffineTorusRankBounds, ...]
    result_bytes_upper_bound: int
    backend_envelope_units_by_category: tuple[
        tuple[AffineTorusBackendEnvelopeCategory, int], ...
    ]

    @property
    def backend_envelope_units(self) -> int:
        return sum(amount for _, amount in self.backend_envelope_units_by_category)

    def bounds_for_rank(self, rank: int) -> AffineTorusRankBounds:
        """Return the precomputed envelope for the kernel's exact source rank."""

        if not 0 <= rank < len(self.rank_bounds):
            raise AssertionError("affine-torus rank is outside its admitted plan")
        return self.rank_bounds[rank]


class _BackendEnvelope:
    """Admission-only reservation for the fixed family of dense backend calls."""

    def __init__(self, *, deadline: float) -> None:
        self.deadline = deadline
        self.total = 0
        self.by_category: dict[AffineTorusBackendEnvelopeCategory, int] = dict.fromkeys(
            _BACKEND_ENVELOPE_CATEGORIES, 0
        )

    def reserve(
        self, category: AffineTorusBackendEnvelopeCategory, amount: int
    ) -> None:
        if amount < 0:
            raise AssertionError(
                "affine-torus backend reservations must be nonnegative"
            )
        require_affine_torus_deadline(
            self.deadline, f"while reserving {category} backend work"
        )
        self.total += amount
        self.by_category[category] += amount
        if self.total > MAX_AFFINE_TORUS_BACKEND_ENVELOPE_UNITS:
            _reject(
                "backend_envelope",
                "affine-torus fixed-locus backend work exceeds the conservative "
                f"{MAX_AFFINE_TORUS_BACKEND_ENVELOPE_UNITS}-unit envelope",
            )

    def freeze(
        self,
    ) -> tuple[tuple[AffineTorusBackendEnvelopeCategory, int], ...]:
        return tuple(
            (category, self.by_category[category])
            for category in _BACKEND_ENVELOPE_CATEGORIES
        )


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


def _digit_chunks(value: int) -> int:
    return (_decimal_digits(value) + 31) // 32


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


def _rank_bounds(
    *,
    dimension: int,
    rank: int,
    displacement_height: int,
    common_denominator: int,
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
    base_point_component_height = max(1, minor_height * common_denominator)
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


def _dense_backend_envelope_units(rows: int, columns: int, height: int) -> int:
    """Reserve dense cells, elimination dimension, and operand-height chunks."""

    if rows == 0 or columns == 0:
        return 0
    return rows * columns * min(rows, columns) * _digit_chunks(height) ** 2


def _work_for_rank(
    *,
    dimension: int,
    displacement_height: int,
    common_denominator: int,
    bounds: AffineTorusRankBounds,
) -> dict[AffineTorusBackendEnvelopeCategory, int]:
    """Reserve every backend primitive used by one rank-specific kernel path.

    The categories cover two source augmented HNFs, one character augmented
    HNF, source/minor rank calls, three rational solves and one inverse, one
    relation HNF, one Smith form, and the integral lift. The formulas use the
    largest source-derived operand height entering each fixed-size call; they do
    not claim to observe FLINT's private instruction count.
    """

    rank = bounds.rank
    nullity = bounds.nullity
    source_chunks = _digit_chunks(max(1, displacement_height))
    minor_chunks = _digit_chunks(bounds.source_minor_height)
    rational_chunks = _digit_chunks(bounds.rational_intermediate_height)
    denominator_chunks = _digit_chunks(common_denominator)

    return {
        "source_conversion": (
            (dimension * dimension + 2 * dimension) * source_chunks
            + dimension * dimension * denominator_chunks**2
        ),
        "source_hnf": 2
        * _dense_backend_envelope_units(
            dimension,
            2 * dimension,
            bounds.source_hnf_transform_height,
        ),
        "character_hnf": _dense_backend_envelope_units(
            dimension,
            dimension + nullity,
            bounds.character_hnf_transform_height,
        ),
        "rank_minor_selection": ((2 * dimension + 7) * dimension**3 * source_chunks),
        "rational_linear_algebra": 4 * dimension**3 * rational_chunks**2,
        "relation_hnf": _dense_backend_envelope_units(
            rank, rank, bounds.image_coordinate_height
        ),
        "smith": _dense_backend_envelope_units(rank, rank, bounds.source_minor_height),
        "integral_lift": (
            dimension
            * nullity
            * _digit_chunks(bounds.character_hnf_transform_height)
            * _digit_chunks(bounds.leading_solution_height)
        ),
        "result_construction": (3 * dimension * dimension + 4 * dimension + 32)
        * max(
            minor_chunks,
            _digit_chunks(bounds.base_point_component_height),
        ),
    }


def build_affine_torus_plan(
    source: RationalAffineTorusMap, *, deadline: float
) -> AffineTorusFixedLocusPlan:
    """Precharge every possible phase before the first FLINT backend call."""

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
    rank_bounds = tuple(
        _rank_bounds(
            dimension=dimension,
            rank=rank,
            displacement_height=displacement_height,
            common_denominator=common_denominator,
        )
        for rank in range(dimension + 1)
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

    rank_work = tuple(
        _work_for_rank(
            dimension=dimension,
            displacement_height=displacement_height,
            common_denominator=common_denominator,
            bounds=bounds,
        )
        for bounds in rank_bounds
    )
    envelope = _BackendEnvelope(deadline=deadline)
    for category in _BACKEND_ENVELOPE_CATEGORIES:
        envelope.reserve(
            category,
            max((work[category] for work in rank_work), default=0),
        )
    require_affine_torus_deadline(deadline, "after semantic admission")
    return AffineTorusFixedLocusPlan(
        dimension=dimension,
        deadline=deadline,
        displacement_height=displacement_height,
        translation_common_denominator=common_denominator,
        rank_bounds=rank_bounds,
        result_bytes_upper_bound=result_bytes,
        backend_envelope_units_by_category=envelope.freeze(),
    )


__all__ = [
    "AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS",
    "MAX_AFFINE_TORUS_BACKEND_ENVELOPE_UNITS",
    "AffineTorusFixedLocusPlan",
    "AffineTorusRankBounds",
    "begin_affine_torus_deadline",
    "build_affine_torus_plan",
    "require_affine_torus_deadline",
]
