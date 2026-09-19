"""Killable python-flint process boundary for affine-torus fixed loci."""

from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.geometry.affine_tori._bounds import (
    AffineTorusFixedLocusPlan,
    _exact_integer_rank,
    require_affine_torus_deadline,
)
from jacobian.math.geometry.affine_tori._kernel_types import (
    EmptyFixedLocusKernel,
    FixedLocusKernel,
    NonemptyFixedLocusKernel,
)
from jacobian.math.geometry.affine_tori.values import (
    MAX_AFFINE_TORUS_POINT_DIGITS,
    RationalAffineTorusMap,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_AFFINE_TORUS_WORKER = Path(__file__).resolve().with_name("_flint_worker.py")
_PROTOCOL_VERSION = 1
_WORKER_STDIN_LIMIT = 64 * 1024
_WORKER_STDERR_LIMIT = 64 * 1024
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_PARENT_FINALIZATION_SECONDS = 1.0
# The generated pairing numerator is the potentially dense parent-side Smith
# input. Keep its admitted scalar height below the fixed finalization budget;
# responses above this bound are rejected as unchecked rather than accepted.
_MAX_IMAGE_VALIDATOR_SMITH_DIGITS = 600


def _positive_worker_allowance(deadline: float) -> float:
    """Reserve bounded parent decoding and result construction time."""

    require_affine_torus_deadline(deadline, "before launching the FLINT worker")
    remaining = deadline - monotonic()
    worker_allowance = remaining - _PARENT_FINALIZATION_SECONDS
    if worker_allowance <= 0:
        raise OperationExecutionTimeoutError(
            "affine-torus fixed-locus deadline has no worker execution allowance"
        )
    return worker_allowance


def _worker_input(source: RationalAffineTorusMap) -> bytes:
    payload = {
        "protocol_version": _PROTOCOL_VERSION,
        "dimension": source.torus.dimension,
        "linear_part": source.linear_part.model_dump(mode="json")["entries"],
        "translation": [
            coordinate.model_dump(mode="json")
            for coordinate in source.translation.coordinates
        ],
    }
    return encode_strict_json(payload)


def _strict_integer(value: Any, *, maximum_digits: int, positive: bool = False) -> int:
    if not isinstance(value, str) or len(value.lstrip("-")) > maximum_digits:
        raise ValueError("worker result integer exceeded its admitted digit bound")
    parsed = parse_canonical_integer(value)
    if value != format_canonical_integer(parsed) or (positive and parsed <= 0):
        raise ValueError("worker result integer is not canonical")
    return parsed


def _strict_fraction(value: Any, *, maximum_digits: int) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("worker result rational has invalid fields")
    numerator = _strict_integer(value["num"], maximum_digits=maximum_digits)
    denominator = _strict_integer(
        value["den"], maximum_digits=maximum_digits, positive=True
    )
    result = Fraction(numerator, denominator)
    if (
        value["num"] != format_canonical_integer(result.numerator)
        or value["den"] != format_canonical_integer(result.denominator)
        or not 0 <= result < 1
    ):
        raise ValueError("worker result rational is not canonical modulo one")
    return result


def _strict_integer_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    maximum_digits: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("worker result matrix has an invalid row count")
    result: list[tuple[int, ...]] = []
    for candidate_row in value:
        if not isinstance(candidate_row, list) or len(candidate_row) != columns:
            raise ValueError("worker result matrix has an invalid column count")
        result.append(
            tuple(
                _strict_integer(entry, maximum_digits=maximum_digits)
                for entry in candidate_row
            )
        )
    return tuple(result)


def _strict_fraction_vector(
    value: Any, *, length: int, maximum_digits: int
) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("worker result point has an invalid dimension")
    return tuple(
        _strict_fraction(entry, maximum_digits=maximum_digits) for entry in value
    )


def _strict_dimension(value: Any, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= maximum
    ):
        raise ValueError("worker result dimension is invalid")
    return value


def _integer_determinant(matrix: tuple[tuple[int, ...], ...]) -> int:
    """Return the exact determinant of one admitted square integer matrix."""

    size = len(matrix)
    if size == 0:
        return 1
    entries = [list(row) for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(size - 1):
        pivot_row = next(
            (row for row in range(pivot_index, size) if entries[row][pivot_index]),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            entries[pivot_index], entries[pivot_row] = (
                entries[pivot_row],
                entries[pivot_index],
            )
            sign = -sign
        pivot = entries[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    entries[row][column] * pivot
                    - entries[row][pivot_index] * entries[pivot_index][column]
                )
                entries[row][column] = numerator // previous
            entries[row][pivot_index] = 0
        previous = pivot
    return sign * entries[-1][-1]


def _integer_smith_normal_form(  # noqa: C901
    matrix: tuple[tuple[int, ...], ...], *, columns: int | None = None
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Return Smith diagonal entries and the right unimodular transform.

    This small fraction-free implementation is intentionally local to the
    response boundary.  The worker's FLINT result is not trusted merely
    because its fields agree with one another: the bounded parent recomputes
    the invariant factors and, for an embedding, a saturated annihilator
    basis.  The latter is the final set of columns of the returned transform.
    """

    row_count = len(matrix)
    column_count = columns if columns is not None else (len(matrix[0]) if matrix else 0)
    if any(len(row) != column_count for row in matrix):
        raise ValueError("integer Smith input has inconsistent row lengths")
    entries = [list(row) for row in matrix]
    right = [
        [int(row == column) for column in range(column_count)]
        for row in range(column_count)
    ]

    def swap_columns(left: int, right_column: int) -> None:
        for row in entries:
            row[left], row[right_column] = row[right_column], row[left]
        for row in right:
            row[left], row[right_column] = row[right_column], row[left]

    pivot_index = 0
    while pivot_index < row_count and pivot_index < column_count:
        position = next(
            (
                (row, column)
                for row in range(pivot_index, row_count)
                for column in range(pivot_index, column_count)
                if entries[row][column]
            ),
            None,
        )
        if position is None:
            break
        pivot_row, pivot_column = position
        entries[pivot_index], entries[pivot_row] = (
            entries[pivot_row],
            entries[pivot_index],
        )
        swap_columns(pivot_index, pivot_column)

        while True:
            if entries[pivot_index][pivot_index] < 0:
                entries[pivot_index] = [-value for value in entries[pivot_index]]
            pivot = entries[pivot_index][pivot_index]
            changed = False
            for row in range(pivot_index + 1, row_count):
                value = entries[row][pivot_index]
                if not value:
                    continue
                quotient, remainder = divmod(value, pivot)
                entries[row] = [
                    entries[row][column] - quotient * entries[pivot_index][column]
                    for column in range(column_count)
                ]
                if remainder:
                    entries[row], entries[pivot_index] = (
                        entries[pivot_index],
                        entries[row],
                    )
                    changed = True
                    break
            if changed:
                continue
            pivot = entries[pivot_index][pivot_index]
            for column in range(pivot_index + 1, column_count):
                value = entries[pivot_index][column]
                if not value:
                    continue
                quotient, remainder = divmod(value, pivot)
                for row in range(row_count):
                    entries[row][column] -= quotient * entries[row][pivot_index]
                for row in range(column_count):
                    right[row][column] -= quotient * right[row][pivot_index]
                if remainder:
                    swap_columns(column, pivot_index)
                    changed = True
                    break
            if changed:
                continue
            pivot = entries[pivot_index][pivot_index]
            offending = next(
                (
                    (row, column)
                    for row in range(pivot_index + 1, row_count)
                    for column in range(pivot_index + 1, column_count)
                    if entries[row][column] % pivot
                ),
                None,
            )
            if offending is not None:
                # The offending entry is brought into the pivot row; the next
                # column-reduction pass strictly decreases the pivot.
                entries[pivot_index] = [
                    entries[pivot_index][column] + entries[offending[0]][column]
                    for column in range(column_count)
                ]
                continue
            break
        pivot_index += 1

    diagonal = tuple(
        abs(entries[index][index]) for index in range(min(row_count, column_count))
    )
    return diagonal, tuple(tuple(row) for row in right)


def _rational_inverse(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Invert one full-rank admitted integer matrix over the rationals."""

    size = len(matrix)
    augmented = [
        [Fraction(value) for value in matrix_row]
        + [Fraction(int(row == column)) for column in range(size)]
        for row, matrix_row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError("relation matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                left - scale * right
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]
    return tuple(tuple(row[size:]) for row in augmented)


def _sympy_smith_diagonal(
    matrix: tuple[tuple[int, ...], ...], *, maximum_digits: int
) -> tuple[int, ...]:
    """Compute Smith factors with a maintained exact fallback backend.

    This is called only after the response's dimensions and integer heights
    have been admitted.  A backend failure is a malformed/uncheckable worker
    response, never a reason to accept it.
    """

    size = len(matrix)
    if any(len(row) != size for row in matrix) or size > 16:
        raise ValueError("worker Smith check exceeds its admitted dimensions")
    if any(
        len(format_canonical_integer(abs(value))) > maximum_digits
        for row in matrix
        for value in row
    ):
        raise ValueError("worker Smith check exceeds its admitted integer height")
    try:
        from sympy import Matrix
        from sympy.matrices.normalforms import smith_normal_form
        from sympy.polys.domains import ZZ

        smith = smith_normal_form(Matrix(matrix), domain=ZZ)
    except Exception as exc:
        raise ValueError("worker Smith factors could not be checked") from exc
    return tuple(abs(int(smith[index, index])) for index in range(size))


def _require_integral(value: Fraction, *, label: str) -> None:
    if value.denominator != 1:
        raise ValueError(f"worker {label} is not integral")


def _source_displacement_and_translation(
    source: RationalAffineTorusMap,
) -> tuple[tuple[tuple[int, ...], ...], tuple[Fraction, ...]]:
    linear_part = tuple(
        tuple(int(value) for value in row) for row in source.linear_part.entries
    )
    return (
        tuple(
            tuple(value - int(row == column) for column, value in enumerate(entries))
            for row, entries in enumerate(linear_part)
        ),
        tuple(
            coordinate.as_fraction() for coordinate in source.translation.coordinates
        ),
    )


def _identity_annihilators(
    identity_embedding: tuple[tuple[int, ...], ...],
    *,
    dimension: int,
    nullity: int,
) -> tuple[tuple[int, ...], ...]:
    """Return a saturated integer character basis annihilating an embedding."""

    embedding_transpose = tuple(
        tuple(identity_embedding[row][column] for row in range(dimension))
        for column in range(nullity)
    )
    diagonal, right_transform = _integer_smith_normal_form(
        embedding_transpose, columns=dimension
    )
    if any(value != 1 for value in diagonal):
        raise ValueError("worker identity embedding is not saturated")
    return tuple(
        tuple(right_transform[row][column] for row in range(dimension))
        for column in range(nullity, dimension)
    )


def _validate_empty_projection(
    *,
    character: tuple[int, ...],
    pairing: Fraction,
    displacement: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> None:
    """Validate a worker's primitive affine obstruction against its source."""

    if not character or math.gcd(*(abs(value) for value in character)) != 1:
        raise ValueError("worker obstruction character is not primitive and nonzero")
    for column in range(len(displacement)):
        if sum(
            character[row] * displacement[row][column]
            for row in range(len(displacement))
        ):
            raise ValueError("worker obstruction character is not in the left kernel")
    expected_pairing = (
        sum(
            (character[row] * translation[row] for row in range(len(translation))),
            Fraction(0),
        )
        % 1
    )
    if not expected_pairing or pairing != expected_pairing:
        raise ValueError("worker obstruction pairing disagrees with the source")


def _validate_source_relations(
    *,
    rank: int,
    nullity: int,
    base_point: tuple[Fraction, ...],
    identity_embedding: tuple[tuple[int, ...], ...],
    component_generators: tuple[tuple[Fraction, ...], ...],
    displacement: tuple[tuple[int, ...], ...],
    translation: tuple[Fraction, ...],
) -> tuple[tuple[int, ...], ...]:
    dimension = len(displacement)
    if _exact_integer_rank(displacement) != rank:
        raise ValueError("worker rank disagrees with the retained source")

    if _exact_integer_rank(identity_embedding) != nullity:
        raise ValueError("worker identity embedding is not independent")
    for row in range(dimension):
        for column in range(nullity):
            if sum(
                displacement[row][index] * identity_embedding[index][column]
                for index in range(dimension)
            ):
                raise ValueError("worker identity embedding is not in the kernel")
    for row in range(dimension):
        affine_value = translation[row] + sum(
            displacement[row][column] * base_point[column]
            for column in range(dimension)
        )
        _require_integral(affine_value, label="base-point affine equation")

    for generator in component_generators:
        for row in range(dimension):
            _require_integral(
                sum(
                    (
                        displacement[row][column] * generator[column]
                        for column in range(dimension)
                    ),
                    Fraction(0),
                ),
                label="component-generator congruence",
            )
    return _identity_annihilators(
        identity_embedding,
        dimension=dimension,
        nullity=nullity,
    )


def _validate_component_presentation(
    *,
    rank: int,
    relation_matrix: tuple[tuple[int, ...], ...],
    generator_orders: tuple[int, ...],
    invariant_factors: tuple[int, ...],
    component_count: int,
    maximum_integer_digits: int,
) -> None:
    determinant = _integer_determinant(relation_matrix)
    if determinant == 0 or abs(determinant) != component_count:
        raise ValueError("worker component count disagrees with relation determinant")
    diagonal = _sympy_smith_diagonal(
        relation_matrix,
        maximum_digits=maximum_integer_digits,
    )
    if any(value <= 0 for value in diagonal) or any(
        right % left for left, right in pairwise(diagonal)
    ):
        raise ValueError("worker relation matrix has invalid Smith factors")
    expected_factors = tuple(value for value in diagonal if value > 1)
    if invariant_factors != expected_factors:
        raise ValueError("worker invariant factors disagree with relation matrix")
    if math.prod(diagonal) != component_count:
        raise ValueError(
            "worker invariant-factor product disagrees with component count"
        )

    inverse = _rational_inverse(relation_matrix)
    expected_orders = tuple(
        math.lcm(*(inverse[row][column].denominator for row in range(rank)))
        for column in range(rank)
    )
    if generator_orders != expected_orders:
        raise ValueError("worker generator orders disagree with relation matrix")


def _validate_relation_congruences(
    *,
    dimension: int,
    rank: int,
    component_generators: tuple[tuple[Fraction, ...], ...],
    relation_matrix: tuple[tuple[int, ...], ...],
    generator_orders: tuple[int, ...],
    annihilators: tuple[tuple[int, ...], ...],
    maximum_integer_digits: int,
) -> None:

    # The relation columns must vanish in the connected identity component.
    # The same saturated character basis also determines the exact order of
    # every submitted component generator and the size of their generated
    # subgroup.
    pairings = tuple(
        tuple(
            sum(
                (
                    Fraction(character[row]) * component_generators[generator][row]
                    for row in range(dimension)
                ),
                Fraction(0),
            )
            for generator in range(rank)
        )
        for character in annihilators
    )
    expected_orders = tuple(
        math.lcm(*(pairings[row][column].denominator for row in range(rank)))
        for column in range(rank)
    )
    if generator_orders != expected_orders:
        raise ValueError("worker generator orders disagree with component congruences")
    for relation_column in range(rank):
        relation_pairing = tuple(
            sum(
                (
                    pairings[row][generator]
                    * relation_matrix[generator][relation_column]
                    for generator in range(rank)
                ),
                Fraction(0),
            )
            for row in range(rank)
        )
        for value in relation_pairing:
            _require_integral(value, label="component relation congruence")

    # Clear the common denominator of the pairing matrix P.  Smith diagonal
    # entries of N=dP determine the image of Z^rank -> (Z/dZ)^rank: each
    # diagonal entry s contributes d/gcd(d,s) elements.  This avoids an
    # unbounded enumeration of the finite quotient while detecting extra
    # relations such as two identical order-two generators.
    denominator = math.lcm(
        1,
        *(value.denominator for row in pairings for value in row),
    )
    numerator = tuple(
        tuple(int(value * denominator) for value in row) for row in pairings
    )
    image_diagonal = _sympy_smith_diagonal(
        numerator,
        maximum_digits=_MAX_IMAGE_VALIDATOR_SMITH_DIGITS,
    )
    image_order = math.prod(
        denominator // math.gcd(denominator, value) if value else 1
        for value in image_diagonal
    )
    if image_order != abs(_integer_determinant(relation_matrix)):
        raise ValueError("worker generators have an unexpected finite subgroup size")


def _validate_nonempty_projection(
    *,
    rank: int,
    nullity: int,
    base_point: tuple[Fraction, ...],
    identity_embedding: tuple[tuple[int, ...], ...],
    component_generators: tuple[tuple[Fraction, ...], ...],
    relation_matrix: tuple[tuple[int, ...], ...],
    generator_orders: tuple[int, ...],
    invariant_factors: tuple[int, ...],
    component_count: int,
    maximum_integer_digits: int,
    source: RationalAffineTorusMap,
) -> None:
    """Validate every mathematical claim needed by result construction."""

    displacement, translation = _source_displacement_and_translation(source)
    annihilators = _validate_source_relations(
        rank=rank,
        nullity=nullity,
        base_point=base_point,
        identity_embedding=identity_embedding,
        component_generators=component_generators,
        displacement=displacement,
        translation=translation,
    )
    _validate_component_presentation(
        rank=rank,
        relation_matrix=relation_matrix,
        generator_orders=generator_orders,
        invariant_factors=invariant_factors,
        component_count=component_count,
        maximum_integer_digits=maximum_integer_digits,
    )
    _validate_relation_congruences(
        dimension=source.torus.dimension,
        rank=rank,
        component_generators=component_generators,
        relation_matrix=relation_matrix,
        generator_orders=generator_orders,
        annihilators=annihilators,
        maximum_integer_digits=maximum_integer_digits,
    )


def _decode_worker_projection(
    payload: Any,
    *,
    request_digest: str,
    source: RationalAffineTorusMap,
    plan: AffineTorusFixedLocusPlan,
) -> FixedLocusKernel:
    common_fields = {"protocol_version", "request_digest", "status"}
    if not isinstance(payload, dict) or not common_fields <= set(payload):
        raise ValueError("worker result is not an object with protocol fields")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != _PROTOCOL_VERSION
        or payload["request_digest"] != request_digest
    ):
        raise ValueError("worker result is not bound to its admitted request")
    dimension = source.torus.dimension
    maximum_integer_digits = max(
        len(format_canonical_integer(bounds.source_minor_height))
        for bounds in plan.rank_bounds
    )
    if payload["status"] == "EMPTY":
        if set(payload) != common_fields | {"character", "pairing"}:
            raise ValueError("empty worker result has invalid fields")
        character_value = payload["character"]
        if not isinstance(character_value, list) or len(character_value) != dimension:
            raise ValueError("worker obstruction has an invalid dimension")
        character = tuple(
            _strict_integer(value, maximum_digits=maximum_integer_digits)
            for value in character_value
        )
        pairing = _strict_fraction(
            payload["pairing"], maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS
        )
        displacement, translation = _source_displacement_and_translation(source)
        _validate_empty_projection(
            character=character,
            pairing=pairing,
            displacement=displacement,
            translation=translation,
        )
        return EmptyFixedLocusKernel(character=character, pairing=pairing)
    if payload["status"] != "NONEMPTY" or set(payload) != common_fields | {
        "rank",
        "nullity",
        "base_point",
        "identity_embedding",
        "component_generators",
        "relation_matrix",
        "generator_orders",
        "invariant_factors",
        "component_count",
    }:
        raise ValueError("nonempty worker result has invalid fields")
    rank = _strict_dimension(payload["rank"], maximum=dimension)
    nullity = _strict_dimension(payload["nullity"], maximum=dimension)
    if rank + nullity != dimension:
        raise ValueError("worker result rank and nullity disagree")
    generators_value = payload["component_generators"]
    if not isinstance(generators_value, list) or len(generators_value) != rank:
        raise ValueError("worker component generators have an invalid count")
    orders_value = payload["generator_orders"]
    factors_value = payload["invariant_factors"]
    if not isinstance(orders_value, list) or len(orders_value) != rank:
        raise ValueError("worker component orders have an invalid count")
    if not isinstance(factors_value, list) or len(factors_value) > rank:
        raise ValueError("worker invariant factors have an invalid count")
    base_point = _strict_fraction_vector(
        payload["base_point"],
        length=dimension,
        maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
    )
    identity_embedding = _strict_integer_matrix(
        payload["identity_embedding"],
        rows=dimension,
        columns=nullity,
        maximum_digits=maximum_integer_digits,
    )
    component_generators = tuple(
        _strict_fraction_vector(
            generator,
            length=dimension,
            maximum_digits=MAX_AFFINE_TORUS_POINT_DIGITS,
        )
        for generator in generators_value
    )
    relation_matrix = _strict_integer_matrix(
        payload["relation_matrix"],
        rows=rank,
        columns=rank,
        maximum_digits=maximum_integer_digits,
    )
    generator_orders = tuple(
        _strict_integer(value, maximum_digits=maximum_integer_digits, positive=True)
        for value in orders_value
    )
    invariant_factors = tuple(
        _strict_integer(value, maximum_digits=maximum_integer_digits, positive=True)
        for value in factors_value
    )
    component_count = _strict_integer(
        payload["component_count"],
        maximum_digits=maximum_integer_digits,
        positive=True,
    )
    _validate_nonempty_projection(
        rank=rank,
        nullity=nullity,
        base_point=base_point,
        identity_embedding=identity_embedding,
        component_generators=component_generators,
        relation_matrix=relation_matrix,
        generator_orders=generator_orders,
        invariant_factors=invariant_factors,
        component_count=component_count,
        maximum_integer_digits=maximum_integer_digits,
        source=source,
    )
    return NonemptyFixedLocusKernel(
        base_point=base_point,
        identity_embedding=identity_embedding,
        component_generators=component_generators,
        relation_matrix=relation_matrix,
        generator_orders=generator_orders,
        invariant_factors=invariant_factors,
        component_count=component_count,
    )


def compute_fixed_locus_kernel(
    source: RationalAffineTorusMap,
    plan: AffineTorusFixedLocusPlan,
) -> FixedLocusKernel:
    """Run one admitted FLINT kernel in a bounded, killable child process."""

    input_bytes = _worker_input(source)
    # The private request omits the canonical source's domain and repeated
    # torus metadata, so the retained-source wire bound is conservative here.
    if (
        len(input_bytes) > plan.worker_input_bytes_upper_bound
        or len(input_bytes) > _WORKER_STDIN_LIMIT
    ):
        raise AssertionError("admitted affine-torus worker request exceeded its bound")
    request_digest = hashlib.sha256(input_bytes).hexdigest()
    require_affine_torus_deadline(plan.deadline, "before FLINT worker setup")
    try:
        with TemporaryDirectory(prefix="jacobian-affine-torus-flint-") as directory:
            allowance = _positive_worker_allowance(plan.deadline)
            completed = run_bounded_process(
                [sys.executable, str(_AFFINE_TORUS_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=allowance,
                environment=worker_environment(locale="C.UTF-8"),
                # The projection drops the complete retained source; that
                # omission dominates its small protocol and digest fields.
                stdout_limit=plan.worker_stdout_bytes_upper_bound,
                stderr_limit=_WORKER_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(allowance)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("bounded affine-torus FLINT worker could not start") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "affine-torus fixed-locus computation cancelled during the FLINT worker"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "affine-torus FLINT worker exhausted its execution allowance"
        )
    require_affine_torus_deadline(plan.deadline, "after cleaning up the FLINT worker")
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError(
            "bounded affine-torus FLINT worker did not establish a fixed locus"
        )
    try:
        decoded = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(
                max_input_bytes=plan.worker_stdout_bytes_upper_bound
            ),
        )
        result = _decode_worker_projection(
            decoded,
            request_digest=request_digest,
            source=source,
            plan=plan,
        )
    except (CanonicalizationError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "bounded affine-torus FLINT worker returned malformed output"
        ) from exc
    require_affine_torus_deadline(plan.deadline, "after decoding the FLINT worker")
    return result


__all__ = ["compute_fixed_locus_kernel"]
