"""Exact associated-graded kernel for filtered chain complexes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_LEVELS,
    MAX_FILTER_VECTORS_PER_GROUP,
    MAX_SPECTRAL_PAGE,
    AssociatedGradedResult,
    FiltrationLevel,
    GradedSquareLedgerEntry,
    SpectralDifferential,
    SpectralPageResult,
    SpectralPageStatus,
    SpectralSquareLedgerEntry,
    Vector,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    require_prime_field_admission,
)

Scalar = Fraction | int
Matrix = list[list[Scalar]]
VectorList = list[Scalar]

__all__ = [
    "admit_filtered",
    "admit_spectral_page",
    "associated_graded",
    "spectral_page",
]


def _fail(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _admit_filtered_structure(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> None:
    """Admit the bounded wire shape before semantic filtered checks."""
    if complex_value.coefficient_ring is CoefficientRing.INTEGER:
        raise _fail(
            ("complex",),
            "filtered_chain_complex.integer_coefficients_unsupported",
            "associated graded supports QQ and GF(p) filtrations",
        )
    try:
        require_prime_field_admission(
            complex_value.coefficient_ring, complex_value.prime
        )
    except ValueError as exc:
        raise _fail(
            ("complex",), "filtered_chain_complex.prime_invalid", str(exc)
        ) from exc
    if not isinstance(filtration, tuple) or not 1 <= len(filtration) <= (
        MAX_FILTER_LEVELS
    ):
        raise _fail(
            ("filtration",),
            "filtered_chain_complex.level_count_invalid",
            f"filtration must carry between 1 and {MAX_FILTER_LEVELS} levels",
        )
    from jacobian.math.topology.chain_complexes.values import (
        _require_coefficient_scalar,
    )

    sizes = complex_value.basis_sizes
    if any(size > MAX_FILTER_AMBIENT_DIMENSION for size in sizes):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="filtered_chain_complex.ambient_dimension_exceeded",
            message="filtered chain complexes admit at most "
            f"{MAX_FILTER_AMBIENT_DIMENSION} basis vectors per chain group",
        )
    for level_index, level in enumerate(filtration):
        if not isinstance(level, FiltrationLevel) or len(level.subspaces) != len(sizes):
            raise _fail(
                ("filtration", level_index),
                "filtered_chain_complex.degree_coverage_invalid",
                "every filtration level must carry one subspace per chain degree",
            )
        for degree_index, subspace in enumerate(level.subspaces):
            if len(subspace.vectors) > MAX_FILTER_VECTORS_PER_GROUP:
                raise OperationResourceAdmissionError(
                    location=("filtration", level_index, degree_index),
                    code="filtered_chain_complex.subspace_size_exceeded",
                    message="a filtration subspace exceeds the admitted vector count",
                )
            for vector in subspace.vectors:
                if not isinstance(vector, tuple) or len(vector) != sizes[degree_index]:
                    raise _fail(
                        ("filtration", level_index, degree_index),
                        "filtered_chain_complex.vector_axis_invalid",
                        "filtration vectors must use ambient chain coordinates",
                    )
                for entry in vector:
                    try:
                        _require_coefficient_scalar(
                            complex_value.coefficient_ring,
                            entry,
                            prime=complex_value.prime,
                        )
                    except (
                        AttributeError,
                        TypeError,
                        ValidationError,
                        ValueError,
                    ) as exc:
                        raise _fail(
                            ("filtration", level_index, degree_index),
                            "filtered_chain_complex.entry_grammar_invalid",
                            f"filtration entry '{entry}' is not canonical",
                        ) from exc


@dataclass
class _FilteredAdmission:
    differentials: list[Matrix]
    parsed_levels: list[list[Matrix]]
    bases: list[list[Matrix]]


def _admit_filtered_semantics(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> _FilteredAdmission:
    """Establish all filtered-complex invariants once for one consumer."""
    _admit_filtered_structure(complex_value, filtration)
    prime = complex_value.prime
    degree_count = len(complex_value.basis_sizes)
    differentials = [
        [[_parse_entry(entry, prime) for entry in row] for row in matrix]
        for matrix in complex_value.differential_matrices
    ]
    for index in range(len(differentials) - 1):
        product = _mat_mul(differentials[index], differentials[index + 1], prime)
        if any(not _is_zero(value) for row in product for value in row):
            raise _fail(
                ("complex",),
                "filtered_chain_complex.source_not_a_complex",
                "the source differentials must satisfy d^2 = 0",
            )
    parsed_levels: list[list[Matrix]] = [
        [
            [
                [_parse_entry(entry, prime) for entry in vector]
                for vector in level.subspaces[degree].vectors
            ]
            for degree in range(degree_count)
        ]
        for level in filtration
    ]
    bases: list[list[Matrix]] = [
        [_row_basis(level[degree], prime) for degree in range(degree_count)]
        for level in parsed_levels
    ]
    for level_index in range(1, len(filtration)):
        for degree in range(degree_count):
            for vector in parsed_levels[level_index - 1][degree]:
                if not _in_span(bases[level_index][degree], vector, prime):
                    raise _fail(
                        ("filtration", level_index),
                        "filtered_chain_complex.not_nested",
                        f"level {level_index - 1} is not contained in level "
                        f"{level_index} in degree {degree}",
                    )
    top = len(filtration) - 1
    for degree in range(degree_count):
        if len(bases[top][degree]) != complex_value.basis_sizes[degree]:
            raise _fail(
                ("filtration", top),
                "filtered_chain_complex.not_exhaustive",
                f"the top filtration level must span chain group {degree}",
            )
    for level_index in range(len(filtration)):
        for degree in range(1, degree_count):
            for vector in parsed_levels[level_index][degree]:
                image = _mat_vec(differentials[degree - 1], vector, prime)
                if not _in_span(bases[level_index][degree - 1], image, prime):
                    raise _fail(
                        ("filtration", level_index),
                        "filtered_chain_complex.differential_not_preserving",
                        f"the differential does not preserve filtration level "
                        f"{level_index} in degree {degree}",
                    )
    return _FilteredAdmission(differentials, parsed_levels, bases)


def admit_filtered(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> None:
    """Shared native+catalog admission for a semantically valid filtration."""
    _admit_filtered_semantics(complex_value, filtration)


def _parse_entry(entry: int | Fraction, prime: int | None) -> Scalar:
    if prime is not None:
        if type(entry) is not int:
            raise RuntimeError("finite-field chain coefficients must be integers")
        return entry % prime
    return entry if type(entry) is Fraction else Fraction(entry)


def _modular_scalar(value: Scalar) -> int:
    if not isinstance(value, int):
        raise RuntimeError("finite-field chain arithmetic received a rational scalar")
    return value


def _rational_scalar(value: Scalar) -> Fraction:
    if not isinstance(value, Fraction):
        raise RuntimeError("rational chain arithmetic received a modular scalar")
    return value


def _serialize_scalar(value: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        return _modular_scalar(value) % prime
    return _rational_scalar(value)


def _add(left: Scalar, right: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        return (_modular_scalar(left) + _modular_scalar(right)) % prime
    return _rational_scalar(left) + _rational_scalar(right)


def _mul(left: Scalar, right: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        return (_modular_scalar(left) * _modular_scalar(right)) % prime
    return _rational_scalar(left) * _rational_scalar(right)


def _neg(value: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        return (-_modular_scalar(value)) % prime
    return -_rational_scalar(value)


def _inv(value: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        return pow(_modular_scalar(value) % prime, -1, prime)
    return Fraction(1, 1) / _rational_scalar(value)


def _is_zero(value: Scalar) -> bool:
    return value == 0


def _rref(matrix: Matrix, prime: int | None) -> tuple[Matrix, tuple[int, ...]]:
    """Exact reduced row echelon form with pivot columns."""
    rows = [list(row) for row in matrix]
    pivots: list[int] = []
    pivot_row = 0
    columns = len(rows[0]) if rows else 0
    for column in range(columns):
        candidate: int | None = None
        for row in range(pivot_row, len(rows)):
            if not _is_zero(rows[row][column]):
                candidate = row
                break
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        scale = _inv(rows[pivot_row][column], prime)
        rows[pivot_row] = [_mul(value, scale, prime) for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row != pivot_row and not _is_zero(rows[row][column]):
                factor = rows[row][column]
                rows[row] = [
                    _add(current, _neg(_mul(factor, pivot, prime), prime), prime)
                    for current, pivot in zip(rows[row], rows[pivot_row], strict=True)
                ]
        pivots.append(column)
        pivot_row += 1
    return rows, tuple(pivots)


def _row_basis(rows: Matrix, prime: int | None) -> Matrix:
    reduced, _ = _rref(rows, prime)
    return [row for row in reduced if any(not _is_zero(value) for value in row)]


def _solve(system: Matrix, target: VectorList, prime: int | None) -> VectorList | None:
    """Solve system x = target for x, or return None when inconsistent."""
    if not system:
        if any(not _is_zero(value) for value in target):
            return None
        return []
    augmented = [[*row, rhs] for row, rhs in zip(system, target, strict=True)]
    reduced, pivots = _rref(augmented, prime)
    width = len(system[0])
    for row in reduced:
        if all(_is_zero(value) for value in row[:width]) and not _is_zero(row[width]):
            return None
    solution: VectorList = [_parse_entry(0, prime) for _ in range(width)]
    # Reduced echelon form keeps pivot rows first, so pivot ``i`` owns
    # ``reduced[i]`` even when dependent rows leave trailing zero rows.
    for row_index, pivot in enumerate(pivots):
        if pivot < width:
            solution[pivot] = reduced[row_index][width]
    return solution


def _transpose(rows: Matrix) -> Matrix:
    if not rows:
        return []
    return [
        [rows[row][column] for row in range(len(rows))]
        for column in range(len(rows[0]))
    ]


def _in_span(basis: Matrix, vector: VectorList, prime: int | None) -> bool:
    if not basis:
        return all(_is_zero(value) for value in vector)
    return _solve(_transpose(basis), vector, prime) is not None


def _coordinates(basis: Matrix, vector: VectorList, prime: int | None) -> VectorList:
    if not basis:
        if any(not _is_zero(value) for value in vector):
            raise ValueError("vector lies outside the empty span")
        return []
    solution = _solve(_transpose(basis), vector, prime)
    if solution is None:
        raise ValueError("vector lies outside the admitted span")
    return solution


def _mat_vec(matrix: Matrix, vector: VectorList, prime: int | None) -> VectorList:
    result: VectorList = []
    for row in matrix:
        total: Scalar = _parse_entry(0, prime)
        for left, right in zip(row, vector, strict=True):
            total = _add(total, _mul(left, right, prime), prime)
        result.append(total)
    return result


def _dot(left: VectorList, right: VectorList, prime: int | None) -> Scalar:
    total: Scalar = _parse_entry(0, prime)
    for left_value, right_value in zip(left, right, strict=True):
        total = _add(total, _mul(left_value, right_value, prime), prime)
    return total


def _mat_mul(left: Matrix, right: Matrix, prime: int | None) -> Matrix:
    if not left or not right or not right[0]:
        return [[] for _ in range(len(left))]
    columns = _transpose(right)
    return [[_dot(row, column, prime) for column in columns] for row in left]


def admit_spectral_page(page: int) -> None:
    """Shared native+catalog admission for one requested spectral page."""
    if isinstance(page, bool) or not isinstance(page, int):
        raise _fail(
            ("page",),
            "spectral_sequence.page_type_invalid",
            "the requested spectral page must be an integer",
        )
    if page < 0:
        raise _fail(
            ("page",),
            "spectral_sequence.page_negative",
            "the requested spectral page must be nonnegative",
        )
    if page > MAX_SPECTRAL_PAGE:
        raise OperationResourceAdmissionError(
            location=("page",),
            code="spectral_sequence.page_budget_exceeded",
            message="spectral pages admit at most "
            f"page {MAX_SPECTRAL_PAGE} within the bounded bidegree window",
        )


def _zero_vector(width: int, prime: int | None) -> VectorList:
    return [_parse_entry(0, prime) for _ in range(width)]


def _rank_of(rows: Matrix, prime: int | None) -> int:
    if not rows or not rows[0]:
        return 0
    _, pivots = _rref(rows, prime)
    return len(pivots)


def _nullspace(rows: Matrix, width: int, prime: int | None) -> Matrix:
    """Basis of ``{x in F^width : rows @ x == 0}`` in ambient coordinates."""
    if not rows:
        one = _parse_entry(1, prime)
        return [
            [_parse_entry(0, prime) if i != j else one for i in range(width)]
            for j in range(width)
        ]
    reduced, pivots = _rref(rows, prime)
    pivot_set = set(pivots)
    free = [column for column in range(width) if column not in pivot_set]
    basis: Matrix = []
    for column in free:
        vector = _zero_vector(width, prime)
        vector[column] = _parse_entry(1, prime)
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = _neg(reduced[row_index][column], prime)
        basis.append(vector)
    return basis


def _cycle_rows(
    level_bases: list[list[Matrix]],
    differentials: list[Matrix],
    ambient_sizes: tuple[int, ...],
    level: int,
    degree_index: int,
    cycles: int,
    prime: int | None,
) -> Matrix:
    """Reduced basis of Z^s_{p, n - p} inside ambient C_n coordinates.

    ``Z^s`` holds the level-``p`` chains whose differential lands ``s``
    levels lower; ``s = 0`` is the full filtration subspace itself.
    """
    if level < 0:
        return []
    upper = level_bases[min(level, len(level_bases) - 1)][degree_index]
    if cycles <= 0 or degree_index == 0:
        return [list(row) for row in upper]
    width = ambient_sizes[degree_index - 1]
    lower = (
        []
        if level - cycles < 0
        else level_bases[min(level - cycles, len(level_bases) - 1)][degree_index - 1]
    )
    images = [_mat_vec(differentials[degree_index - 1], row, prime) for row in upper]
    orthogonal = _nullspace(lower, width, prime)
    if not orthogonal:
        return [list(row) for row in upper]
    constraint = _mat_mul(images, _transpose(orthogonal), prime)
    coordinates = _nullspace(_transpose(constraint), len(constraint), prime)
    combined: Matrix = []
    for weights in coordinates:
        vector = _zero_vector(ambient_sizes[degree_index], prime)
        for weight, row in zip(weights, upper, strict=True):
            vector = [
                _add(current, _mul(weight, value, prime), prime)
                for current, value in zip(vector, row, strict=True)
            ]
        combined.append(vector)
    return _row_basis(combined, prime)


def _denominator_rows(
    level_bases: list[list[Matrix]],
    differentials: list[Matrix],
    ambient_sizes: tuple[int, ...],
    level: int,
    degree_index: int,
    page: int,
    prime: int | None,
) -> Matrix:
    """Reduced basis of D^r inside ambient C_n coordinates.

    ``D^r = Z^{r-1}_{p-1} + d(Z^{r-1}_{p+r-1})``; the incoming cycles are
    clamped to the exhaustive top filtration level.
    """
    degree_count = len(ambient_sizes)
    lower = (
        []
        if level - 1 < 0
        else _cycle_rows(
            level_bases,
            differentials,
            ambient_sizes,
            level - 1,
            degree_index,
            page - 1,
            prime,
        )
    )
    incoming: Matrix = []
    if degree_index + 1 < degree_count:
        rising = _cycle_rows(
            level_bases,
            differentials,
            ambient_sizes,
            level + page - 1,
            degree_index + 1,
            page - 1,
            prime,
        )
        incoming = [_mat_vec(differentials[degree_index], row, prime) for row in rising]
    return _row_basis([*lower, *incoming], prime)


def _quotient_extension(lower: Matrix, upper: Matrix, prime: int | None) -> Matrix:
    """Rows of ``upper`` extending ``lower`` to a quotient basis."""
    chosen: Matrix = []
    for candidate in upper:
        if not _in_span(lower + chosen, candidate, prime):
            chosen.append(list(candidate))
    return chosen


def _spectral_bidegree_page(  # noqa: C901
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
    level_bases: list[list[Matrix]],
    differentials: list[Matrix],
    page: int,
) -> SpectralPageResult:
    """Compute E^r for ``r >= 1`` from the admitted filtration data."""
    prime = complex_value.prime
    degree_count = len(complex_value.basis_sizes)
    level_count = len(filtration)
    ambient_sizes = tuple(complex_value.basis_sizes)

    cycle_grid: list[list[Matrix]] = [
        [[] for _ in range(degree_count)] for _ in range(level_count)
    ]
    denominator_grid: list[list[Matrix]] = [
        [[] for _ in range(degree_count)] for _ in range(level_count)
    ]
    combined_grid: list[list[Matrix]] = [
        [[] for _ in range(degree_count)] for _ in range(level_count)
    ]
    dimensions: list[list[int]] = [
        [0 for _ in range(degree_count)] for _ in range(level_count)
    ]
    representatives: list[list[list[Vector]]] = [
        [[] for _ in range(degree_count)] for _ in range(level_count)
    ]
    for level in range(level_count):
        for index in range(degree_count):
            cycles = _cycle_rows(
                level_bases, differentials, ambient_sizes, level, index, page, prime
            )
            denominator = _denominator_rows(
                level_bases, differentials, ambient_sizes, level, index, page, prime
            )
            for vector in denominator:
                if not _in_span(cycles, vector, prime):
                    raise _fail(
                        ("filtration", level),
                        "spectral_sequence.denominator_not_cycles",
                        "the page denominator must lie inside the page cycles",
                    )
            quotient = _quotient_extension(denominator, cycles, prime)
            cycle_grid[level][index] = cycles
            denominator_grid[level][index] = denominator
            combined_grid[level][index] = [*denominator, *quotient]
            dimensions[level][index] = len(quotient)
            representatives[level][index] = [
                tuple(_serialize_scalar(value, prime) for value in row)
                for row in quotient
            ]

    records: list[SpectralDifferential] = []
    scalar_records: dict[tuple[int, int], Matrix] = {}
    for level in range(level_count):
        for index in range(degree_count):
            if index == 0 or level - page < 0:
                continue
            source_degree = complex_value.degree_min + index
            target_degree = source_degree - 1
            target_level = level - page
            target_combined = combined_grid[target_level][index - 1]
            target_denominator = denominator_grid[target_level][index - 1]
            target_dim = dimensions[target_level][index - 1]
            source_dim = dimensions[level][index]
            block: Matrix = [_zero_vector(source_dim, prime) for _ in range(target_dim)]
            for column, rep in enumerate(
                combined_grid[level][index][len(denominator_grid[level][index]) :]
            ):
                image = _mat_vec(differentials[index - 1], rep, prime)
                try:
                    coordinates = _coordinates(target_combined, image, prime)
                except ValueError as exc:
                    raise _fail(
                        ("filtration", level),
                        "spectral_sequence.differential_not_closed",
                        "a page differential lands outside the target cycles",
                    ) from exc
                for row in range(target_dim):
                    block[row][column] = coordinates[len(target_denominator) + row]
            for bound in denominator_grid[level][index]:
                image = _mat_vec(differentials[index - 1], bound, prime)
                if not _in_span(target_denominator, image, prime):
                    raise _fail(
                        ("filtration", level),
                        "spectral_sequence.differential_not_well_defined",
                        "a page denominator must map into the target denominator",
                    )
            scalar_records[(level, index)] = block
            records.append(
                SpectralDifferential(
                    source_level=level,
                    source_degree=source_degree,
                    target_level=target_level,
                    target_degree=target_degree,
                    rows=target_dim,
                    columns=source_dim,
                    entries=tuple(
                        tuple(_serialize_scalar(value, prime) for value in row)
                        for row in block
                    ),
                )
            )

    by_source = {
        (record.source_level, record.source_degree): record for record in records
    }
    ledger: list[SpectralSquareLedgerEntry] = []
    for record in records:
        follower = by_source.get((record.target_level, record.target_degree))
        if follower is None:
            continue
        first = scalar_records[
            (record.source_level, record.source_degree - complex_value.degree_min)
        ]
        second = scalar_records[
            (follower.source_level, follower.source_degree - complex_value.degree_min)
        ]
        product = _mat_mul(second, first, prime)
        if any(not _is_zero(value) for row in product for value in row):
            raise _fail(
                ("filtration", record.source_level),
                "spectral_sequence.square_nonzero",
                "the page differentials must compose to zero",
            )
        ledger.append(
            SpectralSquareLedgerEntry(
                source_level=record.source_level,
                source_degree=record.source_degree,
                middle_level=record.target_level,
                middle_degree=record.target_degree,
                product_rows=follower.rows,
                product_columns=record.columns,
                nonzero_entries=0,
            )
        )

    ranks = {key: _rank_of(block, prime) for key, block in scalar_records.items()}
    next_dimensions: list[list[int]] = [
        [0 for _ in range(degree_count)] for _ in range(level_count)
    ]
    for level in range(level_count):
        for index in range(degree_count):
            outgoing = ranks.get((level, index), 0)
            incoming = ranks.get((level + page, index + 1), 0)
            next_dimensions[level][index] = (
                dimensions[level][index] - outgoing - incoming
            )
    vanishes = all(rank == 0 for rank in ranks.values())
    if vanishes and page >= level_count - 1:
        status = SpectralPageStatus.STABILIZED
    elif page >= MAX_SPECTRAL_PAGE:
        status = SpectralPageStatus.TRUNCATED
    else:
        status = SpectralPageStatus.ACTIVE
    return SpectralPageResult._from_kernel(
        complex=complex_value,
        filtration=filtration,
        page=page,
        max_page=MAX_SPECTRAL_PAGE,
        level_count=level_count,
        degree_min=complex_value.degree_min,
        degree_max=complex_value.degree_max,
        page_dimensions=tuple(tuple(row) for row in dimensions),
        page_representatives=tuple(
            tuple(tuple(degree) for degree in level) for level in representatives
        ),
        differentials=tuple(records),
        differential_squared_zero=tuple(ledger),
        next_page_dimensions=tuple(tuple(row) for row in next_dimensions),
        page_status=status,
    )


def associated_graded(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> AssociatedGradedResult:
    """Compute Gr_p C with induced differentials for an admitted filtration."""
    admission = _admit_filtered_semantics(complex_value, filtration)
    return _associated_graded_admitted(complex_value, filtration, admission)


def _associated_graded_admitted(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
    admission: _FilteredAdmission,
) -> AssociatedGradedResult:
    """Build Gr C from already admitted filtered semantics."""
    prime = complex_value.prime
    degree_count = len(complex_value.basis_sizes)
    differentials = admission.differentials
    bases = admission.bases

    graded_dimensions: list[tuple[int, ...]] = []
    representatives: list[list[list[Vector]]] = []
    graded_differentials: list[list[tuple[Vector, ...]]] = []
    ledger: list[GradedSquareLedgerEntry] = []
    previous: list[Matrix] = [[] for _ in range(degree_count)]
    for level_index in range(len(filtration)):
        lower = previous
        upper = bases[level_index]
        dims = tuple(
            len(upper[degree]) - len(lower[degree]) for degree in range(degree_count)
        )
        graded_dimensions.append(dims)
        level_reps: list[list[Vector]] = []
        rep_rows: list[Matrix] = []
        for degree in range(degree_count):
            chosen: Matrix = []
            for candidate in upper[degree]:
                if not _in_span(lower[degree] + chosen, candidate, prime):
                    chosen.append(candidate)
            rep_rows.append(chosen)
            level_reps.append(
                [
                    tuple(_serialize_scalar(value, prime) for value in row)
                    for row in chosen
                ]
            )
        representatives.append(level_reps)
        level_diffs: list[tuple[Vector, ...]] = []
        scalar_diffs: list[Matrix] = []
        for index in range(degree_count - 1):
            rows = dims[index]
            columns = dims[index + 1]
            block: Matrix = [
                [_parse_entry(0, prime) for _ in range(columns)] for _ in range(rows)
            ]
            combined_basis = lower[index] + rep_rows[index]
            for column, rep in enumerate(rep_rows[index + 1]):
                image = _mat_vec(differentials[index], rep, prime)
                coords = _coordinates(combined_basis, image, prime)
                for row in range(rows):
                    block[row][column] = coords[len(lower[index]) + row]
            scalar_diffs.append(block)
            level_diffs.append(
                tuple(
                    tuple(_serialize_scalar(value, prime) for value in row)
                    for row in block
                )
            )
        graded_differentials.append(level_diffs)
        for index in range(degree_count - 2):
            product = _mat_mul(scalar_diffs[index], scalar_diffs[index + 1], prime)
            if any(not _is_zero(value) for row in product for value in row):
                raise _fail(
                    ("filtration", level_index),
                    "filtered_chain_complex.graded_square_nonzero",
                    "the induced graded differential must square to zero",
                )
            ledger.append(
                GradedSquareLedgerEntry(
                    filtration_level=level_index,
                    degree=complex_value.degree_min + index + 1,
                    product_rows=len(product),
                    product_columns=len(product[0]) if product else 0,
                    nonzero_entries=0,
                )
            )
        previous = upper

    return AssociatedGradedResult._from_kernel(
        complex=complex_value,
        filtration=filtration,
        graded_dimensions=tuple(graded_dimensions),
        quotient_representatives=tuple(
            tuple(tuple(degree) for degree in level) for level in representatives
        ),
        graded_differentials=tuple(
            tuple(matrix for matrix in level) for level in graded_differentials
        ),
        differential_squared_zero=tuple(ledger),
    )


def _spectral_zero_page(
    graded: AssociatedGradedResult,
    page: int,
) -> SpectralPageResult:
    """Lift the associated graded value to the E^0 page result shape."""
    complex_value = graded.complex
    prime = complex_value.prime
    degree_count = len(complex_value.basis_sizes)
    records: list[SpectralDifferential] = []
    scalar_records: dict[tuple[int, int], Matrix] = {}
    for level, diffs in enumerate(graded.graded_differentials):
        for index, matrix in enumerate(diffs):
            block = [[_parse_entry(entry, prime) for entry in row] for row in matrix]
            scalar_records[(level, index + 1)] = block
            records.append(
                SpectralDifferential(
                    source_level=level,
                    source_degree=complex_value.degree_min + index + 1,
                    target_level=level,
                    target_degree=complex_value.degree_min + index,
                    rows=graded.graded_dimensions[level][index],
                    columns=graded.graded_dimensions[level][index + 1],
                    entries=matrix,
                )
            )
    by_source = {
        (record.source_level, record.source_degree): record for record in records
    }
    ledger: list[SpectralSquareLedgerEntry] = []
    for record in records:
        follower = by_source.get((record.target_level, record.target_degree))
        if follower is None:
            continue
        first = scalar_records[
            (record.source_level, record.source_degree - complex_value.degree_min)
        ]
        second = scalar_records[
            (
                follower.source_level,
                follower.source_degree - complex_value.degree_min,
            )
        ]
        product = _mat_mul(second, first, prime)
        if any(not _is_zero(value) for row in product for value in row):
            raise _fail(
                ("filtration", record.source_level),
                "spectral_sequence.square_nonzero",
                "the page differentials must compose to zero",
            )
        ledger.append(
            SpectralSquareLedgerEntry(
                source_level=record.source_level,
                source_degree=record.source_degree,
                middle_level=record.target_level,
                middle_degree=record.target_degree,
                product_rows=follower.rows,
                product_columns=record.columns,
                nonzero_entries=0,
            )
        )
    ranks = {key: _rank_of(block, prime) for key, block in scalar_records.items()}
    next_dimensions = [
        [
            graded.graded_dimensions[level][index]
            - ranks.get((level, index), 0)
            - ranks.get((level, index + 1), 0)
            for index in range(degree_count)
        ]
        for level in range(len(graded.filtration))
    ]
    vanishes = all(rank == 0 for rank in ranks.values())
    if vanishes and page >= len(graded.filtration) - 1:
        status = SpectralPageStatus.STABILIZED
    elif page >= MAX_SPECTRAL_PAGE:
        status = SpectralPageStatus.TRUNCATED
    else:
        status = SpectralPageStatus.ACTIVE
    return SpectralPageResult._from_kernel(
        complex=complex_value,
        filtration=graded.filtration,
        page=page,
        max_page=MAX_SPECTRAL_PAGE,
        level_count=len(graded.filtration),
        degree_min=complex_value.degree_min,
        degree_max=complex_value.degree_max,
        page_dimensions=graded.graded_dimensions,
        page_representatives=graded.quotient_representatives,
        differentials=tuple(records),
        differential_squared_zero=tuple(ledger),
        next_page_dimensions=tuple(tuple(row) for row in next_dimensions),
        page_status=status,
    )


def spectral_page(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
    page: int,
) -> SpectralPageResult:
    """Compute the E^r page of an admitted filtered chain complex.

    Page 0 reuses the associated-graded kernel unchanged; later pages
    form the bigraded quotients ``Z^r / D^r`` with the induced ``d^r``
    differentials of bidegree ``(-r, r - 1)`` and replay ``d^r d^r = 0``
    inside the kernel.
    """
    admit_spectral_page(page)
    if page == 0:
        return _spectral_zero_page(associated_graded(complex_value, filtration), page)
    admit_filtered(complex_value, filtration)
    prime = complex_value.prime
    degree_count = len(complex_value.basis_sizes)
    differentials = [
        [[_parse_entry(entry, prime) for entry in row] for row in matrix]
        for matrix in complex_value.differential_matrices
    ]
    for index in range(len(differentials) - 1):
        product = _mat_mul(differentials[index], differentials[index + 1], prime)
        if any(not _is_zero(value) for row in product for value in row):
            raise _fail(
                ("complex",),
                "spectral_sequence.source_not_a_complex",
                "the source differentials must satisfy d^2 = 0",
            )
    parsed_levels: list[list[Matrix]] = [
        [
            [
                [_parse_entry(entry, prime) for entry in vector]
                for vector in level.subspaces[degree].vectors
            ]
            for degree in range(degree_count)
        ]
        for level in filtration
    ]
    level_bases: list[list[Matrix]] = [
        [_row_basis(level[degree], prime) for degree in range(degree_count)]
        for level in parsed_levels
    ]
    return _spectral_bidegree_page(
        complex_value, filtration, level_bases, differentials, page
    )
