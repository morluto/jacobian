"""Exact associated-graded kernel for filtered chain complexes."""

from __future__ import annotations

from fractions import Fraction

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_LEVELS,
    MAX_FILTER_VECTORS_PER_GROUP,
    AssociatedGradedResult,
    FiltrationLevel,
    GradedSquareLedgerEntry,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    require_prime_field_admission,
)

Scalar = Fraction | int
Matrix = list[list[Scalar]]
VectorList = list[Scalar]

__all__ = ["admit_filtered", "associated_graded"]


def _fail(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def admit_filtered(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> None:
    """Shared native+catalog admission for filtered complexes."""
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
        _require_rational_entry_grammar,
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
                if (
                    not isinstance(vector, tuple)
                    or len(vector) != sizes[degree_index]
                    or any(not isinstance(entry, str) for entry in vector)
                ):
                    raise _fail(
                        ("filtration", level_index, degree_index),
                        "filtered_chain_complex.vector_axis_invalid",
                        "filtration vectors must use ambient chain coordinates",
                    )
                for entry in vector:
                    try:
                        _require_rational_entry_grammar(
                            complex_value.coefficient_ring,
                            entry,
                            prime=complex_value.prime,
                        )
                    except Exception as exc:
                        raise _fail(
                            ("filtration", level_index, degree_index),
                            "filtered_chain_complex.entry_grammar_invalid",
                            f"filtration entry '{entry}' is not canonical",
                        ) from exc


def _parse_entry(entry: str, prime: int | None) -> Scalar:
    if prime is not None:
        return int(entry) % prime
    if "/" in entry:
        numerator, _, denominator = entry.partition("/")
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(entry))


def _serialize_scalar(value: Scalar, prime: int | None) -> str:
    if prime is not None:
        assert isinstance(value, int)
        return str(value % prime)
    assert isinstance(value, Fraction)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _add(left: Scalar, right: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        assert isinstance(left, int) and isinstance(right, int)
        return (left + right) % prime
    assert isinstance(left, Fraction) and isinstance(right, Fraction)
    return left + right


def _mul(left: Scalar, right: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        assert isinstance(left, int) and isinstance(right, int)
        return (left * right) % prime
    assert isinstance(left, Fraction) and isinstance(right, Fraction)
    return left * right


def _neg(value: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        assert isinstance(value, int)
        return (-value) % prime
    assert isinstance(value, Fraction)
    return -value


def _inv(value: Scalar, prime: int | None) -> Scalar:
    if prime is not None:
        assert isinstance(value, int)
        return pow(value % prime, -1, prime)
    assert isinstance(value, Fraction)
    return Fraction(1, 1) / value


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
    solution: VectorList = [_parse_entry("0", prime) for _ in range(width)]
    for row, pivot in zip(reduced, pivots, strict=True):
        if pivot < width:
            solution[pivot] = row[width]
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
        total: Scalar = _parse_entry("0", prime)
        for left, right in zip(row, vector, strict=True):
            total = _add(total, _mul(left, right, prime), prime)
        result.append(total)
    return result


def _dot(left: VectorList, right: VectorList, prime: int | None) -> Scalar:
    total: Scalar = _parse_entry("0", prime)
    for left_value, right_value in zip(left, right, strict=True):
        total = _add(total, _mul(left_value, right_value, prime), prime)
    return total


def _mat_mul(left: Matrix, right: Matrix, prime: int | None) -> Matrix:
    if not left or not right or not right[0]:
        return [[] for _ in range(len(left))]
    columns = _transpose(right)
    return [[_dot(row, column, prime) for column in columns] for row in left]


def associated_graded(  # noqa: C901
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
) -> AssociatedGradedResult:
    """Compute Gr_p C with induced differentials for an admitted filtration."""
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

    graded_dimensions: list[tuple[int, ...]] = []
    representatives: list[list[list[tuple[str, ...]]]] = []
    graded_differentials: list[list[tuple[tuple[str, ...], ...]]] = []
    ledger: list[GradedSquareLedgerEntry] = []
    previous: list[Matrix] = [[] for _ in range(degree_count)]
    for level_index in range(len(filtration)):
        lower = previous
        upper = bases[level_index]
        dims = tuple(
            len(upper[degree]) - len(lower[degree]) for degree in range(degree_count)
        )
        graded_dimensions.append(dims)
        level_reps: list[list[tuple[str, ...]]] = []
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
        level_diffs: list[tuple[tuple[str, ...], ...]] = []
        scalar_diffs: list[Matrix] = []
        for index in range(degree_count - 1):
            rows = dims[index]
            columns = dims[index + 1]
            block: Matrix = [
                [_parse_entry("0", prime) for _ in range(columns)] for _ in range(rows)
            ]
            for column, rep in enumerate(rep_rows[index + 1]):
                image = _mat_vec(differentials[index], rep, prime)
                coords = _coordinates(upper[index], image, prime)
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
