"""Native exact operations over an explicit prime field."""

from jacobian.math.matrices.finite_fields import linear_algebra
from jacobian.math.matrices.finite_fields._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldRrefResult,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientSpace,
    PrimeFieldQuotientVector,
    PrimeFieldSubspace,
)

__all__ = [
    "matrix_nullspace",
    "matrix_rank",
    "matrix_rref",
    "project_quotient_vector",
    "quotient_space",
    "verify_rank",
    "verify_rref",
]


def verify_rank(claim: PrimeFieldMatrixRankResult) -> bool:
    """Check a serialized rank claim against its retained matrix.

    The source bounds the elimination by 1024 cubed field operations;
    primality is admitted by the same native path as rank production.
    """
    return matrix_rank(claim.source.matrix) == claim.rank


def verify_rref(claim: PrimeFieldRrefResult) -> bool:
    """Check the unique reduced form, pivots, and rank against the source.

    One admitted elimination uses the source's bounded matrix envelope.
    Neither checker is invoked by result construction or deserialization.
    """
    rows, pivots = matrix_rref(claim.source.matrix)
    return (
        rows == claim.rref_matrix.entries
        and pivots == claim.pivot_columns
        and len(pivots) == claim.rank
    )


def matrix_rank(matrix: PrimeFieldMatrix) -> int:
    """Return the rank of one canonical prime-field matrix."""

    return linear_algebra.rank(matrix)


def matrix_rref(
    matrix: PrimeFieldMatrix,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return the canonical reduced row-echelon form and pivot columns."""

    return linear_algebra.rref(matrix)


def matrix_nullspace(matrix: PrimeFieldMatrix) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic basis of the canonical right nullspace."""

    return linear_algebra.nullspace(matrix)


def quotient_space(subspace: PrimeFieldSubspace) -> PrimeFieldQuotientSpace:
    """Construct V/W and its exact ambient-to-quotient coordinate map over GF(p)."""
    dimension = subspace.ambient_dimension
    prime = subspace.prime
    linear_algebra._admit_prime(prime)

    if dimension == 0:
        return PrimeFieldQuotientSpace._from_kernel(
            source=subspace,
            quotient_basis=(),
            projection=PrimeFieldMatrix(prime=prime, entries=(), columns=0),
        )

    # Select an independent subset of the supplied generators by finding
    # pivot columns in their transpose. This preserves the caller's ambient
    # coordinates and costs one admitted exact elimination.
    generator_count = len(subspace.generators)
    if generator_count:
        transposed = PrimeFieldMatrix(
            prime=prime,
            entries=tuple(
                tuple(vector[coordinate] for vector in subspace.generators)
                for coordinate in range(dimension)
            ),
            columns=generator_count,
        )
        _, generator_pivots = linear_algebra._rref_admitted(transposed)
        denominator_basis = tuple(
            subspace.generators[pivot] for pivot in generator_pivots
        )
    else:
        denominator_basis = ()

    denominator_dimension = len(denominator_basis)
    # Appending all standard basis columns chooses a deterministic complement
    # to the denominator span. The first denominator columns are independent,
    # so subsequent pivot columns identify exactly the quotient representatives.
    extended = PrimeFieldMatrix(
        prime=prime,
        entries=tuple(
            tuple(vector[coordinate] for vector in denominator_basis)
            + tuple(int(coordinate == unit) for unit in range(dimension))
            for coordinate in range(dimension)
        ),
        columns=denominator_dimension + dimension,
    )
    _, pivots = linear_algebra._rref_admitted(extended)
    complement_indices = tuple(
        pivot - denominator_dimension for pivot in pivots[denominator_dimension:]
    )
    quotient_basis = tuple(
        tuple(int(coordinate == unit) for coordinate in range(dimension))
        for unit in complement_indices
    )

    basis_rows = denominator_basis + quotient_basis
    basis_matrix = PrimeFieldMatrix(
        prime=prime,
        entries=basis_rows,
        columns=dimension,
    )
    inverse = linear_algebra._backend_matrix(basis_matrix).inv()
    # With row basis B, an ambient column vector has basis coordinates
    # (B^-1)^T v. Keep the quotient-coordinate rows of that map.
    projection_rows = tuple(
        tuple(
            int(inverse[coordinate, denominator_dimension + quotient_index])
            for coordinate in range(dimension)
        )
        for quotient_index in range(len(quotient_basis))
    )
    projection = PrimeFieldMatrix(
        prime=prime,
        entries=projection_rows,
        columns=dimension,
    )
    return PrimeFieldQuotientSpace._from_kernel(
        source=subspace,
        quotient_basis=quotient_basis,
        projection=projection,
    )


def project_quotient_vector(
    quotient: PrimeFieldQuotientSpace, vector: tuple[int, ...]
) -> PrimeFieldQuotientVector:
    """Map an ambient vector to coordinates in its source-bound quotient."""
    _admit_quotient_projection(quotient)
    projection = quotient.projection
    coordinates = tuple(
        sum(row[index] * vector[index] for index in range(projection.columns))
        % projection.prime
        for row in projection.entries
    )
    return PrimeFieldQuotientVector._from_kernel(quotient, coordinates)


def _admit_quotient_projection(quotient: PrimeFieldQuotientSpace) -> None:
    """Check the caller-supplied quotient map before relying on its coordinates."""
    from jacobian.catalog.models import OperationDomainValidationError

    subspace = quotient.source
    dimension = subspace.ambient_dimension
    prime = subspace.prime
    linear_algebra._admit_prime(prime)
    generator_count = len(subspace.generators)
    if generator_count:
        generator_columns = PrimeFieldMatrix(
            prime=prime,
            entries=tuple(
                tuple(vector[coordinate] for vector in subspace.generators)
                for coordinate in range(dimension)
            ),
            columns=generator_count,
        )
        _, denominator_pivots = linear_algebra._rref_admitted(generator_columns)
        denominator_rank = len(denominator_pivots)
    else:
        denominator_rank = 0

    quotient_dimension = len(quotient.quotient_basis)
    quotient_basis_matrix = PrimeFieldMatrix(
        prime=prime,
        entries=quotient.quotient_basis,
        columns=dimension,
    )
    _, basis_pivots = linear_algebra._rref_admitted(quotient_basis_matrix)
    if (
        denominator_rank + quotient_dimension != dimension
        or len(basis_pivots) != quotient_dimension
    ):
        raise OperationDomainValidationError(
            location=("quotient",),
            code="prime_field_quotient.dimension_invalid",
            message="denominator rank and quotient basis must span the full ambient dimension",
        )

    for generator in subspace.generators:
        if any(
            sum(
                entry * coordinate
                for entry, coordinate in zip(row, generator, strict=True)
            )
            % prime
            for row in quotient.projection.entries
        ):
            raise OperationDomainValidationError(
                location=("quotient", "projection"),
                code="prime_field_quotient.denominator_not_kernel",
                message="projection must annihilate every denominator generator",
            )
    for quotient_index, basis_vector in enumerate(quotient.quotient_basis):
        for row_index, row in enumerate(quotient.projection.entries):
            coordinate = (
                sum(
                    entry * value
                    for entry, value in zip(row, basis_vector, strict=True)
                )
                % prime
            )
            if coordinate != int(row_index == quotient_index):
                raise OperationDomainValidationError(
                    location=("quotient", "projection"),
                    code="prime_field_quotient.basis_coordinates_invalid",
                    message="projection must send its quotient basis to standard coordinates",
                )
