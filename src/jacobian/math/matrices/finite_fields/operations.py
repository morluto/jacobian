"""Native exact operations over an explicit prime field."""

from collections.abc import Callable
from typing import NoReturn

from pydantic_core import PydanticCustomError

from jacobian.math.matrices.finite_fields import linear_algebra, quotient_spaces
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_ELIMINATION_WORK,
)
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
    subspace = _admit_subspace(subspace)
    dimension = subspace.ambient_dimension
    prime = subspace.prime

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
    _admit_projection_request(quotient, vector)
    projection = quotient.projection
    coordinates = tuple(
        sum(row[index] * vector[index] for index in range(projection.columns))
        % projection.prime
        for row in projection.entries
    )
    return PrimeFieldQuotientVector._from_kernel(quotient, coordinates)


def _domain_rejection(
    location: tuple[str | int, ...], code: str, message: str
) -> NoReturn:
    from jacobian.catalog.models import OperationDomainValidationError

    raise OperationDomainValidationError(location=location, code=code, message=message)


def _run_admission(
    admission: Callable[[], None], *, location: tuple[str | int, ...]
) -> None:
    """Run a shared structural check, normalizing its wire error for native callers."""
    from jacobian.catalog.models import OperationDomainValidationError

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def _admit_subspace(subspace: object) -> PrimeFieldSubspace:
    """Admit a possibly forged source subspace before elimination runs.

    ``model_copy`` and ``model_construct`` bypass Pydantic's nested validators,
    so native callers must establish the carrier invariants themselves rather
    than trusting an already-instantiated ``PrimeFieldSubspace``.
    """
    if type(subspace) is not PrimeFieldSubspace:
        _domain_rejection(
            ("subspace",),
            "prime_field_quotient.subspace_type",
            "quotient construction requires a PrimeFieldSubspace value",
        )
    _run_admission(
        lambda: quotient_spaces._require_canonical_generators(
            subspace.prime, subspace.ambient_dimension, subspace.generators
        ),
        location=("subspace",),
    )
    _run_admission(
        lambda: quotient_spaces._require_quotient_envelope(
            subspace.ambient_dimension, len(subspace.generators)
        ),
        location=("subspace",),
    )
    linear_algebra._admit_prime(subspace.prime)
    return subspace


def _admit_projection_request(
    quotient: object, vector: object
) -> None:
    """Admit a quotient carrier, its authored relation, and the ambient vector.

    The quotient value is caller-supplied and may have been built by
    ``model_copy`` or ``model_construct``, so re-establish its field, axis,
    residue, and kernel invariants instead of trusting decoding.
    """
    if type(quotient) is not PrimeFieldQuotientSpace:
        _domain_rejection(
            ("quotient",),
            "prime_field_quotient.quotient_type",
            "projection requires a PrimeFieldQuotientSpace value",
        )
    source = quotient.source
    if type(source) is not PrimeFieldSubspace:
        _domain_rejection(
            ("quotient", "source"),
            "prime_field_quotient.subspace_type",
            "quotient source must be a PrimeFieldSubspace value",
        )
    _run_admission(
        lambda: quotient_spaces._require_canonical_generators(
            source.prime, source.ambient_dimension, source.generators
        ),
        location=("quotient", "source"),
    )
    _run_admission(
        lambda: quotient_spaces._require_quotient_structure(
            source, quotient.quotient_basis, quotient.projection
        ),
        location=("quotient",),
    )
    prime = source.prime
    dimension = source.ambient_dimension
    linear_algebra._admit_prime(prime)
    _admit_projection_vector(vector, dimension=dimension, prime=prime)
    _admit_projection_relation(quotient)


def _admit_projection_vector(
    vector: object, *, dimension: int, prime: int
) -> None:
    """Admit the ambient vector against the quotient's declared axis and field."""
    if type(vector) is not tuple:
        _domain_rejection(
            ("vector",),
            "prime_field_quotient.vector_type",
            "vector must be a tuple of GF(p) coordinates",
        )
    if len(vector) != dimension:
        _domain_rejection(
            ("vector",),
            "prime_field_quotient.vector_axis",
            "vector must match the quotient ambient axis",
        )
    if any(type(entry) is not int or not 0 <= entry < prime for entry in vector):
        _domain_rejection(
            ("vector",),
            "prime_field_quotient.vector_residue",
            "vector entries must be canonical residues in GF(p)",
        )


def _admit_projection_relation(quotient: PrimeFieldQuotientSpace) -> None:
    """Check the caller-supplied quotient map before relying on its coordinates."""
    subspace = quotient.source
    dimension = subspace.ambient_dimension
    prime = subspace.prime
    generator_count = len(subspace.generators)
    quotient_dimension = len(quotient.quotient_basis)
    validation_work = (
        dimension * generator_count * min(dimension, generator_count)
        + dimension * quotient_dimension * min(dimension, quotient_dimension)
        + quotient_dimension * dimension * generator_count
        + quotient_dimension * dimension * quotient_dimension
    )
    if validation_work > MAX_PRIME_FIELD_ELIMINATION_WORK:
        _domain_rejection(
            ("quotient",),
            "prime_field_quotient.projection_validation_work",
            "quotient proof exceeds the elimination work bound",
        )
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
        _domain_rejection(
            ("quotient",),
            "prime_field_quotient.dimension_invalid",
            "denominator rank and quotient basis must span the full ambient dimension",
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
            _domain_rejection(
                ("quotient", "projection"),
                "prime_field_quotient.denominator_not_kernel",
                "projection must annihilate every denominator generator",
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
                _domain_rejection(
                    ("quotient", "projection"),
                    "prime_field_quotient.basis_coordinates_invalid",
                    "projection must send its quotient basis to standard coordinates",
                )
