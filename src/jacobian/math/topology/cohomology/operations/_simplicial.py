"""Bounded prime-field simplicial cohomology via the dual cochain complex.

For a canonical :class:`FiniteSimplicialComplex` and a prime ``p`` this module
builds the cochain groups ``C^k = Hom(C_k, GF(p))`` with coboundaries
``delta^k = transpose(boundary_{k+1})`` and returns exact cocycle,
coboundary, and cohomology-class bases with ``dim H^k`` cross-checked
against the existing prime-field homology Betti number.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.finite_fields import linear_algebra as prime_field
from jacobian.math.topology._homology import (
    MAX_INLINE_HOMOLOGY_CHAIN_GROUP,
    ModularVector,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_CHAIN_GROUP,
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_PRIME,
    FiniteSimplicialComplex,
    HomologyConvention,
    _require_canonical_conversion_bounds,
    _validation_error,
    is_bounded_prime,
    require_linear_algebra_bounds,
)
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
    run_topology_admission,
)

MAX_COHOMOLOGY_CHAIN_GROUP = MAX_INLINE_HOMOLOGY_CHAIN_GROUP
"""Inline cochain-basis bound shared with prime-field homology.

Cohomology returns complete cocycle/coboundary/class bases, so it admits the
same per-degree simplex bound as the existing inline homology bases.
"""


class SimplicialCohomologyRequest(StrictModel):
    """Compute prime-field cohomology of a bounded finite simplicial complex."""

    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED


class CohomologyGroupResult(StrictModel):
    """Exact cochain data and quotient basis in one cohomological degree."""

    dimension: StrictInt = Field(ge=-1, le=MAX_TOPOLOGY_DIMENSION)
    cochain_dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    outgoing_coboundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cocycle_dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    incoming_coboundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    betti_number: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cocycle_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    coboundary_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    cohomology_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    quotient_span_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)

    @model_validator(mode="after")
    def require_dimension_ledger(self) -> Self:
        if self.cocycle_dimension != (
            self.cochain_dimension - self.outgoing_coboundary_rank
        ):
            raise _validation_error(
                "topology.require_cohomology_ledger_1",
                "cocycle dimension does not equal cochain nullity",
            )
        if self.betti_number != (
            self.cocycle_dimension - self.incoming_coboundary_rank
        ):
            raise _validation_error(
                "topology.require_cohomology_ledger_2",
                "cohomology Betti number does not equal cocycles minus coboundaries",
            )
        if (
            len(self.cocycle_basis) != self.cocycle_dimension
            or len(self.coboundary_basis) != self.incoming_coboundary_rank
            or len(self.cohomology_basis) != self.betti_number
            or self.quotient_span_rank != self.cocycle_dimension
        ):
            raise _validation_error(
                "topology.require_cohomology_ledger_3",
                "cohomology bases do not match the dimension ledger",
            )
        vectors = (*self.cocycle_basis, *self.coboundary_basis, *self.cohomology_basis)
        if any(
            len(vector.coefficients) != self.cochain_dimension for vector in vectors
        ):
            raise _validation_error(
                "topology.require_cohomology_ledger_4",
                "cohomology vector does not use the declared cochain basis",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the cohomology kernel established all derived fields."""

        return cls.model_construct(**values)


class SimplicialCohomologyResult(StrictModel):
    """Exact prime-field cohomology of one canonical simplicial complex."""

    complex: FiniteSimplicialComplex
    coefficient_field: Literal["PRIME_FIELD"] = "PRIME_FIELD"
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    dimension_range: tuple[StrictInt, StrictInt]
    groups: tuple[CohomologyGroupResult, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_complete_dimension_range(self) -> Self:
        dimensions = tuple(group.dimension for group in self.groups)
        empty_reduced = (
            self.complex.dimension == -1
            and self.convention is HomologyConvention.REDUCED
        )
        expected_dimensions = (-1,) if empty_reduced else tuple(range(len(self.groups)))
        expected_range = (-1, -1) if empty_reduced else (0, len(self.groups) - 1)
        if dimensions != expected_dimensions:
            raise _validation_error(
                "topology.require_cohomology_range_2",
                "cohomology groups must cover contiguous dimensions",
            )
        if self.dimension_range != expected_range:
            raise _validation_error(
                "topology.require_cohomology_range_3",
                "dimension_range does not cover every returned group",
            )
        if any(
            coefficient < 0 or coefficient >= self.prime
            for group in self.groups
            for vector in (
                *group.cocycle_basis,
                *group.coboundary_basis,
                *group.cohomology_basis,
            )
            for coefficient in vector.coefficients
        ):
            raise _validation_error(
                "topology.require_cohomology_range_4",
                "cohomology vector coefficient is outside the prime field",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build after the cohomology kernel established all derived fields."""

        return cls.model_construct(**values)


def _raise_invalid(*, location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def require_simplicial_cohomology_admission(
    complex_: FiniteSimplicialComplex,
    prime: int,
    convention: HomologyConvention,
) -> None:
    """Share one admission path between native and catalog invocations."""

    def admit() -> None:
        require_canonical_complex_admission(complex_)
        if not is_bounded_prime(prime):
            raise ValueError(
                f"prime must be a prime integer at most {MAX_TOPOLOGY_PRIME}"
            )
        if any(size > MAX_COHOMOLOGY_CHAIN_GROUP for size in complex_.f_vector):
            from jacobian.catalog.models import OperationResourceAdmissionError

            raise OperationResourceAdmissionError(
                location=("complex",),
                code="topology.cohomology.source_budget",
                message=(
                    "cohomology bases require at most "
                    f"{MAX_COHOMOLOGY_CHAIN_GROUP} simplices in each cochain group"
                ),
            )
        require_linear_algebra_bounds(complex_)
        _require_canonical_conversion_bounds(complex_, convention)

    run_topology_admission(admit, location=("complex",))


def _dense_boundary(
    complex_: FiniteSimplicialComplex, dimension: int, *, prime: int
) -> list[list[int]]:
    """Return the dense oriented boundary matrix d_dimension over GF(prime)."""

    source = complex_.faces_by_dimension[dimension].faces
    if dimension == 0:
        return []
    target = complex_.faces_by_dimension[dimension - 1].faces
    row_for_face = {face: index for index, face in enumerate(target)}
    dense = [[0] * len(source) for _ in range(len(target))]
    for column, simplex in enumerate(source):
        for removed in range(len(simplex)):
            face = simplex[:removed] + simplex[removed + 1 :]
            value = 1 if removed % 2 == 0 else prime - 1
            dense[row_for_face[face]][column] = value
    return dense


def _transpose(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix, strict=True)]


def _mat_vec_mod(
    matrix: list[list[int]], vector: tuple[int, ...], *, prime: int
) -> list[int]:
    return [
        sum(entry * coefficient for entry, coefficient in zip(row, vector, strict=True))
        % prime
        for row in matrix
    ]


def _mat_mat_mod(
    left: list[list[int]], right: list[list[int]], *, prime: int
) -> list[list[int]]:
    if not left or not right:
        return []
    inner = len(right)
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(inner)) % prime
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _prime_matrix(
    rows: list[list[int]], *, columns: int, prime: int
) -> prime_field.PrimeFieldMatrix:
    if rows:
        if any(len(row) != len(rows[0]) for row in rows):
            raise OperationDomainValidationError(
                location=("complex",),
                code="topology.cohomology.matrix_shape",
                message="coboundary rows do not share one column axis",
            )
        columns = len(rows[0])
    return prime_field.PrimeFieldMatrix(
        prime=prime,
        entries=tuple(tuple(value % prime for value in row) for row in rows),
        columns=columns,
    )


def simplicial_cohomology(  # noqa: C901
    complex_: FiniteSimplicialComplex,
    prime: int,
    convention: HomologyConvention = HomologyConvention.UNREDUCED,
) -> SimplicialCohomologyResult:
    """Compute exact prime-field cohomology with cross-checked Betti numbers.

    Builds ``C^k = Hom(C_k, GF(p))`` with ``delta^k = transpose(d_{k+1})``,
    returns per-degree cocycle/coboundary/class bases, and requires
    ``dim H^k`` to agree with the existing prime-field homology Betti number.
    """

    if not isinstance(complex_, FiniteSimplicialComplex):
        _raise_invalid(
            location=("complex",),
            code="topology.cohomology.complex_type",
            message="complex must be a FiniteSimplicialComplex value",
        )
    if type(prime) is not int:
        _raise_invalid(
            location=("prime",),
            code="topology.cohomology.prime_type",
            message="prime must be a prime integer",
        )
    if not isinstance(convention, HomologyConvention):
        _raise_invalid(
            location=("convention",),
            code="topology.cohomology.convention_type",
            message="convention must be a HomologyConvention",
        )
    require_simplicial_cohomology_admission(complex_, prime, convention)

    dimension = complex_.dimension
    if dimension == -1:
        if convention is HomologyConvention.UNREDUCED:
            group = CohomologyGroupResult._from_kernel(
                dimension=0,
                cochain_dimension=0,
                outgoing_coboundary_rank=0,
                cocycle_dimension=0,
                incoming_coboundary_rank=0,
                betti_number=0,
                cocycle_basis=(),
                coboundary_basis=(),
                cohomology_basis=(),
                quotient_span_rank=0,
            )
            return SimplicialCohomologyResult.model_construct(
                complex=complex_,
                prime=prime,
                convention=convention,
                dimension_range=(0, 0),
                groups=(group,),
            )
        group = CohomologyGroupResult._from_kernel(
            dimension=-1,
            cochain_dimension=1,
            outgoing_coboundary_rank=0,
            cocycle_dimension=1,
            incoming_coboundary_rank=0,
            betti_number=1,
            cocycle_basis=(ModularVector(coefficients=(1,)),),
            coboundary_basis=(),
            cohomology_basis=(ModularVector(coefficients=(1,)),),
            quotient_span_rank=1,
        )
        return SimplicialCohomologyResult.model_construct(
            complex=complex_,
            prime=prime,
            convention=convention,
            dimension_range=(-1, -1),
            groups=(group,),
        )
    cochain_sizes = list(complex_.f_vector)
    boundaries = [
        _dense_boundary(complex_, degree, prime=prime)
        for degree in range(dimension + 1)
    ]
    # Coboundary delta^k = transpose(d_{k+1}); delta^dimension is the zero map
    # out of C^dimension, represented with zero rows.
    coboundaries = [
        _transpose(boundaries[degree + 1]) if degree < dimension else []
        for degree in range(dimension + 1)
    ]
    augmentation: list[list[int]] | None = None
    if convention is HomologyConvention.REDUCED:
        augmentation = [[1] * cochain_sizes[0]]

    groups: list[CohomologyGroupResult] = []
    for degree in range(dimension):
        upper = coboundaries[degree]
        outer = coboundaries[degree + 1]
        if upper and outer:
            product = _mat_mat_mod(outer, upper, prime=prime)
            if any(entry != 0 for row in product for entry in row):
                raise OperationDomainValidationError(
                    location=("complex",),
                    code="topology.cohomology.coboundary_square",
                    message="consecutive coboundaries do not compose to zero",
                )
    for degree in range(dimension + 1):
        size = cochain_sizes[degree]
        outgoing = coboundaries[degree]
        outgoing_matrix = _prime_matrix(outgoing, columns=size, prime=prime)
        cocycles = prime_field._nullspace_admitted(outgoing_matrix)
        outgoing_rank = size - len(cocycles)
        if degree == 0 and convention is HomologyConvention.REDUCED:
            incoming_rows = _transpose(augmentation or [])
            incoming_columns = 1
        elif degree == 0:
            incoming_rows = []
            incoming_columns = size
        else:
            incoming_rows = coboundaries[degree - 1]
            incoming_columns = size
        incoming_matrix = _prime_matrix(
            incoming_rows,
            columns=incoming_columns,
            prime=prime,
        )
        # Column space of delta^{k-1} expressed in C^k coordinates. For the
        # reduced degree-zero map the single column already lives in C^0.
        coboundary_basis = prime_field._column_basis_admitted(incoming_matrix)
        cohomology_basis, quotient_span_rank = prime_field._quotient_extension_admitted(
            cocycles, coboundary_basis, prime=prime
        )
        # Replay the defining kernel inclusions before trusted construction.
        for vector in cocycles:
            if any(value != 0 for value in _mat_vec_mod(outgoing, vector, prime=prime)):
                raise OperationDomainValidationError(
                    location=("complex",),
                    code="topology.cohomology.cocycle_kernel",
                    message="a cocycle basis vector lies outside the kernel",
                )
        for vector in coboundary_basis:
            if any(value != 0 for value in _mat_vec_mod(outgoing, vector, prime=prime)):
                raise OperationDomainValidationError(
                    location=("complex",),
                    code="topology.cohomology.coboundary_closure",
                    message="a coboundary basis vector is not a cocycle",
                )
        groups.append(
            CohomologyGroupResult._from_kernel(
                dimension=degree,
                cochain_dimension=size,
                outgoing_coboundary_rank=outgoing_rank,
                cocycle_dimension=len(cocycles),
                incoming_coboundary_rank=len(coboundary_basis),
                betti_number=len(cohomology_basis),
                cocycle_basis=tuple(
                    ModularVector(coefficients=vector) for vector in cocycles
                ),
                coboundary_basis=tuple(
                    ModularVector(coefficients=vector) for vector in coboundary_basis
                ),
                cohomology_basis=tuple(
                    ModularVector(coefficients=vector) for vector in cohomology_basis
                ),
                quotient_span_rank=quotient_span_rank,
            )
        )

    # Duality cross-check against the existing prime-field homology operation.
    from jacobian.math.topology._simplicial_kernel import homology as _homology_kernel

    homology_result = _homology_kernel(complex_, prime, convention)
    for group, homology_group in zip(groups, homology_result.groups, strict=True):
        if group.betti_number != homology_group.betti_number:
            raise OperationDomainValidationError(
                location=("complex",),
                code="topology.cohomology.duality_mismatch",
                message=(
                    f"dim H^{group.dimension}={group.betti_number} disagrees with "
                    f"homology Betti number {homology_group.betti_number}"
                ),
            )

    return SimplicialCohomologyResult._from_kernel(
        complex=complex_,
        coefficient_field="PRIME_FIELD",
        prime=prime,
        convention=convention,
        orientation_convention="LEXICOGRAPHIC_VERTEX_ORDER",
        dimension_range=(0, dimension),
        groups=tuple(groups),
    )


__all__ = [
    "CohomologyGroupResult",
    "SimplicialCohomologyRequest",
    "SimplicialCohomologyResult",
    "require_simplicial_cohomology_admission",
    "simplicial_cohomology",
]
