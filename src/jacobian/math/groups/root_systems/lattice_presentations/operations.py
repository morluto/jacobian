"""Construction of source-bound root/weight lattice values."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    FiniteCartanDatum,
)
from jacobian.math.groups.root_systems.lattice_presentations._models import (
    RootWeightLatticePresentation,
)
from jacobian.math.groups.root_systems.operations import cartan_datum
from jacobian.math.lattices._models import IntegerLattice
from jacobian.math.matrices.values import IntegerMatrix

MAX_ROOT_WEIGHT_PRESENTATION_CELLS = 6 * MAX_RANK**2 + 3 * MAX_RANK
MAX_ROOT_WEIGHT_PRESENTATION_WORK = MAX_RANK**3


def root_weight_lattice_presentation(
    datum: FiniteCartanDatum,
) -> RootWeightLatticePresentation:
    """Return Q embedded in P using the datum's exact Cartan basis map."""
    if type(datum) is not FiniteCartanDatum:
        raise OperationDomainValidationError(
            location=("datum",),
            code="root_system.lattice_presentation.datum_type",
            message="a canonical finite Cartan datum is required",
        )
    try:
        rank = len(datum.cartan_matrix)
        if not 1 <= rank <= MAX_RANK:
            raise ValueError("invalid Cartan rank")
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("datum",),
            code="root_system.lattice_presentation.invalid_datum",
            message="datum must contain admitted finite Cartan basis maps",
        ) from error

    matrix_cells = rank * rank
    work_units = rank**3
    output_cells = 6 * matrix_cells + 3 * rank
    if (
        work_units > MAX_ROOT_WEIGHT_PRESENTATION_WORK
        or output_cells > MAX_ROOT_WEIGHT_PRESENTATION_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("datum",),
            code="root_system.lattice_presentation.resource_bound",
            message="root/weight lattice presentation exceeds its work/output envelope",
        )

    canonical = cartan_datum(datum.cartan_matrix)
    if datum != canonical:
        raise OperationDomainValidationError(
            location=("datum",),
            code="root_system.lattice_presentation.datum_mismatch",
            message="Cartan basis maps must equal the canonical finite datum maps",
        )

    identity = tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )
    # Cartan columns give simple roots in fundamental-weight coordinates;
    # IntegerLattice stores basis vectors as rows, so transpose the map.
    embedding = tuple(
        tuple(canonical.root_to_weight.entries[column][row] for column in range(rank))
        for row in range(rank)
    )
    identity_value = IntegerMatrix(
        row_count=rank,
        column_count=rank,
        entries=identity,
    )
    embedding_value = IntegerMatrix(
        row_count=rank,
        column_count=rank,
        entries=embedding,
    )
    return RootWeightLatticePresentation(
        datum=canonical,
        root_lattice=IntegerLattice(
            ambient_dimension=rank,
            basis=embedding_value,
        ),
        weight_lattice=IntegerLattice(
            ambient_dimension=rank,
            basis=identity_value,
        ),
        root_to_weight_embedding=embedding_value,
    )
