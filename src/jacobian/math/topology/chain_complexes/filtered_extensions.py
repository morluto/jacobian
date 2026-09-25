"""Filtered-map transport and explicit finite stabilization profiles."""

from __future__ import annotations

from fractions import Fraction
from itertools import chain
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_VECTORS_PER_GROUP,
    MAX_SPECTRAL_PAGE,
    FilteredChainComplexRequest,
    FiltrationLevel,
    SpectralPageResult,
    Vector,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    _admit_filtered_semantics,
    _admit_filtered_structure,
    _associated_graded_admitted,
    _coordinates,
    _denominator_rows,
    _fail,
    _in_span,
    _mat_vec,
    _nullspace,
    _parse_entry,
    _quotient_extension,
    _rank_of,
    _row_basis,
    _serialize_scalar,
    _solve,
    _spectral_bidegree_page,
    _spectral_zero_page,
    _transpose,
    admit_filtered,
    admit_spectral_page,
    spectral_page,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS,
    MAX_CHAIN_MAP_CELLS,
    MAX_CHAIN_MAP_ENTRY_CHARS,
    ChainCoefficient,
    ChainComplexValue,
    CoefficientRing,
)

MAX_FILTERED_HOMOLOGY_PRIME = 2**31 - 1
MAX_FILTERED_HOMOLOGY_RESULT_CELLS = 250_000
MAX_FILTERED_HOMOLOGY_RESULT_CHARS = 20_000_000
MAX_FILTERED_HOMOLOGY_WORK = 50_000_000
# Every output cell contracts at most one admitted 32-dimensional chain group.
MAX_FILTERED_CHAIN_MAP_COMPOSITION_WORK = (
    MAX_CHAIN_MAP_CELLS * MAX_FILTER_AMBIENT_DIMENSION
)


class FilteredHomologyDegree(StrictModel):
    """A source-bound cycle, boundary, and quotient basis in one degree."""

    kind: Literal["FILTERED_HOMOLOGY_VECTOR_SPACE"] = "FILTERED_HOMOLOGY_VECTOR_SPACE"
    degree: int
    cycle_basis: tuple[Vector, ...]
    boundary_basis: tuple[Vector, ...]
    homology_basis: tuple[Vector, ...]


class HomologyFiltrationImage(StrictModel):
    """One image subspace in H_n with chain-level representatives."""

    basis_coordinates: tuple[Vector, ...]
    cycle_representatives: tuple[Vector, ...]
    boundary_preimages: tuple[Vector, ...]


class FilteredHomologyLevel(StrictModel):
    """The images of F_p C_n in H_n for every supported chain degree."""

    subspaces: tuple[HomologyFiltrationImage, ...]


class FilteredHomologyResult(StrictModel):
    """Exact homology bases and their source-bound induced filtration."""

    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...]
    homology: tuple[FilteredHomologyDegree, ...]
    image_filtration: tuple[FilteredHomologyLevel, ...]

    @model_validator(mode="after")
    def require_axes(self) -> Self:
        dimensions = self.complex.basis_sizes
        if len(self.homology) != len(dimensions):
            raise ValueError("homology values must cover every source degree")
        if len(self.image_filtration) != len(self.filtration):
            raise ValueError("image filtration must retain every source level")
        for index, (group, ambient_dimension) in enumerate(
            zip(self.homology, dimensions, strict=True)
        ):
            if (
                group.degree != self.complex.degree_min + index
                or any(len(vector) != ambient_dimension for vector in group.cycle_basis)
                or any(
                    len(vector) != ambient_dimension for vector in group.boundary_basis
                )
                or any(
                    len(vector) != ambient_dimension for vector in group.homology_basis
                )
                or len(group.cycle_basis)
                != len(group.boundary_basis) + len(group.homology_basis)
            ):
                raise ValueError("homology basis axes or rank ledger are inconsistent")
        for level in self.image_filtration:
            if len(level.subspaces) != len(dimensions):
                raise ValueError("each image level must cover every source degree")
            for index, subspace in enumerate(level.subspaces):
                h_dimension = len(self.homology[index].homology_basis)
                rank = len(subspace.basis_coordinates)
                if (
                    len(subspace.cycle_representatives) != rank
                    or len(subspace.boundary_preimages) != rank
                    or any(
                        len(vector) != h_dimension
                        for vector in subspace.basis_coordinates
                    )
                    or any(
                        len(vector) != dimensions[index]
                        for vector in subspace.cycle_representatives
                    )
                    or any(
                        len(vector)
                        != (dimensions[index + 1] if index + 1 < len(dimensions) else 0)
                        for vector in subspace.boundary_preimages
                    )
                ):
                    raise ValueError("image-space witnesses do not match their axes")
        return self


class FilteredChainMapRequest(StrictModel):
    source: ChainComplexValue
    source_filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    target: ChainComplexValue
    target_filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    maps: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]


class FilteredChainMapResult(StrictModel):
    source: ChainComplexValue
    target: ChainComplexValue
    source_filtration: tuple[FiltrationLevel, ...]
    target_filtration: tuple[FiltrationLevel, ...]
    maps: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]
    filtration_preserving: bool
    chain_map: bool


class FilteredChainMapPageRequest(StrictModel):
    """Induce one bounded spectral-sequence page map."""

    map: FilteredChainMapResult
    page: int = Field(ge=0, le=MAX_SPECTRAL_PAGE)


class FilteredChainMapPageResult(StrictModel):
    """The map on one page in the retained source and target quotient bases."""

    map: FilteredChainMapResult
    source_page: SpectralPageResult
    target_page: SpectralPageResult
    maps: tuple[tuple[tuple[tuple[ChainCoefficient, ...], ...], ...], ...]

    @model_validator(mode="after")
    def require_page_axes(self) -> Self:
        if (
            self.source_page.complex != self.map.source
            or self.target_page.complex != self.map.target
            or self.source_page.filtration != self.map.source_filtration
            or self.target_page.filtration != self.map.target_filtration
            or self.source_page.page != self.target_page.page
            or len(self.maps) != len(self.source_page.page_dimensions)
            or len(self.maps) != len(self.target_page.page_dimensions)
        ):
            raise ValueError(
                "induced page-map source, target, and page axes must agree"
            )
        for level, blocks in enumerate(self.maps):
            if len(blocks) != len(self.source_page.page_dimensions[level]) or len(
                blocks
            ) != len(self.target_page.page_dimensions[level]):
                raise ValueError("induced page maps must cover every chain degree")
            for degree, matrix in enumerate(blocks):
                rows = self.target_page.page_dimensions[level][degree]
                columns = self.source_page.page_dimensions[level][degree]
                if len(matrix) != rows or any(len(row) != columns for row in matrix):
                    raise ValueError("induced page matrix axes are inconsistent")
        return self


class FilteredChainMapCompositionRequest(StrictModel):
    """Two composable filtered chain maps, in application order."""

    first: FilteredChainMapResult
    second: FilteredChainMapResult


class FilteredChainMapPageZeroResult(StrictModel):
    """The exact map induced by a filtered chain map on the E^0 page.

    ``maps[p][n]`` is written in the quotient bases chosen by the source and
    target associated-graded values. Its source and target remain the original
    filtered chain complexes so the map composes with their page results.
    """

    source: ChainComplexValue
    target: ChainComplexValue
    source_filtration: tuple[FiltrationLevel, ...]
    target_filtration: tuple[FiltrationLevel, ...]
    source_dimensions: tuple[tuple[int, ...], ...]
    target_dimensions: tuple[tuple[int, ...], ...]
    maps: tuple[tuple[tuple[tuple[ChainCoefficient, ...], ...], ...], ...]

    @model_validator(mode="after")
    def require_map_axes(self) -> Self:
        if len(self.source_dimensions) != len(self.source_filtration):
            raise ValueError("source E0 dimensions must cover every filtration level")
        if len(self.target_dimensions) != len(self.target_filtration):
            raise ValueError("target E0 dimensions must cover every filtration level")
        if len(self.maps) != len(self.source_filtration):
            raise ValueError("E0 maps must cover every source filtration level")
        degree_count = len(self.source.basis_sizes)
        if (
            len(self.target.basis_sizes) != degree_count
            or self.source.degree_min != self.target.degree_min
            or self.source.degree_max != self.target.degree_max
            or self.source.coefficient_ring != self.target.coefficient_ring
            or self.source.prime != self.target.prime
        ):
            raise ValueError("source and target degree windows must agree")
        for dimensions, complex_value in (
            (self.source_dimensions, self.source),
            (self.target_dimensions, self.target),
        ):
            if len(dimensions) != len(self.source_filtration) or any(
                len(level) != degree_count for level in dimensions
            ):
                raise ValueError("E0 dimensions must cover every degree and level")
            if any(
                sum(level[degree] for level in dimensions)
                != complex_value.basis_sizes[degree]
                for degree in range(degree_count)
            ):
                raise ValueError("E0 dimensions must sum to each chain rank")
        for level, blocks in enumerate(self.maps):
            if len(blocks) != degree_count:
                raise ValueError("E0 maps must cover every chain degree")
            for degree, matrix in enumerate(blocks):
                if len(matrix) != self.target_dimensions[level][degree] or any(
                    len(row) != self.source_dimensions[level][degree] for row in matrix
                ):
                    raise ValueError(
                        "E0 map matrix axes do not match its graded spaces"
                    )
        return self


class SpectralPagesRequest(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)
    through_page: int = Field(ge=0, le=MAX_SPECTRAL_PAGE, default=MAX_SPECTRAL_PAGE)


class SpectralPagesResult(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...]
    pages: tuple[SpectralPageResult, ...]
    stabilized_page: int | None = None
    status: str


class SpectralAbutmentRequest(StrictModel):
    complex: ChainComplexValue
    filtration: tuple[FiltrationLevel, ...] = Field(min_length=1)


class SpectralAbutmentResult(StrictModel):
    """Stable spectral page and its exact comparison with ``Gr H(C)``.

    ``comparisons[p][n].matrix`` is the canonical map from the returned
    stable-page basis to a chosen quotient basis of ``F_p H_n/F_(p-1) H_n``.
    The matrix is square and invertible; it does not choose a splitting of
    filtered homology.
    """

    complex: ChainComplexValue
    page: SpectralPageResult
    status: Literal["STABILIZED"]
    homology: tuple[FilteredHomologyDegree, ...]
    comparisons: tuple[tuple[SpectralAbutmentComparison, ...], ...]

    @model_validator(mode="after")
    def require_comparison_axes(self) -> Self:
        if len(self.homology) != len(self.complex.basis_sizes):
            raise ValueError("abutment homology must cover every chain degree")
        if len(self.comparisons) != len(self.page.page_dimensions):
            raise ValueError("abutment comparisons must cover every filtration level")
        for level, rows in enumerate(self.comparisons):
            if len(rows) != len(self.complex.basis_sizes):
                raise ValueError("abutment comparisons must cover every degree")
            for index, comparison in enumerate(rows):
                homology_dimension = len(self.homology[index].homology_basis)
                if (
                    comparison.filtration_level != level
                    or comparison.degree != self.complex.degree_min + index
                    or comparison.page_dimension
                    != self.page.page_dimensions[level][index]
                    or comparison.homology_graded_dimension
                    != len(comparison.homology_graded_basis_coordinates)
                    or any(
                        len(vector) != homology_dimension
                        for vector in comparison.homology_graded_basis_coordinates
                    )
                    or len(comparison.matrix) != comparison.homology_graded_dimension
                    or any(
                        len(row) != comparison.page_dimension
                        for row in comparison.matrix
                    )
                ):
                    raise ValueError("abutment comparison axes are inconsistent")
        return self


class SpectralAbutmentComparison(StrictModel):
    filtration_level: int = Field(ge=0)
    degree: int
    page_dimension: int = Field(ge=0)
    homology_graded_dimension: int = Field(ge=0)
    homology_graded_basis_coordinates: tuple[Vector, ...]
    matrix: tuple[Vector, ...]


def _admit_homology_filtration(
    complex_value: ChainComplexValue, filtration: tuple[FiltrationLevel, ...]
) -> None:
    """Admit the complete image-subspace and witness envelope before elimination."""
    level_count = len(filtration)
    if complex_value.coefficient_ring is not CoefficientRing.PRIME_FIELD:
        raise OperationDomainValidationError(
            location=("complex", "coefficient_ring"),
            code="filtered_homology.field_required",
            message="filtered homology currently supports bounded GF(p) coefficients",
        )
    if complex_value.prime is None or complex_value.prime > MAX_FILTERED_HOMOLOGY_PRIME:
        raise OperationResourceAdmissionError(
            location=("complex", "prime"),
            code="filtered_homology.prime_bound",
            message=(
                "filtered homology admits prime fields with p at most "
                f"{MAX_FILTERED_HOMOLOGY_PRIME}"
            ),
        )
    sizes = complex_value.basis_sizes
    result_cells = sum(
        2 * size * size + size * (sizes[index + 1] if index + 1 < len(sizes) else 0)
        for index, size in enumerate(sizes)
    )
    per_level_cells = sum(
        2 * size * size + size * (sizes[index + 1] if index + 1 < len(sizes) else 0)
        for index, size in enumerate(sizes)
    )
    result_cells += level_count * per_level_cells
    if result_cells > MAX_FILTERED_HOMOLOGY_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("complex", "basis_sizes"),
            code="filtered_homology.result_bound",
            message=(
                f"the homology bases and filtration witnesses need {result_cells} "
                "scalar cells, above the admitted result envelope of "
                f"{MAX_FILTERED_HOMOLOGY_RESULT_CELLS}"
            ),
        )
    input_cells = sum(
        len(vector)
        for level in filtration
        for subspace in level.subspaces
        for vector in subspace.vectors
    ) + sum(
        len(row) for matrix in complex_value.differential_matrices for row in matrix
    )
    input_vectors = sum(
        len(subspace.vectors) for level in filtration for subspace in level.subspaces
    )
    output_chars = 17 * (input_cells + result_cells) + 4 * input_vectors + 100_000
    if output_chars > MAX_FILTERED_HOMOLOGY_RESULT_CHARS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="filtered_homology.result_bytes_bound",
            message=(
                f"the retained source and homology witnesses need an estimated "
                f"{output_chars} characters, above the "
                f"{MAX_FILTERED_HOMOLOGY_RESULT_CHARS}-character envelope"
            ),
        )
    work = sum(
        (level_count + 2) * (size + 1) ** 4
        + level_count * MAX_FILTER_VECTORS_PER_GROUP * (size + 1) ** 3
        + level_count
        * (size + 1) ** 2
        * ((sizes[index + 1] if index + 1 < len(sizes) else 0) + 1)
        for index, size in enumerate(sizes)
    )
    if work > MAX_FILTERED_HOMOLOGY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex", "basis_sizes"),
            code="filtered_homology.work_bound",
            message=(
                f"the homology and filtration elimination estimate {work} "
                "exceeds the admitted work envelope of "
                f"{MAX_FILTERED_HOMOLOGY_WORK}"
            ),
        )


def _filtered_cycle_basis(
    filtration_basis: list[list[Any]],
    differential: list[list[Any]],
    prime: int,
) -> list[list[Any]]:
    """Intersect one admitted filtration space with the chain-cycle kernel."""
    image_rows = [_mat_vec(differential, vector, prime) for vector in filtration_basis]
    coefficients = _nullspace(_transpose(image_rows), len(filtration_basis), prime)
    cycles: list[list[Any]] = []
    for weights in coefficients:
        width = len(filtration_basis[0]) if filtration_basis else 0
        vector = [_parse_entry(0, prime) for _ in range(width)]
        for weight, basis_vector in zip(weights, filtration_basis, strict=True):
            vector = [
                (value + weight * coefficient) % prime
                for value, coefficient in zip(vector, basis_vector, strict=True)
            ]
        cycles.append(vector)
    return _row_basis(cycles, prime)


def _linear_combination(
    basis: list[list[Any]], coordinates: list[Any], ambient_dimension: int, prime: int
) -> list[Any]:
    result = [_parse_entry(0, prime) for _ in range(ambient_dimension)]
    for coefficient, vector in zip(coordinates, basis, strict=True):
        result = [
            (value + coefficient * entry) % prime
            for value, entry in zip(result, vector, strict=True)
        ]
    return result


def _serialize_vectors(vectors: list[list[Any]], prime: int) -> tuple[Vector, ...]:
    return tuple(
        tuple(_serialize_scalar(value, prime) for value in vector) for vector in vectors
    )


def _homology_bases(
    sizes: tuple[int, ...], differentials: list[list[list[Any]]], prime: int
) -> tuple[list[list[list[Any]]], list[list[list[Any]]], list[list[list[Any]]]]:
    cycles_by_degree: list[list[list[Any]]] = []
    boundaries_by_degree: list[list[list[Any]]] = []
    homology_by_degree: list[list[list[Any]]] = []
    for degree, dimension in enumerate(sizes):
        outgoing = differentials[degree - 1] if degree > 0 else []
        cycles = _nullspace(outgoing, dimension, prime)
        if any(any(_mat_vec(outgoing, vector, prime)) for vector in cycles):
            raise OperationDomainValidationError(
                location=("complex", "differential_matrices"),
                code="filtered_homology.cycle_replay_failed",
                message="a returned homology-basis vector is not a source cycle",
            )
        incoming = differentials[degree] if degree < len(differentials) else []
        boundaries = _row_basis(_transpose(incoming), prime)
        if any(not _in_span(cycles, vector, prime) for vector in boundaries):
            raise OperationDomainValidationError(
                location=("complex", "differential_matrices"),
                code="filtered_homology.boundary_not_cycle",
                message="the incoming boundary space is not contained in cycles",
            )
        homology = _quotient_extension(boundaries, cycles, prime)
        if len(cycles) != len(boundaries) + len(homology):
            raise OperationDomainValidationError(
                location=("complex", "differential_matrices"),
                code="filtered_homology.rank_identity_failed",
                message="cycle, boundary, and quotient bases do not form a complete homology basis",
            )
        cycles_by_degree.append(cycles)
        boundaries_by_degree.append(boundaries)
        homology_by_degree.append(homology)
    return cycles_by_degree, boundaries_by_degree, homology_by_degree


def _homology_image_subspace(
    filtration_basis: list[list[Any]],
    outgoing: list[list[Any]],
    incoming: list[list[Any]],
    boundary_basis: list[list[Any]],
    homology_basis: list[list[Any]],
    degree_dimension: int,
    level_index: int,
    degree_index: int,
    prime: int,
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    filtered_cycles = _filtered_cycle_basis(filtration_basis, outgoing, prime)
    if any(
        not _in_span(filtration_basis, cycle, prime)
        or any(_mat_vec(outgoing, cycle, prime))
        for cycle in filtered_cycles
    ):
        raise OperationDomainValidationError(
            location=("filtration", level_index, degree_index),
            code="filtered_homology.filtered_cycle_replay_failed",
            message="a returned filtration representative is not a source-bound cycle",
        )
    full_cycle_basis = [*boundary_basis, *homology_basis]
    selected_coordinates: list[list[Any]] = []
    selected_representatives: list[list[Any]] = []
    for cycle in filtered_cycles:
        coordinates = _coordinates(full_cycle_basis, cycle, prime)
        class_coordinates = coordinates[len(boundary_basis) :]
        if not _in_span(selected_coordinates, class_coordinates, prime):
            selected_coordinates.append(class_coordinates)
            selected_representatives.append(cycle)
    for cycle in filtered_cycles:
        coordinates = _coordinates(full_cycle_basis, cycle, prime)
        if not _in_span(
            selected_coordinates, coordinates[len(boundary_basis) :], prime
        ):
            raise OperationDomainValidationError(
                location=("filtration", level_index, degree_index),
                code="filtered_homology.image_incomplete",
                message="returned representatives do not span the filtration image in homology",
            )

    preimages: list[list[Any]] = []
    for coordinates, representative in zip(
        selected_coordinates, selected_representatives, strict=True
    ):
        homology_cycle = _linear_combination(
            homology_basis, coordinates, degree_dimension, prime
        )
        difference = [
            (value - homology_value) % prime
            for value, homology_value in zip(
                representative, homology_cycle, strict=True
            )
        ]
        preimage = (
            []
            if not incoming and not any(difference)
            else _solve(incoming, difference, prime)
        )
        boundary_matches = (not incoming and not any(difference)) or (
            preimage is not None and _mat_vec(incoming, preimage, prime) == difference
        )
        if not boundary_matches:
            raise OperationDomainValidationError(
                location=("filtration", level_index, degree_index),
                code="filtered_homology.boundary_witness_invalid",
                message="the returned chain does not replay its homology-class boundary relation",
            )
        assert preimage is not None
        preimages.append(preimage)
    return selected_coordinates, selected_representatives, preimages


def _homology_image_levels(
    request: FilteredChainComplexRequest,
    sizes: tuple[int, ...],
    filtration_bases: list[list[list[list[Any]]]],
    differentials: list[list[list[Any]]],
    boundaries: list[list[list[Any]]],
    homology: list[list[list[Any]]],
    prime: int,
) -> list[list[tuple[list[Any], list[Any], list[Any]]]]:
    image_data: list[list[tuple[list[Any], list[Any], list[Any]]]] = []
    image_coordinates: list[list[list[list[Any]]]] = []
    for level_index, _level in enumerate(request.filtration):
        level_data = []
        level_coordinates = []
        for degree, dimension in enumerate(sizes):
            outgoing = differentials[degree - 1] if degree > 0 else []
            incoming = differentials[degree] if degree < len(differentials) else []
            data = _homology_image_subspace(
                filtration_bases[level_index][degree],
                outgoing,
                incoming,
                boundaries[degree],
                homology[degree],
                dimension,
                level_index,
                degree,
                prime,
            )
            level_data.append(data)
            level_coordinates.append(data[0])
        image_data.append(level_data)
        image_coordinates.append(level_coordinates)
    for level_index in range(1, len(request.filtration)):
        for degree in range(len(sizes)):
            if any(
                not _in_span(image_coordinates[level_index][degree], vector, prime)
                for vector in image_coordinates[level_index - 1][degree]
            ):
                raise OperationDomainValidationError(
                    location=("filtration", level_index),
                    code="filtered_homology.image_not_nested",
                    message="the induced homology images must follow source filtration inclusions",
                )
    return image_data


def filtered_homology_filtration(
    request: FilteredChainComplexRequest,
) -> FilteredHomologyResult:
    """Return exact images of filtration levels in source-bound homology."""
    if not isinstance(request, FilteredChainComplexRequest):
        raise OperationDomainValidationError(
            location=(),
            code="filtered_homology.request_type_invalid",
            message="filtered homology requires a canonical filtered-complex request",
        )
    _admit_homology_filtration(request.complex, request.filtration)
    admitted = _admit_filtered_semantics(request.complex, request.filtration)
    complex_value = request.complex
    prime = complex_value.prime
    assert prime is not None
    sizes = complex_value.basis_sizes
    differentials = admitted.differentials
    cycles, boundaries, homology = _homology_bases(sizes, differentials, prime)
    image_data = _homology_image_levels(
        request,
        sizes,
        admitted.bases,
        differentials,
        boundaries,
        homology,
        prime,
    )

    groups = tuple(
        FilteredHomologyDegree(
            degree=complex_value.degree_min + degree,
            cycle_basis=_serialize_vectors(cycles[degree], prime),
            boundary_basis=_serialize_vectors(boundaries[degree], prime),
            homology_basis=_serialize_vectors(homology[degree], prime),
        )
        for degree in range(len(sizes))
    )
    levels = tuple(
        FilteredHomologyLevel(
            subspaces=tuple(
                HomologyFiltrationImage(
                    basis_coordinates=_serialize_vectors(data[0], prime),
                    cycle_representatives=_serialize_vectors(data[1], prime),
                    boundary_preimages=_serialize_vectors(data[2], prime),
                )
                for data in level
            )
        )
        for level in image_data
    )
    return FilteredHomologyResult(
        complex=complex_value,
        filtration=request.filtration,
        homology=groups,
        image_filtration=levels,
    )


def _check_filtered_chain_map_axes(request: FilteredChainMapRequest) -> None:
    if (
        request.source.coefficient_ring != request.target.coefficient_ring
        or request.source.prime != request.target.prime
        or request.source.degree_min != request.target.degree_min
        or request.source.degree_max != request.target.degree_max
    ):
        raise OperationDomainValidationError(
            location=("target",),
            code="filtered_chain_map.parent_mismatch",
            message="source and target complexes must share coefficient and degree parents",
        )
    if len(request.source_filtration) != len(request.target_filtration) or len(
        request.maps
    ) != len(request.source.basis_sizes):
        raise OperationDomainValidationError(
            location=("maps",),
            code="filtered_chain_map.axis_mismatch",
            message="map and filtration degree axes must agree",
        )
    for degree, matrix in enumerate(request.maps):
        if len(matrix) != request.target.basis_sizes[degree] or any(
            len(row) != request.source.basis_sizes[degree] for row in matrix
        ):
            raise OperationDomainValidationError(
                location=("maps", degree),
                code="filtered_chain_map.shape_invalid",
                message="each map must have target-by-source chain axes",
            )


def filtered_map(request: FilteredChainMapRequest) -> FilteredChainMapResult:
    _check_filtered_chain_map_axes(request)
    source_admission = _admit_filtered_semantics(
        request.source, request.source_filtration
    )
    target_admission = _admit_filtered_semantics(
        request.target, request.target_filtration
    )
    parsed, chain_ok, preserving = _filtered_map_status_admitted(
        request, source_admission, target_admission
    )
    return _filtered_map_result(request, parsed, chain_ok, preserving)


def _filtered_map_admitted(
    request: FilteredChainMapRequest,
    source_admission: Any,
    target_admission: Any,
) -> FilteredChainMapResult:
    parsed, chain_ok, preserving = _filtered_map_status_admitted(
        request, source_admission, target_admission
    )
    return _filtered_map_result(request, parsed, chain_ok, preserving)


def _filtered_map_status_admitted(
    request: FilteredChainMapRequest,
    source_admission: Any,
    target_admission: Any,
    parsed: list[list[list[int | Fraction]]] | None = None,
) -> tuple[list[list[list[int | Fraction]]], bool, bool]:
    p = request.source.prime
    if parsed is None:
        parsed = []
        for degree, matrix in enumerate(request.maps):
            try:
                parsed.append([[_parse_entry(v, p) for v in row] for row in matrix])
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise OperationDomainValidationError(
                    location=("maps", degree),
                    code="filtered_chain_map.entry_invalid",
                    message="map entries must use the retained canonical coefficient grammar",
                ) from exc
    # f d = d f, with matrix convention d rows lower x upper
    chain_ok = True
    for degree in range(len(parsed) - 1):
        output_width = request.source.basis_sizes[degree + 1]
        left = _mul(
            parsed[degree],
            source_admission.differentials[degree],
            p,
            output_width=output_width,
        )
        right = _mul(
            target_admission.differentials[degree],
            parsed[degree + 1],
            p,
            output_width=output_width,
        )
        chain_ok = chain_ok and left == right
    preserving = True
    for source_level, target_level in zip(
        source_admission.bases, target_admission.bases, strict=True
    ):
        for degree, (source_basis, target_basis) in enumerate(
            zip(source_level, target_level, strict=True)
        ):
            for vector in source_basis:
                image = _mat_vec(parsed[degree], vector, p)
                if not _in_span(target_basis, image, p):
                    preserving = False
    return parsed, chain_ok, preserving


def _filtered_map_result(
    request: FilteredChainMapRequest,
    parsed: list[list[list[int | Fraction]]],
    chain_ok: bool,
    preserving: bool,
) -> FilteredChainMapResult:
    p = request.source.prime
    canonical_maps = tuple(
        tuple(tuple(_serialize_scalar(value, p) for value in row) for row in matrix)
        for matrix in parsed
    )
    return FilteredChainMapResult(
        source=request.source,
        target=request.target,
        source_filtration=request.source_filtration,
        target_filtration=request.target_filtration,
        maps=canonical_maps,
        filtration_preserving=preserving,
        chain_map=chain_ok,
    )


def _admit_e0_map_request(request: FilteredChainMapRequest) -> tuple[Any, Any, Any]:
    _check_filtered_chain_map_axes(request)
    source_admission = _admit_filtered_semantics(
        request.source, request.source_filtration
    )
    target_admission = _admit_filtered_semantics(
        request.target, request.target_filtration
    )
    map_cells = sum(
        (
            len(source_admission.bases[level][degree])
            - (len(source_admission.bases[level - 1][degree]) if level else 0)
        )
        * (
            len(target_admission.bases[level][degree])
            - (len(target_admission.bases[level - 1][degree]) if level else 0)
        )
        for level in range(len(request.source_filtration))
        for degree in range(len(request.source.basis_sizes))
    )
    if map_cells > MAX_FILTERED_HOMOLOGY_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.e0_output_cells_exceeded",
            message=(
                "the E0 map envelope exceeds the admitted "
                f"{MAX_FILTERED_HOMOLOGY_RESULT_CELLS} cells"
            ),
        )
    input_chars = 0
    scalar_count = 0
    max_scalar_chars = 1
    for value in chain(
        (entry for matrix in request.maps for row in matrix for entry in row),
        *(
            chain(
                (
                    entry
                    for matrix in complex_value.differential_matrices
                    for row in matrix
                    for entry in row
                ),
                (
                    entry
                    for level in filtration
                    for subspace in level.subspaces
                    for vector in subspace.vectors
                    for entry in vector
                ),
            )
            for complex_value, filtration in (
                (request.source, request.source_filtration),
                (request.target, request.target_filtration),
            )
        ),
    ):
        value_chars = len(str(value))
        input_chars += value_chars
        scalar_count += 1
        max_scalar_chars = max(max_scalar_chars, value_chars)
    # Determinant expansion bounds the exact numerator and denominator growth
    # of the at-most-32-dimensional coordinate solves used below.
    scalar_chars_bound = 96 * max_scalar_chars + 512
    output_chars = (
        input_chars
        + scalar_count * 16
        + map_cells * scalar_chars_bound
        + len(request.source_filtration) * len(request.source.basis_sizes) * 128
    )
    if output_chars > MAX_FILTERED_HOMOLOGY_RESULT_CHARS:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.e0_output_chars_exceeded",
            message=(
                "the conservative exact E0 map character envelope exceeds "
                f"{MAX_FILTERED_HOMOLOGY_RESULT_CHARS} characters"
            ),
        )
    map_value = _filtered_map_admitted(request, source_admission, target_admission)
    if not map_value.chain_map:
        raise OperationDomainValidationError(
            location=("maps",),
            code="filtered_chain_map.not_chain_map",
            message="an E0 page map requires a chain map",
        )
    if not map_value.filtration_preserving:
        raise OperationDomainValidationError(
            location=("maps",),
            code="filtered_chain_map.not_filtration_preserving",
            message="an E0 page map requires filtration preservation",
        )
    return source_admission, target_admission, map_value


def filtered_chain_map_page_zero(
    request: FilteredChainMapRequest,
) -> FilteredChainMapPageZeroResult:
    """Induce the degreewise E0 map of an exact filtered chain map."""
    source_admission, target_admission, map_value = _admit_e0_map_request(request)

    source_graded = _associated_graded_admitted(
        request.source, request.source_filtration, source_admission
    )
    target_graded = _associated_graded_admitted(
        request.target, request.target_filtration, target_admission
    )
    prime = request.source.prime
    map_matrices = [
        [[_parse_entry(entry, prime) for entry in row] for row in degree]
        for degree in map_value.maps
    ]
    source_blocks: list[tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]] = []
    lower_target: list[list[Vector]] = [[] for _ in request.target.basis_sizes]
    for level in range(len(request.source_filtration)):
        level_blocks = []
        for degree in range(len(request.source.basis_sizes)):
            source_representatives = source_graded.quotient_representatives[level][
                degree
            ]
            target_representatives = target_graded.quotient_representatives[level][
                degree
            ]
            target_basis = [
                [_parse_entry(entry, prime) for entry in vector]
                for vector in (*lower_target[degree], *target_representatives)
            ]
            columns = []
            for vector in source_representatives:
                image = _mat_vec(
                    map_matrices[degree],
                    [_parse_entry(entry, prime) for entry in vector],
                    prime,
                )
                try:
                    coordinates = _coordinates(target_basis, image, prime)
                except ValueError as exc:
                    raise _fail(
                        ("maps", level, degree),
                        "filtered_chain_map.e0_image_outside_target",
                        "a filtration-preserving map must send each E0 representative into the matching target filtration level",
                    ) from exc
                columns.append(coordinates[len(lower_target[degree]) :])
            rows = len(target_representatives)
            matrix = [
                [columns[column][row] for column in range(len(columns))]
                for row in range(rows)
            ]
            level_blocks.append(
                tuple(
                    tuple(_serialize_scalar(value, prime) for value in row)
                    for row in matrix
                )
            )
        source_blocks.append(tuple(level_blocks))
        for degree, representatives in enumerate(
            target_graded.quotient_representatives[level]
        ):
            lower_target[degree].extend(
                tuple(_parse_entry(entry, prime) for entry in vector)
                for vector in representatives
            )

    # The induced degreewise maps must commute with the associated-graded
    # differentials. This replay is bounded by the already admitted axes.
    for level, blocks in enumerate(source_blocks):
        for degree in range(len(request.source.basis_sizes) - 1):
            left = _mul(
                [
                    [_parse_entry(entry, prime) for entry in row]
                    for row in target_graded.graded_differentials[level][degree]
                ],
                [
                    [_parse_entry(entry, prime) for entry in row]
                    for row in blocks[degree + 1]
                ],
                prime,
                output_width=source_graded.graded_dimensions[level][degree + 1],
            )
            right = _mul(
                [
                    [_parse_entry(entry, prime) for entry in row]
                    for row in blocks[degree]
                ],
                [
                    [_parse_entry(entry, prime) for entry in row]
                    for row in source_graded.graded_differentials[level][degree]
                ],
                prime,
                output_width=source_graded.graded_dimensions[level][degree + 1],
            )
            if left != right:
                raise _fail(
                    ("maps", level, degree),
                    "filtered_chain_map.e0_square_failed",
                    "the induced E0 maps must commute with the associated-graded differential",
                )

    return FilteredChainMapPageZeroResult(
        source=request.source,
        target=request.target,
        source_filtration=request.source_filtration,
        target_filtration=request.target_filtration,
        source_dimensions=source_graded.graded_dimensions,
        target_dimensions=target_graded.graded_dimensions,
        maps=tuple(source_blocks),
    )


def _mul(left: Any, right: Any, prime: int | None, *, output_width: int) -> Any:
    if not left:
        return []
    if not right:
        return [[0] * output_width for _ in left]
    cols = list(zip(*right, strict=False))
    result = []
    for row in left:
        output = []
        for col in cols:
            value = sum(a * b for a, b in zip(row, col, strict=False))
            output.append(value % prime if prime is not None else value)
        result.append(output)
    return result


def filtered_chain_map_page(
    request: FilteredChainMapPageRequest,
) -> FilteredChainMapPageResult:
    """Induce the exact map on one bounded E^r page.

    The operation reapplies chain-map and filtration admission, transports the
    source page representatives in ambient coordinates, reduces them modulo
    the target page denominators, and checks naturality against every d^r.
    """
    authored = request.map
    chain_map_request = FilteredChainMapRequest(
        source=authored.source,
        source_filtration=authored.source_filtration,
        target=authored.target,
        target_filtration=authored.target_filtration,
        maps=authored.maps,
    )
    source_admission, target_admission, map_value = _admit_e0_map_request(
        chain_map_request
    )

    source, target = authored.source, authored.target
    levels = len(authored.source_filtration)
    degree_count = len(source.basis_sizes)
    prime = source.prime
    # Bound repeated exact subspace reductions and the dense page-map output
    # before page representatives or denominator bases are expanded.
    cubic = sum(
        max(1, rank) ** 3 for rank in (*source.basis_sizes, *target.basis_sizes)
    )
    coordinate_solves = sum(
        max(1, rank) ** 4 for rank in (*source.basis_sizes, *target.basis_sizes)
    )
    work_bound = levels * (cubic * (3 * request.page + 3) + coordinate_solves)
    if work_bound > MAX_FILTERED_HOMOLOGY_WORK:
        raise OperationResourceAdmissionError(
            location=("page",),
            code="filtered_chain_map.page_work_exceeded",
            message="the conservative representative-transport work bound exceeds the admitted page-map work",
        )
    map_cells = levels * sum(
        source.basis_sizes[index] * target.basis_sizes[index]
        for index in range(degree_count)
    )
    page_cells = (
        levels
        * 3
        * sum(
            source.basis_sizes[index] ** 2 + target.basis_sizes[index] ** 2
            for index in range(degree_count)
        )
    )
    input_scalars = [
        value
        for complex_value, filtration in (
            (source, authored.source_filtration),
            (target, authored.target_filtration),
        )
        for matrix in complex_value.differential_matrices
        for row in matrix
        for value in row
    ]
    input_scalars.extend(
        value
        for filtration in (authored.source_filtration, authored.target_filtration)
        for level in filtration
        for subspace in level.subspaces
        for vector in subspace.vectors
        for value in vector
    )
    input_scalars.extend(
        value for matrix in map_value.maps for row in matrix for value in row
    )
    max_scalar_chars = max((len(str(value)) for value in input_scalars), default=1)
    scalar_chars_bound = 96 * max_scalar_chars + 512
    output_bound = (map_cells + page_cells) * scalar_chars_bound + 4096
    if output_bound > MAX_FILTERED_HOMOLOGY_RESULT_CHARS:
        raise OperationResourceAdmissionError(
            location=("page",),
            code="filtered_chain_map.page_output_exceeded",
            message="the conservative page-map output bound exceeds the admitted result size",
        )

    admit_spectral_page(request.page)
    if request.page == 0:
        source_page = _spectral_zero_page(
            _associated_graded_admitted(
                source, authored.source_filtration, source_admission
            ),
            request.page,
        )
        target_page = _spectral_zero_page(
            _associated_graded_admitted(
                target, authored.target_filtration, target_admission
            ),
            request.page,
        )
    else:
        source_page = _spectral_bidegree_page(
            source,
            authored.source_filtration,
            source_admission.bases,
            source_admission.differentials,
            request.page,
        )
        target_page = _spectral_bidegree_page(
            target,
            authored.target_filtration,
            target_admission.bases,
            target_admission.differentials,
            request.page,
        )
    if request.page == 0:
        target_denominators: list[list[list[list[int | Fraction]]]] = [
            [[] for _ in range(degree_count)] for _ in range(levels)
        ]
        lower: list[list[list[int | Fraction]]] = [[] for _ in range(degree_count)]
        for level in range(levels):
            target_denominators[level] = [list(rows) for rows in lower]
            for degree, reps in enumerate(target_page.page_representatives[level]):
                lower[degree].extend(
                    [_parse_entry(value, prime) for value in vector] for vector in reps
                )
    else:
        target_denominators = [[[] for _ in range(degree_count)] for _ in range(levels)]
        for level in range(levels):
            for degree in range(degree_count):
                target_denominators[level][degree] = _denominator_rows(
                    target_admission.bases,
                    target_admission.differentials,
                    target.basis_sizes,
                    level,
                    degree,
                    request.page,
                    prime,
                )

    parsed_maps = [
        [[_parse_entry(value, prime) for value in row] for row in matrix]
        for matrix in map_value.maps
    ]
    map_blocks: list[tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]] = []
    for level in range(levels):
        degree_blocks = []
        for degree in range(degree_count):
            denominator = target_denominators[level][degree]
            target_reps = [
                [_parse_entry(value, prime) for value in vector]
                for vector in target_page.page_representatives[level][degree]
            ]
            target_basis = [*denominator, *target_reps]
            columns = []
            for vector in source_page.page_representatives[level][degree]:
                image = _mat_vec(
                    parsed_maps[degree],
                    [_parse_entry(value, prime) for value in vector],
                    prime,
                )
                try:
                    coordinates = _coordinates(target_basis, image, prime)
                except ValueError as exc:
                    raise _fail(
                        ("map", "maps", level, degree),
                        "filtered_chain_map.page_image_outside_target",
                        "a source page representative must map into the target page cycles",
                    ) from exc
                columns.append(coordinates[len(denominator) :])
            rows = target_page.page_dimensions[level][degree]
            block = [
                [columns[column][row] for column in range(len(columns))]
                for row in range(rows)
            ]
            degree_blocks.append(
                tuple(
                    tuple(_serialize_scalar(value, prime) for value in row)
                    for row in block
                )
            )
        map_blocks.append(tuple(degree_blocks))

    maps = tuple(map_blocks)
    # Check target d^r after f equals f after source d^r. Dimensions on each
    # differential retain empty rows/columns even when its dense entries do not.
    target_records = {
        (entry.source_level, entry.source_degree): entry
        for entry in target_page.differentials
    }
    for source_record in source_page.differentials:
        key = (source_record.source_level, source_record.source_degree)
        target_record = target_records[key]
        level = source_record.source_level
        degree = source_record.source_degree - source.degree_min
        target_level = source_record.target_level
        target_degree = degree - 1
        f_source = [
            [_parse_entry(value, prime) for value in row] for row in maps[level][degree]
        ]
        f_target = [
            [_parse_entry(value, prime) for value in row]
            for row in maps[target_level][target_degree]
        ]
        source_d = [
            [_parse_entry(value, prime) for value in row]
            for row in source_record.entries
        ]
        target_d = [
            [_parse_entry(value, prime) for value in row]
            for row in target_record.entries
        ]
        left = _rectangular_product(
            target_d,
            target_record.rows,
            target_record.columns,
            f_source,
            source_page.page_dimensions[level][degree],
            prime,
        )
        right = _rectangular_product(
            f_target,
            target_page.page_dimensions[target_level][target_degree],
            source_page.page_dimensions[target_level][target_degree],
            source_d,
            source_page.page_dimensions[level][degree],
            prime,
        )
        if left != right:
            raise _fail(
                ("map", "maps", level, degree),
                "filtered_chain_map.page_square_failed",
                "the induced page map must commute with the page differential",
            )
    return FilteredChainMapPageResult(
        map=map_value,
        source_page=source_page,
        target_page=target_page,
        maps=maps,
    )


def _rectangular_product(
    left: list[list[int | Fraction]],
    left_rows: int,
    inner: int,
    right: list[list[int | Fraction]],
    right_columns: int,
    prime: int | None,
) -> list[list[int | Fraction]]:
    """Multiply matrices while retaining explicitly admitted empty axes."""
    if left_rows and (len(left) != left_rows or any(len(row) != inner for row in left)):
        raise ValueError("left page matrix has inconsistent axes")
    if inner and (
        len(right) != inner or any(len(row) != right_columns for row in right)
    ):
        raise ValueError("right page matrix has inconsistent axes")
    if not left_rows:
        return []
    if not inner:
        return [[0 for _ in range(right_columns)] for _ in range(left_rows)]
    columns = list(zip(*right, strict=True)) if right_columns else []
    return [
        [
            (sum(a * b for a, b in zip(row, column, strict=True)) % prime)
            if prime is not None
            else sum(a * b for a, b in zip(row, column, strict=True))
            for column in columns
        ]
        for row in left
    ]


def _coefficient_size(value: int | Fraction) -> tuple[int, int]:
    """Return decimal numerator and denominator digit counts without expansion."""
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    return len(str(abs(fraction.numerator))), len(str(fraction.denominator))


def _parse_bounded_map(
    request: FilteredChainMapRequest, label: str, prime: int | None
) -> list[list[list[int | Fraction]]]:
    if len(request.maps) != len(request.source.basis_sizes):
        raise _fail(
            (label, "maps"),
            "filtered_chain_map.shape_invalid",
            "each map must carry one matrix per chain degree",
        )
    cells = 0
    chars = 0
    parsed: list[list[list[int | Fraction]]] = []
    for degree, matrix in enumerate(request.maps):
        rows = request.target.basis_sizes[degree]
        columns = request.source.basis_sizes[degree]
        if len(matrix) != rows or any(len(row) != columns for row in matrix):
            raise _fail(
                (label, "maps", degree),
                "filtered_chain_map.shape_invalid",
                "each map must have target-by-source chain axes",
            )
        cells += rows * columns
        parsed_matrix = []
        for row in matrix:
            parsed_row = []
            for value in row:
                numerator_digits, denominator_digits = _coefficient_size(value)
                if (
                    numerator_digits > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS
                    or denominator_digits > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=(label, "maps", degree),
                        code="filtered_chain_map.coefficient_exceeded",
                        message="an input map coefficient exceeds the exact "
                        "chain-map coefficient digit limit",
                    )
                chars += numerator_digits + denominator_digits + 1
                parsed_row.append(_parse_entry(value, prime))
            parsed_matrix.append(parsed_row)
        parsed.append(parsed_matrix)
    if cells > MAX_CHAIN_MAP_CELLS or chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise OperationResourceAdmissionError(
            location=(label, "maps"),
            code="filtered_chain_map.input_envelope_exceeded",
            message="the degreewise map exceeds its admitted cell or "
            "coefficient-character envelope",
        )
    return parsed


def _coefficient_sum_bound(terms: list[tuple[int | Fraction, int | Fraction]]) -> int:
    """Bound decimal numerator and denominator sizes of a rational sum."""
    term_sizes = []
    for left, right in terms:
        left_numerator, left_denominator = _coefficient_size(left)
        right_numerator, right_denominator = _coefficient_size(right)
        denominator_is_one = (
            Fraction(left).denominator == 1 and Fraction(right).denominator == 1
        )
        term_sizes.append(
            (
                left_numerator + right_numerator,
                1 if denominator_is_one else left_denominator + right_denominator,
                denominator_is_one,
            )
        )
    denominator_digits = (
        1
        if all(size[2] for size in term_sizes)
        else sum(size[1] for size in term_sizes)
    )
    numerator_digits = max(
        numerator + denominator_digits - term_denominator
        for numerator, term_denominator, _denominator_is_one in term_sizes
    ) + len(str(len(terms)))
    if (
        numerator_digits > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS
        or denominator_digits > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_coefficient_exceeded",
            message="a composed coefficient may exceed the exact chain-map "
            "coefficient digit limit",
        )
    return numerator_digits + denominator_digits + 1


def _composition_preflight(
    first: FilteredChainMapRequest, second: FilteredChainMapRequest
) -> tuple[
    int | None,
    list[list[list[int | Fraction]]],
    list[list[list[int | Fraction]]],
]:
    """Bound map products and coefficient growth before multiplying matrices."""
    if first.target != second.source:
        raise _fail(
            ("second",),
            "filtered_chain_map.composition_middle_mismatch",
            "the target complex of the first map must equal the source complex "
            "of the second map",
        )
    if (
        first.source.coefficient_ring != second.target.coefficient_ring
        or first.source.prime != second.target.prime
    ):
        raise _fail(
            ("second",),
            "filtered_chain_map.composition_coefficient_mismatch",
            "the composite must retain one exact coefficient field",
        )

    prime = first.source.prime
    parsed_first = _parse_bounded_map(first, "first", prime)
    parsed_second = _parse_bounded_map(second, "second", prime)

    output_cells = sum(
        rows * columns
        for rows, columns in zip(
            second.target.basis_sizes, first.source.basis_sizes, strict=True
        )
    )
    work = sum(
        rows * columns * middle
        for rows, columns, middle in zip(
            second.target.basis_sizes,
            first.source.basis_sizes,
            first.target.basis_sizes,
            strict=True,
        )
    )
    if (
        output_cells > MAX_CHAIN_MAP_CELLS
        or work > MAX_FILTERED_CHAIN_MAP_COMPOSITION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_work_exceeded",
            message="the composed map exceeds its admitted cell or exact "
            "multiplication-work envelope",
        )

    output_chars = 0
    for left, right in zip(parsed_second, parsed_first, strict=True):
        for row in left:
            for column in zip(*right, strict=False):
                terms = [
                    (a, b)
                    for a, b in zip(row, column, strict=True)
                    if a != 0 and b != 0
                ]
                if not terms:
                    output_chars += 1
                    continue
                if prime is not None:
                    output_chars += len(str(prime - 1))
                    continue
                output_chars += _coefficient_sum_bound(terms)
    if output_chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_output_exceeded",
            message="the composed map exceeds the aggregate exact output "
            "coefficient-character limit",
        )
    return prime, parsed_first, parsed_second


def filtered_chain_map_compose(
    request: FilteredChainMapCompositionRequest,
) -> FilteredChainMapResult:
    """Compose exact filtration-preserving chain maps in application order."""
    first_request = FilteredChainMapRequest(
        source=request.first.source,
        source_filtration=request.first.source_filtration,
        target=request.first.target,
        target_filtration=request.first.target_filtration,
        maps=request.first.maps,
    )
    second_request = FilteredChainMapRequest(
        source=request.second.source,
        source_filtration=request.second.source_filtration,
        target=request.second.target,
        target_filtration=request.second.target_filtration,
        maps=request.second.maps,
    )
    prime, parsed_first, parsed_second = _composition_preflight(
        first_request, second_request
    )
    _check_filtered_chain_map_axes(first_request)
    _check_filtered_chain_map_axes(second_request)
    first_source = _admit_filtered_semantics(
        first_request.source, first_request.source_filtration
    )
    middle_first = _admit_filtered_semantics(
        first_request.target, first_request.target_filtration
    )
    if first_request.target_filtration == second_request.source_filtration:
        middle_second = middle_first
    else:
        middle_second = _admit_filtered_semantics(
            second_request.source, second_request.source_filtration
        )
        if middle_first.bases != middle_second.bases:
            raise _fail(
                ("second", "source_filtration"),
                "filtered_chain_map.composition_middle_filtration_mismatch",
                "the two middle filtrations must define the same subspaces",
            )
    last_target = _admit_filtered_semantics(
        second_request.target, second_request.target_filtration
    )
    _, first_chain_map, first_preserving = _filtered_map_status_admitted(
        first_request, first_source, middle_first, parsed_first
    )
    _, second_chain_map, second_preserving = _filtered_map_status_admitted(
        second_request, middle_second, last_target, parsed_second
    )
    for label, chain_map, preserving in (
        ("first", first_chain_map, first_preserving),
        ("second", second_chain_map, second_preserving),
    ):
        if not chain_map:
            raise _fail(
                (label, "maps"),
                "filtered_chain_map.composition_input_not_chain_map",
                "both inputs must commute with their chain differentials",
            )
        if not preserving:
            raise _fail(
                (label, "maps"),
                "filtered_chain_map.composition_input_not_filtered",
                "both inputs must preserve their supplied filtrations",
            )
    maps = tuple(
        tuple(
            tuple(_serialize_scalar(value, prime) for value in row)
            for row in _mul(
                parsed_second[degree],
                parsed_first[degree],
                prime,
                output_width=first_request.source.basis_sizes[degree],
            )
        )
        for degree in range(len(parsed_first))
    )
    return FilteredChainMapResult(
        source=first_request.source,
        target=second_request.target,
        source_filtration=first_request.source_filtration,
        target_filtration=second_request.target_filtration,
        maps=maps,
        filtration_preserving=True,
        chain_map=True,
    )


def pages_through(request: SpectralPagesRequest) -> SpectralPagesResult:
    admit_filtered(request.complex, request.filtration)
    pages = []
    stable = None
    for page in range(request.through_page + 1):
        value = spectral_page(request.complex, request.filtration, page)
        pages.append(value)
        if value.page_status.value == "STABILIZED":
            stable = page
            break
    return SpectralPagesResult(
        complex=request.complex,
        filtration=request.filtration,
        pages=tuple(pages),
        stabilized_page=stable,
        status="STABILIZED"
        if stable is not None
        else ("TRUNCATED" if len(pages) == request.through_page + 1 else "ACTIVE"),
    )


def _exact_filtered_homology_coordinates(
    request: SpectralAbutmentRequest,
    admitted: Any,
) -> tuple[
    list[tuple[list[list[Any]], list[list[Any]]]],
    list[list[list[list[Any]]]],
]:
    """Return global homology bases and each filtration image in those bases."""
    sizes = request.complex.basis_sizes
    levels = len(request.filtration)
    prime = request.complex.prime
    differentials = admitted.differentials
    homology_data: list[tuple[list[list[Any]], list[list[Any]]]] = []
    filtration_images: list[list[list[list[Any]]]] = [
        [[] for _ in sizes] for _ in range(levels)
    ]
    for degree, dimension in enumerate(sizes):
        outgoing = differentials[degree - 1] if degree > 0 else []
        incoming = differentials[degree] if degree < len(differentials) else []
        cycles = _nullspace(outgoing, dimension, prime)
        boundaries = _row_basis(_transpose(incoming), prime)
        if any(not _in_span(cycles, vector, prime) for vector in boundaries):
            raise OperationDomainValidationError(
                location=("complex", "differential_matrices"),
                code="spectral_sequence.abutment_boundary_not_cycle",
                message="incoming boundaries must lie in the source cycle space",
            )
        homology_basis = _quotient_extension(boundaries, cycles, prime)
        homology_data.append((boundaries, homology_basis))
        for level in range(levels):
            filtered_basis = admitted.bases[level][degree]
            images = [_mat_vec(outgoing, vector, prime) for vector in filtered_basis]
            coefficient_cycles = _nullspace(
                _transpose(images), len(filtered_basis), prime
            )
            filtered_cycles = []
            for weights in coefficient_cycles:
                vector = [_parse_entry(0, prime) for _ in range(dimension)]
                for weight, basis_vector in zip(weights, filtered_basis, strict=True):
                    vector = [
                        current + weight * value
                        if prime is None
                        else (current + weight * value) % prime
                        for current, value in zip(vector, basis_vector, strict=True)
                    ]
                filtered_cycles.append(vector)
            filtered_cycles = _row_basis(filtered_cycles, prime)
            full_cycle_basis = [*boundaries, *homology_basis]
            image_coordinates: list[list[int | Fraction]] = []
            for vector in filtered_cycles:
                if any(_mat_vec(outgoing, vector, prime)):
                    raise OperationDomainValidationError(
                        location=("filtration", level, degree),
                        code="spectral_sequence.abutment_cycle_replay_failed",
                        message="a filtered homology representative is not a cycle",
                    )
                coordinates = _coordinates(full_cycle_basis, vector, prime)
                class_coordinates = coordinates[len(boundaries) :]
                if not _in_span(image_coordinates, class_coordinates, prime):
                    image_coordinates.append(class_coordinates)
            filtration_images[level][degree] = image_coordinates
    return homology_data, filtration_images


def _compare_stable_component(
    request: SpectralAbutmentRequest,
    page: SpectralPageResult,
    filtration_images: list[list[list[list[Any]]]],
    level: int,
    degree: int,
    boundaries: list[list[Any]],
    homology_basis: list[list[Any]],
) -> SpectralAbutmentComparison:
    """Construct and check the representative-induced map to one Gr H term."""
    prime = request.complex.prime
    lower = filtration_images[level - 1][degree] if level else []
    quotient_basis = _quotient_extension(lower, filtration_images[level][degree], prime)
    page_representatives = page.page_representatives[level][degree]
    full_cycle_basis = [*boundaries, *homology_basis]
    quotient_coordinates = [*lower, *quotient_basis]
    outgoing = (
        [
            [_parse_entry(value, prime) for value in row]
            for row in request.complex.differential_matrices[degree - 1]
        ]
        if degree
        else []
    )
    columns = []
    for representative in page_representatives:
        chain = [_parse_entry(value, prime) for value in representative]
        if any(_mat_vec(outgoing, chain, prime)):
            raise OperationDomainValidationError(
                location=("page", level, degree),
                code="spectral_sequence.abutment_page_representative_not_cycle",
                message="a stable-page representative is not a homology cycle",
            )
        coordinates = _coordinates(full_cycle_basis, chain, prime)
        homology_coordinates = coordinates[len(boundaries) :]
        try:
            quotient_class = _coordinates(
                quotient_coordinates, homology_coordinates, prime
            )
        except ValueError as exc:
            raise OperationDomainValidationError(
                location=("page", level, degree),
                code="spectral_sequence.abutment_image_not_in_filtration",
                message="a stable-page class does not land in the matching homology filtration quotient",
            ) from exc
        columns.append(quotient_class[len(lower) :])
    matrix = tuple(
        tuple(
            _serialize_scalar(columns[column][row], prime)
            for column in range(len(columns))
        )
        for row in range(len(quotient_basis))
    )
    rank = _rank_of(
        [[_parse_entry(value, prime) for value in row] for row in matrix], prime
    )
    if len(quotient_basis) != len(page_representatives) or rank != len(
        page_representatives
    ):
        raise OperationDomainValidationError(
            location=("page", level, degree),
            code="spectral_sequence.abutment_comparison_not_isomorphism",
            message="the stable-page comparison must be an isomorphism onto the graded homology quotient",
        )
    return SpectralAbutmentComparison(
        filtration_level=level,
        degree=request.complex.degree_min + degree,
        page_dimension=len(page_representatives),
        homology_graded_dimension=len(quotient_basis),
        homology_graded_basis_coordinates=tuple(
            tuple(_serialize_scalar(value, prime) for value in vector)
            for vector in quotient_basis
        ),
        matrix=matrix,
    )


def abutment(request: SpectralAbutmentRequest) -> SpectralAbutmentResult:
    # The comparison is part of the abutment postcondition: a stable page is
    # only identified with the associated graded of homology after the exact
    # representative map has been shown to be an isomorphism.
    sizes = request.complex.basis_sizes
    levels = len(request.filtration)
    comparison_cells = levels * sum(size * size for size in sizes)
    if comparison_cells > MAX_FILTERED_HOMOLOGY_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="spectral_sequence.abutment_comparison_bound",
            message=(
                f"the abutment comparison needs {comparison_cells} matrix cells, "
                "above the admitted exact-result envelope"
            ),
        )
    work = sum(
        (levels + 2) * (size + 1) ** 4
        + levels * MAX_FILTER_VECTORS_PER_GROUP * (size + 1) ** 3
        + levels
        * (size + 1) ** 2
        * ((sizes[index + 1] if index + 1 < len(sizes) else 0) + 1)
        for index, size in enumerate(sizes)
    )
    if work > MAX_FILTERED_HOMOLOGY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex", "basis_sizes"),
            code="spectral_sequence.abutment_work_bound",
            message=(
                f"the homology and comparison work estimate {work} exceeds "
                f"the admitted envelope of {MAX_FILTERED_HOMOLOGY_WORK}"
            ),
        )
    _admit_filtered_structure(request.complex, request.filtration)
    input_scalars = [
        entry
        for matrix in request.complex.differential_matrices
        for row in matrix
        for entry in row
    ] + [
        entry
        for level in request.filtration
        for subspace in level.subspaces
        for vector in subspace.vectors
        for entry in vector
    ]
    if request.complex.prime is not None:
        coefficient_digits = len(str(request.complex.prime))
    else:
        coefficient_digits = (
            max(
                (
                    max(
                        len(str(abs(entry.numerator))),
                        len(str(entry.denominator)),
                    )
                    if isinstance(entry, Fraction)
                    else len(str(abs(entry)))
                )
                for entry in input_scalars
            )
            if input_scalars
            else 1
        )
    maximum_dimension = max(sizes, default=0)
    result_scalar_chars = (
        coefficient_digits
        if request.complex.prime is not None
        else 2
        * maximum_dimension
        * (coefficient_digits + max(1, len(str(maximum_dimension))))
    )
    output_chars = comparison_cells * (result_scalar_chars + 4) + 100_000
    if output_chars > MAX_FILTERED_HOMOLOGY_RESULT_CHARS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="spectral_sequence.abutment_output_bound",
            message=(
                f"the exact abutment comparison is estimated at {output_chars} "
                f"characters, above {MAX_FILTERED_HOMOLOGY_RESULT_CHARS}"
            ),
        )
    admitted = _admit_filtered_semantics(request.complex, request.filtration)
    profile = pages_through(
        SpectralPagesRequest(complex=request.complex, filtration=request.filtration)
    )
    if profile.stabilized_page is None:
        raise OperationDomainValidationError(
            location=("filtration",),
            code="spectral_sequence.not_stabilized",
            message="the admitted page window did not establish an abutment",
        )
    page = profile.pages[-1]
    homology_data, filtration_images = _exact_filtered_homology_coordinates(
        request, admitted
    )
    comparisons: list[tuple[SpectralAbutmentComparison, ...]] = []
    for level in range(levels):
        degree_comparisons = []
        for degree, (boundaries, homology_basis) in enumerate(homology_data):
            degree_comparisons.append(
                _compare_stable_component(
                    request,
                    page,
                    filtration_images,
                    level,
                    degree,
                    boundaries,
                    homology_basis,
                )
            )
        comparisons.append(tuple(degree_comparisons))
    return SpectralAbutmentResult(
        complex=request.complex,
        page=page,
        status="STABILIZED",
        homology=tuple(
            FilteredHomologyDegree(
                degree=request.complex.degree_min + index,
                cycle_basis=tuple(
                    tuple(
                        _serialize_scalar(value, request.complex.prime)
                        for value in vector
                    )
                    for vector in (*boundaries, *homology_basis)
                ),
                boundary_basis=tuple(
                    tuple(
                        _serialize_scalar(value, request.complex.prime)
                        for value in vector
                    )
                    for vector in boundaries
                ),
                homology_basis=tuple(
                    tuple(
                        _serialize_scalar(value, request.complex.prime)
                        for value in vector
                    )
                    for vector in homology_basis
                ),
            )
            for index, (boundaries, homology_basis) in enumerate(homology_data)
        ),
        comparisons=tuple(comparisons),
    )


__all__ = [
    "FilteredChainMapCompositionRequest",
    "FilteredChainMapPageRequest",
    "FilteredChainMapPageResult",
    "FilteredChainMapPageZeroResult",
    "FilteredChainMapRequest",
    "FilteredChainMapResult",
    "FilteredHomologyDegree",
    "FilteredHomologyLevel",
    "FilteredHomologyResult",
    "HomologyFiltrationImage",
    "SpectralAbutmentComparison",
    "SpectralAbutmentRequest",
    "SpectralAbutmentResult",
    "SpectralPagesRequest",
    "SpectralPagesResult",
    "abutment",
    "filtered_chain_map_compose",
    "filtered_chain_map_page",
    "filtered_chain_map_page_zero",
    "filtered_homology_filtration",
    "filtered_map",
    "pages_through",
]
