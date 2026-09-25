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
    _transpose,
    admit_filtered,
    spectral_page,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainCoefficient,
    ChainComplexValue,
    CoefficientRing,
)

MAX_FILTERED_HOMOLOGY_PRIME = 2**31 - 1
MAX_FILTERED_HOMOLOGY_RESULT_CELLS = 250_000
MAX_FILTERED_HOMOLOGY_RESULT_CHARS = 20_000_000
MAX_FILTERED_HOMOLOGY_WORK = 50_000_000


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
    return _filtered_map_admitted(request, source_admission, target_admission)


def _filtered_map_admitted(
    request: FilteredChainMapRequest,
    source_admission: Any,
    target_admission: Any,
) -> FilteredChainMapResult:
    p = request.source.prime
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
                [_parse_entry(entry, prime) for entry in vector]
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
            image_coordinates = []
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
    "filtered_chain_map_page_zero",
    "filtered_homology_filtration",
    "filtered_map",
    "pages_through",
]
