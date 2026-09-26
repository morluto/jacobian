"""Finite simplicial maps and normalized-chain prefixes."""

from __future__ import annotations

from fractions import Fraction

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    ExactInteger,
    format_canonical_integer,
)
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._integral_homology import (
    admit_integral_homology,
)
from jacobian.math.topology.chain_complexes.operations import (
    chain_map_commutes,
    homology_groups,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_CHAIN_MAP_CELLS,
    MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK,
    MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS,
    MAX_INTEGRAL_HOMOLOGY_WORK_UNITS,
    MAX_OPERATION_MATRIX_CELLS,
    ChainComplexValue,
    ChainMapValue,
    CoefficientRing,
    HomologyGroup,
    IntegralHomologyGroupValue,
    IntegralTorsionGenerator,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_NORMALIZED_CHAIN_OUTPUT_CELLS = 256_000
MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS = 600_000
MAX_INDUCED_CHAIN_MAP_WORK_UNITS = 1_000_000
MAX_HOMOLOGY_MAP_COMPOSITION_WORK = 1_000_000
MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK = 100_000_000
_NORMALIZED_CHAIN_OUTPUT_STRUCTURE = 4_096


def _normalized_output_cells(
    simplicial_set: FiniteTruncatedSimplicialSet, matrix_cells: int
) -> int:
    """Bound retained label characters, matrix entries, and structure."""
    label_chars = sum(len(label) for level in simplicial_set.sets for label in level)
    return _NORMALIZED_CHAIN_OUTPUT_STRUCTURE + 2 * label_chars + 4 * matrix_cells


class SimplicialMapRequest(StrictModel):
    source: FiniteTruncatedSimplicialSet
    target: FiniteTruncatedSimplicialSet
    maps: tuple[tuple[int, ...], ...]


class SimplicialMapResult(StrictModel):
    source: FiniteTruncatedSimplicialSet
    target: FiniteTruncatedSimplicialSet
    maps: tuple[tuple[int, ...], ...]
    identities_preserved: bool


class TruncatedSimplicialMap(StrictModel):
    """A degreewise map between two finite prefixes.

    The operation that consumes this value rechecks naturality. The model
    validates only the structural axes, since serialized caller values are
    not mathematical evidence that the face and degeneracy squares commute.
    """

    source: FiniteTruncatedSimplicialSet
    target: FiniteTruncatedSimplicialSet
    maps: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_degreewise_axes(self) -> TruncatedSimplicialMap:
        if self.source.max_degree != self.target.max_degree:
            raise ValueError("source and target prefixes must have equal degree")
        if len(self.maps) != self.source.max_degree + 1:
            raise ValueError("maps must cover each degree in the prefix")
        for degree, row in enumerate(self.maps):
            if len(row) != len(self.source.sets[degree]) or any(
                type(index) is not int or not 0 <= index < len(self.target.sets[degree])
                for index in row
            ):
                raise ValueError(f"map row {degree} has invalid simplex axes")
        return self


class SimplicialMapCompositionRequest(StrictModel):
    """Two maps X -> Y and Y -> Z, represented by their exact prefixes."""

    first: TruncatedSimplicialMap
    second: TruncatedSimplicialMap


def _require_naturality(map_value: TruncatedSimplicialMap, *, location: str) -> None:
    """Replay every visible face and degeneracy square for a supplied map."""
    source, target, maps = map_value.source, map_value.target, map_value.maps
    for degree in range(1, source.max_degree + 1):
        for face_index, face in enumerate(source.face_maps[degree - 1]):
            target_face = target.face_maps[degree - 1][face_index]
            for simplex, face_image in enumerate(face):
                if target_face[maps[degree][simplex]] != maps[degree - 1][face_image]:
                    raise OperationDomainValidationError(
                        location=(location, "maps", degree, simplex),
                        code="simplicial_map.face_naturality_failed",
                        message=(
                            f"map does not commute with d_{face_index} in "
                            f"degree {degree} at simplex index {simplex}"
                        ),
                    )
    for degree in range(source.max_degree):
        for degeneracy_index, degeneracy in enumerate(source.degeneracy_maps[degree]):
            target_degeneracy = target.degeneracy_maps[degree][degeneracy_index]
            for simplex, degeneracy_image in enumerate(degeneracy):
                if (
                    target_degeneracy[maps[degree][simplex]]
                    != maps[degree + 1][degeneracy_image]
                ):
                    raise OperationDomainValidationError(
                        location=(location, "maps", degree, simplex),
                        code="simplicial_map.degeneracy_naturality_failed",
                        message=(
                            f"map does not commute with s_{degeneracy_index} in "
                            f"degree {degree} at simplex index {simplex}"
                        ),
                    )


def _require_carrier(
    value: FiniteTruncatedSimplicialSet, *, location: str
) -> FiniteTruncatedSimplicialSet:
    checked = from_tables(
        value.max_degree, value.sets, value.face_maps, value.degeneracy_maps
    )
    if checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=(location,),
            code="simplicial_map.carrier_invalid",
            message="map carrier fails a visible simplicial identity",
        )
    return checked.simplicial_set


def compose_simplicial_maps(
    request: SimplicialMapCompositionRequest,
) -> TruncatedSimplicialMap:
    """Compose finite-prefix simplicial maps after checking their claims."""
    first, second = request.first, request.second
    first_source = _require_carrier(first.source, location="first.source")
    first_target = _require_carrier(first.target, location="first.target")
    second_source = _require_carrier(second.source, location="second.source")
    second_target = _require_carrier(second.target, location="second.target")
    if first_target != second_source:
        raise OperationDomainValidationError(
            location=("second", "source"),
            code="simplicial_map.composition_carrier_mismatch",
            message="the first map target must equal the second map source",
        )
    _require_naturality(first, location="first")
    _require_naturality(second, location="second")
    maps = tuple(
        tuple(second.maps[degree][image] for image in first.maps[degree])
        for degree in range(first.source.max_degree + 1)
    )
    # Naturality follows by composition, but replay it at this public boundary
    # so the returned map is independently checked against its bound carriers.
    composite = TruncatedSimplicialMap(
        source=first_source, target=second_target, maps=maps
    )
    _require_naturality(composite, location="composite")
    return composite


def identity_simplicial_map(
    simplicial_set: FiniteTruncatedSimplicialSet,
) -> TruncatedSimplicialMap:
    """Return the identity map on every degree of a finite prefix."""
    maps = tuple(tuple(range(len(level))) for level in simplicial_set.sets)
    return TruncatedSimplicialMap(
        source=simplicial_set, target=simplicial_set, maps=maps
    )


class NormalizedChainsRequest(StrictModel):
    simplicial_set: FiniteTruncatedSimplicialSet


class NormalizedChainsResult(StrictModel):
    """Normalized chain value with source labels bound to its rank axes.

    Decoding checks structural source/axis correspondence only. Consumers that
    rely on the source-to-boundary relation recheck it when admitting this
    caller-supplied value; model validation does not replay the boundary build.
    """

    simplicial_set: FiniteTruncatedSimplicialSet
    nondegenerate_bases: tuple[tuple[str, ...], ...]
    chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_source_bound_chain_axes(self) -> NormalizedChainsResult:
        if len(self.nondegenerate_bases) != self.simplicial_set.max_degree + 1:
            raise ValueError("nondegenerate bases must cover every source degree")
        expected_bases = []
        for degree, level in enumerate(self.simplicial_set.sets):
            degenerate = {
                simplex
                for row in (
                    self.simplicial_set.degeneracy_maps[degree - 1] if degree else ()
                )
                for simplex in row
            }
            expected_bases.append(
                tuple(
                    label
                    for index, label in enumerate(level)
                    if index not in degenerate
                )
            )
        if tuple(expected_bases) != self.nondegenerate_bases:
            raise ValueError("normalized axes do not match source degeneracies")
        value = self.chain_complex
        if (
            value.degree_min != 0
            or value.degree_max != self.simplicial_set.max_degree
            or value.basis_sizes != tuple(map(len, self.nondegenerate_bases))
            or value.coefficient_ring is not CoefficientRing.INTEGER
        ):
            raise ValueError("chain complex must use the normalized source axes")
        return self


class NormalizedHomologyResult(StrictModel):
    """Exact homology in degrees whose incoming differential is present."""

    simplicial_set: FiniteTruncatedSimplicialSet
    nondegenerate_bases: tuple[tuple[str, ...], ...]
    chain_complex: ChainComplexValue
    homology_groups: tuple[HomologyGroup, ...]

    @model_validator(mode="after")
    def require_supported_degrees(self) -> NormalizedHomologyResult:
        if len(self.nondegenerate_bases) != self.simplicial_set.max_degree + 1:
            raise ValueError("nondegenerate bases must cover every source degree")
        if tuple(group.degree for group in self.homology_groups) != tuple(
            range(self.simplicial_set.max_degree)
        ):
            raise ValueError("homology must cover exactly degrees below the prefix top")
        if (
            self.chain_complex.degree_min != 0
            or self.chain_complex.degree_max != self.simplicial_set.max_degree
            or self.chain_complex.basis_sizes
            != tuple(map(len, self.nondegenerate_bases))
            or self.chain_complex.coefficient_ring is not CoefficientRing.INTEGER
        ):
            raise ValueError("chain axes must retain the normalized source bases")
        expected_bases = []
        for degree, level in enumerate(self.simplicial_set.sets):
            degenerate = {
                simplex
                for row in (
                    self.simplicial_set.degeneracy_maps[degree - 1] if degree else ()
                )
                for simplex in row
            }
            expected_bases.append(
                tuple(
                    label
                    for index, label in enumerate(level)
                    if index not in degenerate
                )
            )
        if tuple(expected_bases) != self.nondegenerate_bases:
            raise ValueError("normalized basis labels do not match source degeneracies")
        for degree, group in enumerate(self.homology_groups):
            if not isinstance(group, IntegralHomologyGroupValue):
                raise ValueError(
                    "integral normalized homology requires integral groups"
                )
            expected_outgoing = (
                self.chain_complex.differential_matrices[degree - 1] if degree else ()
            )
            certificate = group.outgoing_smith_certificate.source
            if (
                group.chain_rank != self.chain_complex.basis_sizes[degree]
                or group.incoming_chain_rank
                != self.chain_complex.basis_sizes[degree + 1]
                or certificate.column_count != group.chain_rank
                or certificate.entries != expected_outgoing
            ):
                raise ValueError("homology representatives do not match source axes")
        return self


class IntegralHomologyCoordinates(StrictModel):
    """Coordinates in a retained free and invariant-factor basis."""

    free: tuple[ExactInteger, ...] = Field(max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK)
    torsion: tuple[ExactInteger, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK
    )


class NormalizedHomologyDegreeMap(StrictModel):
    """Images of the canonical free and torsion generators in one degree."""

    degree: StrictInt = Field(ge=0, le=4)
    free_generator_images: tuple[IntegralHomologyCoordinates, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK
    )
    torsion_generator_images: tuple[IntegralHomologyCoordinates, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK
    )


class SimplicialHomologyMapValue(StrictModel):
    """An induced map between the supported normalized integral homology."""

    simplicial_map: TruncatedSimplicialMap
    source: NormalizedHomologyResult
    target: NormalizedHomologyResult
    degree_maps: tuple[NormalizedHomologyDegreeMap, ...] = Field(max_length=4)

    @model_validator(mode="after")
    def require_source_bound_matrices(self) -> SimplicialHomologyMapValue:
        if (
            self.source.simplicial_set != self.simplicial_map.source
            or self.target.simplicial_set != self.simplicial_map.target
        ):
            raise ValueError("homology endpoints must match the simplicial map")
        supported_degrees = self.simplicial_map.source.max_degree
        if self.simplicial_map.target.max_degree != supported_degrees or tuple(
            item.degree for item in self.degree_maps
        ) != tuple(range(supported_degrees)):
            raise ValueError("homology maps must cover exactly the supported degrees")
        for degree, matrix in enumerate(self.degree_maps):
            source_group = self.source.homology_groups[degree]
            target_group = self.target.homology_groups[degree]
            assert isinstance(source_group, IntegralHomologyGroupValue)
            assert isinstance(target_group, IntegralHomologyGroupValue)
            target_torsion = target_group.torsion_invariant_factors
            if len(matrix.free_generator_images) != source_group.free_rank or len(
                matrix.torsion_generator_images
            ) != len(source_group.torsion_generators):
                raise ValueError("homology images must cover every source generator")
            for image in (
                *matrix.free_generator_images,
                *matrix.torsion_generator_images,
            ):
                if len(image.free) != target_group.free_rank or len(
                    image.torsion
                ) != len(target_torsion):
                    raise ValueError("homology image coordinates have invalid axes")
                if any(
                    coordinate < 0 or coordinate >= order
                    for coordinate, order in zip(
                        image.torsion, target_torsion, strict=True
                    )
                ):
                    raise ValueError("torsion coordinates must be canonical residues")
            for generator, image in zip(
                source_group.torsion_generators,
                matrix.torsion_generator_images,
                strict=True,
            ):
                order = int(generator.order)
                if any(image.free) or any(
                    order * int(coordinate) % int(target_order)
                    for coordinate, target_order in zip(
                        image.torsion, target_torsion, strict=True
                    )
                ):
                    raise ValueError("torsion generator image violates its order")
        return self


def simplicial_map(request: SimplicialMapRequest) -> SimplicialMapResult:
    s, t = request.source, request.target
    if s.max_degree != t.max_degree or len(request.maps) != s.max_degree + 1:
        raise OperationDomainValidationError(
            location=("maps",),
            code="simplicial_map.degree_axis",
            message="map must cover every degree of the retained prefixes",
        )
    for n, row in enumerate(request.maps):
        if len(row) != len(s.sets[n]) or any(
            not isinstance(i, int) or i < 0 or i >= len(t.sets[n]) for i in row
        ):
            raise OperationDomainValidationError(
                location=("maps", n),
                code="simplicial_map.row_axis",
                message="map rows must cover source simplices with target indices",
            )
    ok = True
    for n in range(1, s.max_degree + 1):
        for i, face in enumerate(s.face_maps[n - 1]):
            for q, x in enumerate(face):
                if request.maps[n][q] >= len(t.face_maps[n - 1][i]) or (
                    t.face_maps[n - 1][i][request.maps[n][q]] != request.maps[n - 1][x]
                ):
                    ok = False
    for n in range(s.max_degree):
        for i, deg in enumerate(s.degeneracy_maps[n]):
            for q, x in enumerate(deg):
                if (
                    t.degeneracy_maps[n][i][request.maps[n][q]]
                    != request.maps[n + 1][x]
                ):
                    ok = False
    return SimplicialMapResult(
        source=s, target=t, maps=request.maps, identities_preserved=ok
    )


def normalized_chains(
    simplicial_set: FiniteTruncatedSimplicialSet,
) -> NormalizedChainsResult:
    source_sizes = tuple(len(level) for level in simplicial_set.sets)
    cells = sum(
        source_sizes[degree - 1] * source_sizes[degree]
        for degree in range(1, len(source_sizes))
    )
    if cells > MAX_OPERATION_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_chain_matrix_budget_exceeded",
            message=(
                f"normalized boundary matrices may require {cells} cells, "
                f"exceeding the {MAX_OPERATION_MATRIX_CELLS}-cell construction bound"
            ),
        )
    output_bound = _normalized_output_cells(simplicial_set, cells)
    if output_bound > MAX_NORMALIZED_CHAIN_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_chain_output_budget_exceeded",
            message=(
                f"estimated normalized chain result size {output_bound} cells "
                f"exceeds the {MAX_NORMALIZED_CHAIN_OUTPUT_CELLS}-cell output bound"
            ),
        )
    checked = from_tables(
        simplicial_set.max_degree,
        simplicial_set.sets,
        simplicial_set.face_maps,
        simplicial_set.degeneracy_maps,
    )
    if checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_source_invalid",
            message="source tables fail a visible simplicial identity",
        )
    s = checked.simplicial_set
    nd = []
    for n, level in enumerate(s.sets):
        degenerate = {x for row in (s.degeneracy_maps[n - 1] if n else ()) for x in row}
        nd.append(tuple(i for i in range(len(level)) if i not in degenerate))
    matrices = []
    for n in range(1, s.max_degree + 1):
        lower = nd[n - 1]
        lower_index = {x: i for i, x in enumerate(lower)}
        matrix = []
        matrix = [[0] * len(nd[n]) for _ in lower]
        for col, q in enumerate(nd[n]):
            for i, face in enumerate(s.face_maps[n - 1]):
                target = face[q]
                if target in lower_index:
                    matrix[lower_index[target]][col] += (-1) ** i
        matrices.append(tuple(tuple(v for v in row) for row in matrix))
    square = True
    for i in range(len(matrices) - 1):
        a, b = matrices[i], matrices[i + 1]
        prod = [
            [
                sum(a[r][k] * b[k][c] for k in range(len(b)))
                for c in range(len(b[0]) if b else 0)
            ]
            for r in range(len(a))
        ]
        square = square and all(v == 0 for row in prod for v in row)
    if not square:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_chain_identity_failed",
            message="normalized differential does not square to zero",
        )
    bases = tuple(
        tuple(s.sets[degree][index] for index in row) for degree, row in enumerate(nd)
    )
    chain = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=s.max_degree,
        basis_sizes=tuple(map(len, nd)),
        differential_matrices=tuple(matrices),
    )
    return NormalizedChainsResult(
        simplicial_set=s,
        nondegenerate_bases=bases,
        chain_complex=chain,
    )


def induced_normalized_chain_map(
    map_value: TruncatedSimplicialMap,
) -> ChainMapValue:
    """Return the normalized chain map induced by a finite simplicial map.

    The component in degree ``n`` sends a nondegenerate source simplex to its
    image if that image is nondegenerate, and to zero otherwise. Basis labels
    retain the exact simplex axes at each endpoint.
    """
    source = _require_carrier(map_value.source, location="source")
    target = _require_carrier(map_value.target, location="target")
    try:
        map_value = TruncatedSimplicialMap.model_validate(
            {"source": source, "target": target, "maps": map_value.maps}
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("map",),
            code="simplicial_map.degreewise_axes_invalid",
            message="map rows must match the canonical source and target axes",
        ) from exc
    source_sizes = tuple(map(len, source.sets))
    target_sizes = tuple(map(len, target.sets))
    source_nondegenerate = tuple(
        tuple(
            simplex
            for simplex in range(len(level))
            if simplex
            not in {
                image
                for row in (source.degeneracy_maps[degree - 1] if degree else ())
                for image in row
            }
        )
        for degree, level in enumerate(source.sets)
    )
    target_nondegenerate = tuple(
        tuple(
            simplex
            for simplex in range(len(level))
            if simplex
            not in {
                image
                for row in (target.degeneracy_maps[degree - 1] if degree else ())
                for image in row
            }
        )
        for degree, level in enumerate(target.sets)
    )
    map_cells = sum(
        len(source_basis) * len(target_basis)
        for source_basis, target_basis in zip(
            source_nondegenerate, target_nondegenerate, strict=True
        )
    )
    source_chain_cells = sum(
        source_sizes[degree - 1] * source_sizes[degree]
        for degree in range(1, len(source_sizes))
    )
    target_chain_cells = sum(
        target_sizes[degree - 1] * target_sizes[degree]
        for degree in range(1, len(target_sizes))
    )
    relation_work = sum(
        len(target_nondegenerate[degree - 1])
        * len(target_nondegenerate[degree])
        * len(source_nondegenerate[degree])
        + len(target_nondegenerate[degree - 1])
        * len(source_nondegenerate[degree - 1])
        * len(source_nondegenerate[degree])
        for degree in range(1, len(source_nondegenerate))
    )
    work_bound = (
        map_cells
        + source_chain_cells
        + target_chain_cells
        + relation_work
        + 16 * (source.total_simplices + target.total_simplices)
    )
    if work_bound > MAX_INDUCED_CHAIN_MAP_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("map",),
            code="simplicial_set.induced_chain_map_work_budget_exceeded",
            message=(
                f"normalized chain-map construction estimates {work_bound} work "
                f"units, exceeding the {MAX_INDUCED_CHAIN_MAP_WORK_UNITS}-unit bound"
            ),
        )
    if map_cells > MAX_CHAIN_MAP_CELLS:
        raise OperationResourceAdmissionError(
            location=("map",),
            code="simplicial_set.induced_chain_map_cell_budget_exceeded",
            message=(
                f"normalized map matrices require {map_cells} cells, exceeding "
                f"the {MAX_CHAIN_MAP_CELLS}-cell chain-map bound"
            ),
        )
    source_chars = sum(len(label) for level in source.sets for label in level)
    target_chars = sum(len(label) for level in target.sets for label in level)
    output_bound = (
        _NORMALIZED_CHAIN_OUTPUT_STRUCTURE * 2
        + 4 * (source_chain_cells + target_chain_cells + map_cells)
        + 2 * (source_chars + target_chars)
    )
    if output_bound > MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("map",),
            code="simplicial_set.induced_chain_map_output_budget_exceeded",
            message=(
                f"estimated normalized chain-map output {output_bound} cells "
                f"exceeds the {MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS}-cell bound"
            ),
        )

    normalized_source = normalized_chains(source)
    normalized_target = (
        normalized_source if target == source else normalized_chains(target)
    )
    _require_naturality(map_value, location="map")
    target_rows = tuple(
        {simplex: index for index, simplex in enumerate(level)}
        for level in target_nondegenerate
    )
    components = []
    for degree, source_basis in enumerate(source_nondegenerate):
        target_basis = target_nondegenerate[degree]
        matrix = [[0] * len(source_basis) for _ in target_basis]
        for column, simplex in enumerate(source_basis):
            image = map_value.maps[degree][simplex]
            row = target_rows[degree].get(image)
            if row is not None:
                matrix[row][column] = 1
        components.append(tuple(tuple(row) for row in matrix))
    chain_map = ChainMapValue(
        source=normalized_source.chain_complex,
        target=normalized_target.chain_complex,
        map_matrices=tuple(components),
        source_basis_labels=normalized_source.nondegenerate_bases,
        target_basis_labels=normalized_target.nondegenerate_bases,
    )
    relation = chain_map_commutes(chain_map)
    if not relation.is_valid:
        raise OperationDomainValidationError(
            location=("map",),
            code="simplicial_set.induced_chain_map_identity_failed",
            message="the induced normalized boundary does not commute with the map",
        )
    return chain_map


def _mat_vec(
    matrix: tuple[tuple[int, ...], ...], vector: tuple[int, ...]
) -> tuple[int, ...]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector axes are incompatible")
    return tuple(
        sum(entry * vector[index] for index, entry in enumerate(row)) for row in matrix
    )


def _integer_matrix(
    matrix: tuple[tuple[int | Fraction, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row in matrix:
        integer_row: list[int] = []
        for entry in row:
            if type(entry) is not int:
                raise OperationDomainValidationError(
                    location=("chain_map",),
                    code="simplicial_set.induced_homology_requires_integral_map",
                    message="normalized simplicial homology maps require integral coefficients",
                )
            integer_row.append(entry)
        rows.append(tuple(integer_row))
    return tuple(rows)


def _admit_projection_mat_vec(
    matrix: tuple[tuple[int, ...], ...],
    input_bits: int,
    *,
    entry_bits: int | None = None,
) -> tuple[int, int]:
    if not matrix:
        return 0, 0
    columns = len(matrix[0])
    if not columns:
        return 0, 0
    if entry_bits is None:
        entry_bits = max(
            (abs(value).bit_length() for row in matrix for value in row), default=1
        )
    output_bits = entry_bits + input_bits + (columns - 1).bit_length()
    output_digits = (output_bits * 30_103 + 99_999) // 100_000
    if output_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_projection_height_exceeded",
            message="homology-coordinate projection exceeds the admitted exact integer height",
        )
    entry_limbs = max(1, (entry_bits + 63) // 64)
    input_limbs = max(1, (input_bits + 63) // 64)
    output_limbs = max(1, (output_bits + 63) // 64)
    work = sum(len(row) * (entry_limbs * input_limbs + output_limbs) for row in matrix)
    return output_bits, work


def _admit_homology_projection(
    chain_map: ChainMapValue,
    source: NormalizedHomologyResult,
    target: NormalizedHomologyResult,
    target_right_inverses: list[list[list[int]]],
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    source_differentials = tuple(
        _integer_matrix(matrix) for matrix in chain_map.source.differential_matrices
    )
    target_differentials = tuple(
        _integer_matrix(matrix) for matrix in chain_map.target.differential_matrices
    )
    map_matrices = tuple(_integer_matrix(matrix) for matrix in chain_map.map_matrices)
    inverse_matrices = tuple(
        tuple(tuple(row) for row in inverse) for inverse in target_right_inverses
    )
    all_matrices = (
        *source_differentials,
        *target_differentials,
        *map_matrices,
        *inverse_matrices,
        *(
            group.incoming_smith_certificate.left_transformation.entries
            for group in target.homology_groups
        ),
    )
    entry_bits_by_identity = {
        id(matrix): max(
            (abs(value).bit_length() for row in matrix for value in row), default=1
        )
        for matrix in all_matrices
    }
    metadata_scan_work = sum(sum(len(row) for row in matrix) for matrix in all_matrices)
    request_checkpoint("after homology-coordinate matrix admission")
    if metadata_scan_work > MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_projection_work_exceeded",
            message="homology-coordinate matrix scans exceed the admitted work envelope",
        )

    def admit_mat_vec(
        matrix: tuple[tuple[int, ...], ...], input_bits: int
    ) -> tuple[int, int]:
        return _admit_projection_mat_vec(
            matrix,
            input_bits,
            entry_bits=entry_bits_by_identity[id(matrix)],
        )

    work = metadata_scan_work
    output_scalars = 0
    for degree, (source_group, target_group) in enumerate(
        zip(source.homology_groups, target.homology_groups, strict=True)
    ):
        assert isinstance(source_group, IntegralHomologyGroupValue)
        assert isinstance(target_group, IntegralHomologyGroupValue)
        inverse_right = inverse_matrices[degree]
        incoming_left = (
            target_group.incoming_smith_certificate.left_transformation.entries
        )
        target_coordinates = target_group.free_rank + sum(
            int(order) > 1 for order in target_group.torsion_invariant_factors
        )
        generators = (
            *source_group.free_generators,
            *source_group.torsion_generators,
        )
        output_scalars += len(generators) * target_coordinates
        for generator in generators:
            request_checkpoint("during induced homology coordinate admission")
            cycle = tuple(generator.cycle.coefficients)
            cycle_bits = max((abs(value).bit_length() for value in cycle), default=0)
            if degree:
                _, cost = admit_mat_vec(source_differentials[degree - 1], cycle_bits)
                work += cost
            image_bits, cost = admit_mat_vec(map_matrices[degree], cycle_bits)
            work += cost
            if degree:
                _, cost = admit_mat_vec(target_differentials[degree - 1], image_bits)
                work += cost
            coordinate_bits, cost = admit_mat_vec(inverse_right, image_bits)
            work += cost
            _, cost = admit_mat_vec(incoming_left, coordinate_bits)
            work += cost
            if isinstance(generator, IntegralTorsionGenerator):
                bounding = tuple(generator.bounding_chain.coefficients)
                bounding_bits = max(
                    (abs(value).bit_length() for value in bounding), default=0
                )
                source_bound_bits, cost = admit_mat_vec(
                    source_differentials[degree], bounding_bits
                )
                work += cost
                mapped_bound_bits, cost = admit_mat_vec(
                    map_matrices[degree + 1], source_bound_bits
                )
                work += cost
                _, cost = admit_mat_vec(target_differentials[degree], mapped_bound_bits)
                work += cost
            if work > MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK:
                raise OperationResourceAdmissionError(
                    location=("map", "homology"),
                    code="simplicial_set.induced_homology_projection_work_exceeded",
                    message=(
                        "homology-coordinate projection exceeds the "
                        f"{MAX_HOMOLOGY_COORDINATE_PROJECTION_WORK}-unit work envelope"
                    ),
                )
    if output_scalars > MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_projection_output_exceeded",
            message="induced homology coordinates exceed their output reservation",
        )
    return source_differentials, target_differentials, map_matrices


def _coordinates_in_target_homology(
    cycle: tuple[int, ...],
    group: IntegralHomologyGroupValue,
    inverse_right: tuple[tuple[int, ...], ...],
) -> IntegralHomologyCoordinates:
    inverse_digits = max(
        (
            len(format_canonical_integer(abs(value)))
            for row in inverse_right
            for value in row
        ),
        default=1,
    )
    cycle_digits = max(
        (len(format_canonical_integer(abs(value))) for value in cycle), default=1
    )
    rank = len(cycle)
    growth_bound = inverse_digits + cycle_digits + len(str(max(1, rank)))
    if growth_bound > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_coordinate_growth_exceeded",
            message="target coordinate multiplication exceeds its admitted integer height",
        )
    chain_coordinates = _mat_vec(inverse_right, cycle)
    rank = group.outgoing_boundary_rank
    if any(chain_coordinates[:rank]):
        raise OperationDomainValidationError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_cycle_coordinates_invalid",
            message="mapped representative is not a cycle in the target complex",
        )
    cycle_coordinates = chain_coordinates[rank:]
    incoming = group.incoming_smith_certificate
    left_digits = max(
        (
            len(format_canonical_integer(abs(value)))
            for row in incoming.left_transformation.entries
            for value in row
        ),
        default=1,
    )
    coordinate_digits = max(
        (len(format_canonical_integer(abs(value))) for value in cycle_coordinates),
        default=1,
    )
    smith_rank = len(cycle_coordinates)
    smith_growth_bound = left_digits + coordinate_digits + len(str(max(1, smith_rank)))
    if smith_growth_bound > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("map", "homology"),
            code="simplicial_set.induced_homology_smith_coordinate_growth_exceeded",
            message="Smith coordinate multiplication exceeds its admitted integer height",
        )
    smith_coordinates = _mat_vec(
        incoming.left_transformation.entries, cycle_coordinates
    )
    boundary_rank = group.incoming_boundary_rank
    torsion = tuple(
        smith_coordinates[index] % int(order)
        for index, order in enumerate(incoming.invariant_factors)
        if int(order) > 1
    )
    return IntegralHomologyCoordinates(
        free=tuple(smith_coordinates[boundary_rank:]), torsion=torsion
    )


def _map_homology_generator(
    degree: int,
    cycle: tuple[int, ...],
    target_group: IntegralHomologyGroupValue,
    inverse_right: tuple[tuple[int, ...], ...],
    source_differentials: tuple[tuple[tuple[int, ...], ...], ...],
    target_differentials: tuple[tuple[tuple[int, ...], ...], ...],
    map_matrices: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    torsion_order: int | None = None,
    bounding_chain: tuple[int, ...] | None = None,
) -> IntegralHomologyCoordinates:
    request_checkpoint("during induced homology generator projection")
    if degree and any(
        _mat_vec(
            source_differentials[degree - 1],
            cycle,
        )
    ):
        raise OperationDomainValidationError(
            location=("source", "homology", degree),
            code="simplicial_set.induced_homology_source_cycle_invalid",
            message="a retained homology representative is not a source cycle",
        )
    image = _mat_vec(map_matrices[degree], cycle)
    if degree:
        outgoing = target_differentials[degree - 1]
        if any(_mat_vec(outgoing, image)):
            raise OperationDomainValidationError(
                location=("map", "homology", degree),
                code="simplicial_set.induced_homology_cycle_identity_failed",
                message="the induced chain map did not send a cycle to a cycle",
            )
    if torsion_order is not None:
        assert bounding_chain is not None
        source_boundary = _mat_vec(
            source_differentials[degree],
            bounding_chain,
        )
        if source_boundary != tuple(torsion_order * value for value in cycle):
            raise OperationDomainValidationError(
                location=("source", "homology", degree),
                code="simplicial_set.induced_homology_torsion_witness_invalid",
                message="the retained torsion representative has an invalid boundary witness",
            )
        mapped_bounding_chain = _mat_vec(map_matrices[degree + 1], bounding_chain)
        target_boundary = _mat_vec(
            target_differentials[degree],
            mapped_bounding_chain,
        )
        if target_boundary != tuple(torsion_order * value for value in image):
            raise OperationDomainValidationError(
                location=("map", "homology", degree),
                code="simplicial_set.induced_homology_torsion_identity_failed",
                message="the chain map failed to preserve a torsion bounding relation",
            )
    return _coordinates_in_target_homology(image, target_group, inverse_right)


def induced_normalized_homology_map(
    map_value: TruncatedSimplicialMap,
) -> SimplicialHomologyMapValue:
    """Compute the induced map on every homology degree supported by a prefix."""
    chain_map = induced_normalized_chain_map(map_value)
    source_plan = admit_integral_homology(chain_map.source)
    target_plan = (
        source_plan
        if map_value.target == map_value.source
        else admit_integral_homology(chain_map.target)
    )
    combined_work = source_plan.total_work + (
        0 if target_plan is source_plan else target_plan.total_work
    )
    combined_output = source_plan.output_scalar_count + (
        0 if target_plan is source_plan else target_plan.output_scalar_count
    )
    if combined_work > MAX_INTEGRAL_HOMOLOGY_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_set.induced_homology_endpoint_work_budget_exceeded",
            message=(
                "combined source and target integral-homology work exceeds "
                f"the {MAX_INTEGRAL_HOMOLOGY_WORK_UNITS}-unit envelope"
            ),
        )
    if combined_output > MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS:
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_set.induced_homology_endpoint_output_budget_exceeded",
            message=(
                "combined source and target homology output exceeds "
                f"the {MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS}-scalar envelope"
            ),
        )
    source_right_inverses: list[list[list[int]]] = []
    source = normalized_homology(
        map_value.source, _integral_right_inverses=source_right_inverses
    )
    target_right_inverses = source_right_inverses
    target = (
        source
        if map_value.target == map_value.source
        else normalized_homology(
            map_value.target, _integral_right_inverses=(target_right_inverses := [])
        )
    )
    source_differentials, target_differentials, map_matrices = (
        _admit_homology_projection(chain_map, source, target, target_right_inverses)
    )
    degree_maps: list[NormalizedHomologyDegreeMap] = []
    for degree, (source_group, target_group) in enumerate(
        zip(source.homology_groups, target.homology_groups, strict=True)
    ):
        assert isinstance(source_group, IntegralHomologyGroupValue)
        assert isinstance(target_group, IntegralHomologyGroupValue)
        inverse_right = tuple(tuple(row) for row in target_right_inverses[degree])
        degree_maps.append(
            NormalizedHomologyDegreeMap(
                degree=degree,
                free_generator_images=tuple(
                    _map_homology_generator(
                        degree,
                        tuple(generator.cycle.coefficients),
                        target_group,
                        inverse_right,
                        source_differentials,
                        target_differentials,
                        map_matrices,
                    )
                    for generator in source_group.free_generators
                ),
                torsion_generator_images=tuple(
                    _map_homology_generator(
                        degree,
                        tuple(generator.cycle.coefficients),
                        target_group,
                        inverse_right,
                        source_differentials,
                        target_differentials,
                        map_matrices,
                        torsion_order=int(generator.order),
                        bounding_chain=tuple(generator.bounding_chain.coefficients),
                    )
                    for generator in source_group.torsion_generators
                ),
            )
        )
    checked_map = TruncatedSimplicialMap(
        source=source.simplicial_set,
        target=target.simplicial_set,
        maps=map_value.maps,
    )
    return SimplicialHomologyMapValue(
        simplicial_map=checked_map,
        source=source,
        target=target,
        degree_maps=tuple(degree_maps),
    )


def _compose_homology_coordinates(
    coordinates: IntegralHomologyCoordinates,
    middle_images: NormalizedHomologyDegreeMap,
    target_group: IntegralHomologyGroupValue,
) -> IntegralHomologyCoordinates:
    free = [0] * target_group.free_rank
    torsion = [0] * len(target_group.torsion_invariant_factors)
    images = (
        *middle_images.free_generator_images,
        *middle_images.torsion_generator_images,
    )
    coefficients = (*coordinates.free, *coordinates.torsion)
    for coefficient, image in zip(coefficients, images, strict=True):
        for index, value in enumerate(image.free):
            free[index] += int(coefficient) * int(value)
        for index, value in enumerate(image.torsion):
            torsion[index] += int(coefficient) * int(value)
    torsion = [
        value % int(order)
        for value, order in zip(
            torsion, target_group.torsion_invariant_factors, strict=True
        )
    ]
    return IntegralHomologyCoordinates(free=tuple(free), torsion=tuple(torsion))


def compose_simplicial_homology_maps(
    first: SimplicialHomologyMapValue,
    second: SimplicialHomologyMapValue,
) -> SimplicialHomologyMapValue:
    """Compose two induced normalized integral homology maps."""
    if first.target != second.source:
        raise OperationDomainValidationError(
            location=("second", "source"),
            code="simplicial_set.homology_map_composition_mismatch",
            message="the first target homology must equal the second source homology",
        )
    # Admit the union of both endpoint computations before either computes
    # normalized homology. Reuse plans for equal complexes, including the
    # shared middle complex.
    first_chain = induced_normalized_chain_map(first.simplicial_map)
    second_chain = induced_normalized_chain_map(second.simplicial_map)
    plans = []
    for complex_value in (
        first_chain.source,
        first_chain.target,
        second_chain.source,
        second_chain.target,
    ):
        if not any(existing == complex_value for existing, _ in plans):
            plans.append((complex_value, admit_integral_homology(complex_value)))
    if sum(plan.total_work for _, plan in plans) > MAX_INTEGRAL_HOMOLOGY_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("homology_map",),
            code="simplicial_set.induced_homology_endpoint_work_budget_exceeded",
            message="combined endpoint homology work exceeds its admitted envelope",
        )
    if (
        sum(plan.output_scalar_count for _, plan in plans)
        > MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS
    ):
        raise OperationResourceAdmissionError(
            location=("homology_map",),
            code="simplicial_set.induced_homology_endpoint_output_budget_exceeded",
            message="combined endpoint homology output exceeds its admitted envelope",
        )
    # Authenticate the authored coordinates before using them as operands.
    checked_first = induced_normalized_homology_map(first.simplicial_map)
    checked_second = induced_normalized_homology_map(second.simplicial_map)
    if checked_first != first or checked_second != second:
        raise OperationDomainValidationError(
            location=("homology_map",),
            code="simplicial_set.homology_map_claim_mismatch",
            message="homology coordinates must agree with the induced simplicial maps",
        )
    composite_map = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=first.simplicial_map, second=second.simplicial_map
        )
    )
    composition_work = 0
    maximum_input_bits = 0
    maximum_middle_rank = 0
    for first_degree, second_degree in zip(
        first.degree_maps, second.degree_maps, strict=True
    ):
        first_images = (
            *first_degree.free_generator_images,
            *first_degree.torsion_generator_images,
        )
        second_images = (
            *second_degree.free_generator_images,
            *second_degree.torsion_generator_images,
        )
        middle_rank = len(second_images)
        maximum_middle_rank = max(maximum_middle_rank, middle_rank)
        target_rank = (
            len(second_images[0].free) + len(second_images[0].torsion)
            if second_images
            else 0
        )
        composition_work += len(first_images) * middle_rank * target_rank
        for coordinates in (*first_images, *second_images):
            maximum_input_bits = max(
                maximum_input_bits,
                *(
                    abs(value).bit_length()
                    for value in (*coordinates.free, *coordinates.torsion)
                ),
                0,
            )
    if composition_work > MAX_HOMOLOGY_MAP_COMPOSITION_WORK:
        raise OperationResourceAdmissionError(
            location=("homology_map",),
            code="simplicial_set.homology_map_composition_work_exceeded",
            message="homology-coordinate composition exceeds its admitted work bound",
        )
    # Each output is a sum of at most the middle rank many products. This
    # integer-only upper bound is checked before any coordinate multiplication.
    maximum_output_digits = (
        (2 * maximum_input_bits * 30_103 + 99_999) // 100_000
        + (maximum_middle_rank.bit_length() * 30_103 + 99_999) // 100_000
        + 1
    )
    if maximum_output_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("homology_map",),
            code="simplicial_set.homology_map_composition_digits_exceeded",
            message="homology-coordinate products may exceed the canonical integer digit limit",
        )
    degree_maps: list[NormalizedHomologyDegreeMap] = []
    for degree, first_degree in enumerate(first.degree_maps):
        target_group = second.target.homology_groups[degree]
        assert isinstance(target_group, IntegralHomologyGroupValue)
        second_degree = second.degree_maps[degree]
        degree_maps.append(
            NormalizedHomologyDegreeMap(
                degree=degree,
                free_generator_images=tuple(
                    _compose_homology_coordinates(image, second_degree, target_group)
                    for image in first_degree.free_generator_images
                ),
                torsion_generator_images=tuple(
                    _compose_homology_coordinates(image, second_degree, target_group)
                    for image in first_degree.torsion_generator_images
                ),
            )
        )
    return SimplicialHomologyMapValue(
        simplicial_map=composite_map,
        source=first.source,
        target=second.target,
        degree_maps=tuple(degree_maps),
    )


def normalized_homology(
    simplicial_set: FiniteTruncatedSimplicialSet,
    *,
    _integral_right_inverses: list[list[list[int]]] | None = None,
) -> NormalizedHomologyResult:
    """Compute integral normalized homology below the finite prefix top.

    The complete prefix includes d_N, so H_0 through H_(N-1) are determined.
    H_N is intentionally omitted because X_(N+1), and therefore d_(N+1), is
    absent. The standard exact integral homology kernel supplies torsion and
    free-cycle representatives in the normalized simplex axes.
    """
    normalized = normalized_chains(simplicial_set)
    source = normalized.simplicial_set
    sizes = normalized.chain_complex.basis_sizes
    if any(size > 64 for size in sizes):
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_homology_basis_budget_exceeded",
            message="normalized homology basis exceeds the chain-complex basis bound",
        )
    cells = sum(sizes[n - 1] * sizes[n] for n in range(1, len(sizes)))
    if cells > 4096:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_homology_matrix_budget_exceeded",
            message="normalized homology matrices exceed the 4096-cell bound",
        )
    chain = normalized.chain_complex
    # This shared exact kernel computes all group data, including the formal
    # top group of the retained chain prefix. Only degrees with a known incoming
    # simplicial boundary are exposed by this operation.
    computed = homology_groups(chain, _integral_right_inverses=_integral_right_inverses)
    nondegenerate_bases: list[tuple[str, ...]] = []
    nondegenerate_bases.extend(normalized.nondegenerate_bases)
    return NormalizedHomologyResult(
        simplicial_set=source,
        nondegenerate_bases=tuple(nondegenerate_bases),
        chain_complex=chain,
        homology_groups=computed.homology_groups[: source.max_degree],
    )


__all__ = [
    "IntegralHomologyCoordinates",
    "NormalizedChainsRequest",
    "NormalizedChainsResult",
    "NormalizedHomologyDegreeMap",
    "NormalizedHomologyResult",
    "SimplicialHomologyMapValue",
    "SimplicialMapRequest",
    "SimplicialMapResult",
    "TruncatedSimplicialMap",
    "compose_simplicial_homology_maps",
    "compose_simplicial_maps",
    "identity_simplicial_map",
    "induced_normalized_homology_map",
    "normalized_chains",
    "normalized_homology",
    "simplicial_map",
]
