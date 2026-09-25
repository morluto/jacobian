"""Finite simplicial maps and normalized-chain prefixes."""

from __future__ import annotations

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.operations import (
    chain_map_commutes,
    homology_groups,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_CHAIN_MAP_CELLS,
    MAX_OPERATION_MATRIX_CELLS,
    ChainComplexValue,
    ChainMapValue,
    CoefficientRing,
    HomologyGroup,
    IntegralHomologyGroupValue,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_NORMALIZED_CHAIN_OUTPUT_CELLS = 256_000
MAX_INDUCED_CHAIN_MAP_OUTPUT_CELLS = 600_000
MAX_INDUCED_CHAIN_MAP_WORK_UNITS = 1_000_000
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
    source, target = map_value.source, map_value.target
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
    normalized_target = normalized_chains(target)
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


def normalized_homology(
    simplicial_set: FiniteTruncatedSimplicialSet,
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
    computed = homology_groups(chain)
    nondegenerate_bases: list[tuple[str, ...]] = []
    nondegenerate_bases.extend(normalized.nondegenerate_bases)
    return NormalizedHomologyResult(
        simplicial_set=source,
        nondegenerate_bases=tuple(nondegenerate_bases),
        chain_complex=chain,
        homology_groups=computed.homology_groups[: source.max_degree],
    )


__all__ = [
    "NormalizedChainsRequest",
    "NormalizedChainsResult",
    "NormalizedHomologyResult",
    "SimplicialMapRequest",
    "SimplicialMapResult",
    "TruncatedSimplicialMap",
    "identity_simplicial_map",
    "normalized_chains",
    "normalized_homology",
    "simplicial_map",
]
