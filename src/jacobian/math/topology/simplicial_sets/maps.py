"""Finite simplicial maps and normalized-chain prefixes."""

from __future__ import annotations

import json

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    MAX_OPERATION_MATRIX_CELLS,
    ChainComplexValue,
    CoefficientRing,
    HomologyGroup,
    IntegralHomologyGroupValue,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_NORMALIZED_CHAIN_OUTPUT_BYTES = 256_000
_NORMALIZED_CHAIN_OUTPUT_OVERHEAD = 4_096


def _normalized_output_byte_bound(
    simplicial_set: FiniteTruncatedSimplicialSet, matrix_cells: int
) -> int:
    """Bound retained source, repeated label axes, and dense integer matrices."""
    source_bytes = len(simplicial_set.model_dump_json().encode("utf-8"))
    # ensure_ascii bounds Unicode label expansion in the repeated basis axes.
    basis_bytes = len(
        json.dumps(
            simplicial_set.sets, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    )
    # Boundary coefficients have magnitude at most N+1, with N<=4. Four
    # characters per matrix cell cover signed entries and their comma.
    return (
        _NORMALIZED_CHAIN_OUTPUT_OVERHEAD
        + source_bytes
        + basis_bytes
        + 4 * matrix_cells
    )


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
    def require_degreewise_axes(self):
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


def compose_simplicial_maps(
    request: SimplicialMapCompositionRequest,
) -> TruncatedSimplicialMap:
    """Compose finite-prefix simplicial maps after checking their claims."""
    first, second = request.first, request.second
    if first.target != second.source:
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
        source=first.source, target=second.target, maps=maps
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
    def require_source_bound_chain_axes(self):
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
    def require_supported_degrees(self):
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
    output_bound = _normalized_output_byte_bound(simplicial_set, cells)
    if output_bound > MAX_NORMALIZED_CHAIN_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.normalized_chain_output_budget_exceeded",
            message=(
                f"estimated normalized chain result size {output_bound} bytes "
                f"exceeds the {MAX_NORMALIZED_CHAIN_OUTPUT_BYTES}-byte output bound"
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
        tuple(s.sets[degree][index] for index in row)
        for degree, row in enumerate(nd)
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
