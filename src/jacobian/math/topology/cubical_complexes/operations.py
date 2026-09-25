"""Exact cubical complex operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import product as cartesian_product

from pydantic import ValidationError

from jacobian._exact import (
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    IndexedSimpleUndirectedGraph,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_LEVELS,
    FilteredChainComplexRequest,
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.operations import (
    construct_chain_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_OPERATION_MATRIX_CELLS,
    CoefficientRing,
    require_prime_field_admission,
)
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CELLS,
    MAX_CUBICAL_CHAIN_CELLS,
    MAX_CUBICAL_CHAIN_GROUP,
    MAX_CUBICAL_CLOSED_STAR_COORDINATE_DIGITS,
    MAX_CUBICAL_CLOSED_STAR_RESULT_BYTES,
    MAX_CUBICAL_CLOSED_STAR_WORK,
    MAX_CUBICAL_FACE_POSET_CANDIDATES,
    MAX_CUBICAL_FACE_POSET_COORDINATE_DIGITS,
    MAX_CUBICAL_FACE_POSET_COVER_CANDIDATES,
    MAX_CUBICAL_FACE_POSET_RESULT_BYTES,
    MAX_CUBICAL_GRAPH_EDGES,
    MAX_CUBICAL_GRAPH_RESULT_BYTES,
    MAX_CUBICAL_GRAPH_VERTICES,
    MAX_CUBICAL_GRAPH_WORK,
    MAX_CUBICAL_PRODUCT_RESULT_BYTES,
    MAX_CUBICAL_SKELETON_RESULT_BYTES,
    MAX_DIM,
    MAX_FACE_CELLS,
    MAX_LOWER_STAR_CELLS,
    MAX_LOWER_STAR_COORDINATE_DIGITS,
    MAX_LOWER_STAR_FILTER_VECTOR_ENTRIES,
    MAX_LOWER_STAR_INCIDENCES,
    MAX_LOWER_STAR_RESULT_BYTES,
    MAX_LOWER_STAR_VALUE_DIGITS,
    MAX_LOWER_STAR_VERTICES,
    CubicalCell,
    CubicalCellBasis,
    CubicalCellBirth,
    CubicalCellPosetElement,
    CubicalChainCoefficient,
    CubicalChainComplexResult,
    CubicalClosedStarRequest,
    CubicalClosedStarResult,
    CubicalComplex,
    CubicalComplexRequest,
    CubicalFacePosetResult,
    CubicalLowerStarRequest,
    CubicalOneSkeletonResult,
    CubicalProductResult,
    CubicalSkeletonResult,
    CubicalSquareLedgerEntry,
    CubicalTopCellBirth,
    CubicalTopCellFiltrationRequest,
    FaceClosureResult,
    FilteredCubicalComplex,
    FilteredCubicalComplexFromTopCells,
    FVector,
    FVectorResult,
)

_CHAIN_RING = {
    CubicalChainCoefficient.INTEGER: CoefficientRing.INTEGER,
    CubicalChainCoefficient.PRIME_FIELD: CoefficientRing.PRIME_FIELD,
}


def _face_cells(
    cells: tuple[CubicalCell, ...], *, output_limit: int = MAX_FACE_CELLS
) -> tuple[CubicalCell, ...]:
    """Materialize the canonical face closure once during operation admission."""
    all_cells: set[tuple[tuple[int, int], ...]] = set()

    def add_faces(intervals: tuple[tuple[int, int], ...]) -> None:
        if intervals in all_cells:
            return
        if len(all_cells) >= output_limit:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.face_output_budget",
                message=(
                    f"cubical face closure exceeds the {output_limit}-cell output bound"
                ),
            )
        all_cells.add(intervals)
        for i, (a, b) in enumerate(intervals):
            if b > a:
                for endpoint in (a, b):
                    face = list(intervals)
                    face[i] = (endpoint, endpoint)
                    add_faces(tuple(face))

    for cell in cells:
        add_faces(cell.intervals)
    return tuple(CubicalCell(intervals=intervals) for intervals in sorted(all_cells))


def _canonical_complex(
    cells: tuple[CubicalCell, ...],
    *,
    face_output_limit: int = MAX_FACE_CELLS,
) -> tuple[CubicalComplex, tuple[CubicalCell, ...]]:
    if not cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message="at least one cell is required",
        )
    ambient_dimension = len(cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message="all cells must use one ambient coordinate axis",
        )
    if len(cells) > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.source_cell_budget",
            message="cubical source exceeds the cell input bound",
        )
    source_cells = tuple(sorted(set(cells), key=lambda cell: cell.intervals))
    closed_cells = _face_cells(source_cells, output_limit=face_output_limit)
    return (
        CubicalComplex(
            ambient_dimension=ambient_dimension,
            cells=closed_cells,
        ),
        source_cells,
    )


def _admit_face_poset_source(
    cells: tuple[CubicalCell, ...],
) -> tuple[tuple[CubicalCell, ...], int, int]:
    if len(cells) > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.source_cell_budget",
            message=f"source exceeds the {MAX_CELLS}-cell input limit",
        )
    if not cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.face_poset.empty_source",
            message="the cubical complex representation requires at least one cell",
        )
    ambient_dimension = len(cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.face_poset.ambient_axis",
            message="all cells must use the same ambient coordinate axis",
        )
    source_cells = tuple(sorted(set(cells), key=lambda cell: cell.intervals))
    if len(source_cells) > MAX_POSET_ELEMENTS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.element_budget",
            message=(
                "the source already has more cells than the "
                f"{MAX_POSET_ELEMENTS}-element finite-poset carrier"
            ),
        )
    if any(cell.dimension > 3 for cell in source_cells):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.single_cell_closure_budget",
            message=(
                "a dimension-four cell alone has 81 distinct faces, above the "
                f"{MAX_POSET_ELEMENTS}-element finite-poset carrier"
            ),
        )
    face_candidate_count = sum(3**cell.dimension for cell in source_cells)
    if face_candidate_count > MAX_CUBICAL_FACE_POSET_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.candidate_budget",
            message=(
                "candidate faces exceed the admitted "
                f"{MAX_CUBICAL_FACE_POSET_CANDIDATES}-candidate work bound"
            ),
        )

    maximum_coordinate_digits = 1
    for cell in source_cells:
        for endpoint in (value for interval in cell.intervals for value in interval):
            digit_bound = max(
                1, (abs(endpoint).bit_length() * 30103 + 99_999) // 100_000
            )
            maximum_coordinate_digits = max(maximum_coordinate_digits, digit_bound)
    if maximum_coordinate_digits > MAX_CUBICAL_FACE_POSET_COORDINATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.coordinate_digit_budget",
            message=(
                "coordinates exceed the "
                f"{MAX_CUBICAL_FACE_POSET_COORDINATE_DIGITS}-digit result limit"
            ),
        )
    return source_cells, ambient_dimension, maximum_coordinate_digits


def _admit_face_poset_result_bytes(
    ambient_dimension: int, maximum_coordinate_digits: int
) -> None:
    relation_pair_bound = MAX_POSET_ELEMENTS * (MAX_POSET_ELEMENTS - 1) // 2
    estimated_result_bytes = (
        4 * MAX_POSET_ELEMENTS * ambient_dimension * (maximum_coordinate_digits + 2)
        + 3 * relation_pair_bound * 128
        + 512 * MAX_POSET_ELEMENTS
        + 32_768
    )
    if estimated_result_bytes > MAX_CUBICAL_FACE_POSET_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.result_byte_budget",
            message=(
                "estimated face-poset encoding exceeds the "
                f"{MAX_CUBICAL_FACE_POSET_RESULT_BYTES}-byte result bound"
            ),
        )


def _face_poset_cover_pairs(
    complex_: CubicalComplex,
    label_by_cell: dict[CubicalCell, str],
) -> set[tuple[str, str]]:
    cell_intervals = {cell.intervals for cell in complex_.cells}
    cover_pairs: set[tuple[str, str]] = set()
    for upper in complex_.cells:
        upper_label = label_by_cell[upper]
        for axis, (start, end) in enumerate(upper.intervals):
            if start == end:
                continue
            for endpoint in (start, end):
                intervals = list(upper.intervals)
                intervals[axis] = (endpoint, endpoint)
                face_intervals = tuple(intervals)
                if face_intervals not in cell_intervals:
                    raise OperationDomainValidationError(
                        location=("cells",),
                        code="cubical_complex.face_poset.closure_incomplete",
                        message="canonical cubical face closure omitted a codimension-one face",
                    )
                face = CubicalCell(intervals=face_intervals)
                cover_pairs.add((label_by_cell[face], upper_label))
    if len(cover_pairs) > MAX_CUBICAL_FACE_POSET_COVER_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.face_poset.cover_budget",
            message="cubical face-poset covers exceed the admitted output bound",
        )
    return cover_pairs


def face_poset(request: CubicalComplexRequest) -> CubicalFacePosetResult:
    """Return the inclusion poset of all cells in the face-closed complex."""
    source_cells, ambient_dimension, maximum_coordinate_digits = (
        _admit_face_poset_source(request.cells)
    )
    _admit_face_poset_result_bytes(ambient_dimension, maximum_coordinate_digits)
    complex_, _ = _canonical_complex(source_cells, face_output_limit=MAX_POSET_ELEMENTS)
    elements = tuple(f"c{index:02d}" for index in range(len(complex_.cells)))
    label_by_cell = dict(zip(complex_.cells, elements, strict=True))
    cover_pairs = _face_poset_cover_pairs(complex_, label_by_cell)
    poset = materialize_finite_poset(
        elements,
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in sorted(cover_pairs)
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    cell_elements = tuple(
        CubicalCellPosetElement(
            element=elements[index], cell=cell, dimension=cell.dimension
        )
        for index, cell in enumerate(complex_.cells)
    )
    return CubicalFacePosetResult(
        complex=complex_, poset=poset, cell_elements=cell_elements
    )


def _counts(complex_: CubicalComplex) -> FVector:
    by_dimension = [0] * (complex_.ambient_dimension + 1)
    for cell in complex_.cells:
        by_dimension[cell.dimension] += 1
    return FVector(
        dimension_axis=tuple(range(complex_.ambient_dimension + 1)),
        counts=tuple(by_dimension),
    )


def f_vector(cells: tuple[CubicalCell, ...]) -> FVectorResult:
    """Compute the f-vector and Euler characteristic of a cubical complex.

    The f-vector counts all faces (including the supplied maximal cells) by
    dimension.  A single square [0,1]x[0,1] has 4 vertices, 4 edges, 1 square,
    so its f-vector is (4, 4, 1).
    """
    complex_, source_cells = _canonical_complex(cells)
    vector = _counts(complex_)
    euler = sum((-1) ** d * count for d, count in enumerate(vector.counts))
    return FVectorResult(
        complex=complex_,
        source_cells=source_cells,
        f_vector=vector,
        euler_characteristic=euler,
    )


def face_closure(cells: tuple[CubicalCell, ...]) -> FaceClosureResult:
    """Compute the full face closure of a set of cells."""
    complex_, source_cells = _canonical_complex(cells)
    cells_by_dimension = _counts(complex_)

    return FaceClosureResult(
        complex=complex_,
        source_cells=source_cells,
        original_cells=len(source_cells),
        total_cells=len(complex_.cells),
        cells_by_dimension=cells_by_dimension,
    )


def _is_face_of(face: CubicalCell, coface: CubicalCell) -> bool:
    return len(face.intervals) == len(coface.intervals) and all(
        outer_lower <= inner_lower and inner_upper <= outer_upper
        for (inner_lower, inner_upper), (outer_lower, outer_upper) in zip(
            face.intervals, coface.intervals, strict=True
        )
    )


def _common_coface_hull(
    first: CubicalCell, second: CubicalCell
) -> tuple[tuple[int, int], ...] | None:
    hull = tuple(
        (min(left[0], right[0]), max(left[1], right[1]))
        for left, right in zip(first.intervals, second.intervals, strict=True)
    )
    if any(upper - lower > 1 for lower, upper in hull):
        return None
    return hull


def closed_star(request: CubicalClosedStarRequest) -> CubicalClosedStarResult:
    """Return all faces of all source cofaces containing ``request.cell``.

    A candidate source cell belongs to the closed star exactly when the
    coordinatewise hull of it and the selected cell is itself a face-closed
    source cell. This avoids a quadratic coface-by-face expansion.
    """
    if type(request) is not CubicalClosedStarRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="cubical_complex.closed_star_request_type",
            message="closed_star requires a canonical CubicalClosedStarRequest",
        )
    cells = request.cells
    selected = request.cell
    if not cells or len(cells) > MAX_CELLS:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.closed_star_source_shape",
            message="closed star requires a nonempty bounded generator family",
        )
    ambient_dimension = len(cells[0].intervals)
    if len(selected.intervals) != ambient_dimension or any(
        len(cell.intervals) != ambient_dimension for cell in cells
    ):
        raise OperationDomainValidationError(
            location=("cell",),
            code="cubical_complex.closed_star_axis_mismatch",
            message="the selected cell and all generators must use the same ambient axes",
        )
    coordinate_digits = max(
        _coordinate_digit_count(coordinate)
        for cell in (*cells, selected)
        for interval in cell.intervals
        for coordinate in interval
    )
    if coordinate_digits > MAX_CUBICAL_CLOSED_STAR_COORDINATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.closed_star_coordinate_bound",
            message="closed-star coordinates exceed the admitted decimal digit bound",
        )

    source_cells = tuple(sorted(set(cells), key=lambda cell: cell.intervals))
    if not any(_is_face_of(selected, coface) for coface in source_cells):
        raise OperationDomainValidationError(
            location=("cell",),
            code="cubical_complex.closed_star_cell_absent",
            message="the selected cell does not occur in the generated face closure",
        )

    face_count_upper = sum(3**cell.dimension for cell in source_cells)
    closure_work_bound = ambient_dimension * face_count_upper
    star_work_bound = ambient_dimension * face_count_upper
    cell_bytes_bound = ambient_dimension * (2 * coordinate_digits + 18) + 32
    output_bytes_bound = 512 + 2 * face_count_upper * cell_bytes_bound
    if (
        face_count_upper > MAX_FACE_CELLS
        or closure_work_bound + star_work_bound > MAX_CUBICAL_CLOSED_STAR_WORK
        or output_bytes_bound > MAX_CUBICAL_CLOSED_STAR_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.closed_star_bounds",
            message=(
                "closed-star face closure, incidence scan, or serialized output "
                "exceeds its admitted bound"
            ),
        )

    complex_, _generators = _canonical_complex(source_cells)
    closed_cell_intervals = {cell.intervals for cell in complex_.cells}
    closed_cells = tuple(
        candidate
        for candidate in complex_.cells
        if (hull := _common_coface_hull(selected, candidate)) is not None
        and hull in closed_cell_intervals
    )
    closed_star_complex = CubicalComplex(
        ambient_dimension=ambient_dimension,
        cells=closed_cells,
    )
    return CubicalClosedStarResult(
        complex=complex_, cell=selected, closed_star=closed_star_complex
    )


def one_skeleton(cells: tuple[CubicalCell, ...]) -> CubicalOneSkeletonResult:
    """Project closed cubical vertices and edges to an indexed simple graph."""

    if type(cells) is not tuple or not cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.one_skeleton.cells_shape",
            message="at least one cubical generator is required",
        )
    if len(cells) > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.source_cell_budget",
            message=f"source has more than the {MAX_CELLS}-cell input limit",
        )
    validated_cells: list[CubicalCell] = []
    try:
        for cell in cells:
            if not isinstance(cell, CubicalCell):
                raise TypeError("every generator must be a cubical cell")
            validated_cells.append(
                CubicalCell.model_validate(cell.model_dump(mode="python"))
            )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.one_skeleton.invalid_cell",
            message="generators must satisfy the canonical cubical-cell contract",
        ) from exc

    source_cells = tuple(sorted(set(validated_cells), key=lambda cell: cell.intervals))
    ambient_dimension = len(source_cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in source_cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.one_skeleton.ambient_axis",
            message="all generators must use one ambient coordinate axis",
        )
    maximum_coordinate_digits = max(
        max(1, (abs(endpoint).bit_length() * 30_103 + 99_999) // 100_000)
        for cell in source_cells
        for interval in cell.intervals
        for endpoint in interval
    )
    face_work_bound = sum(3**cell.dimension for cell in source_cells)
    vertex_count_bound = sum(2**cell.dimension for cell in source_cells)
    edge_count_bound = sum(
        cell.dimension * 2 ** (cell.dimension - 1)
        for cell in source_cells
        if cell.dimension
    )
    if face_work_bound > MAX_CUBICAL_GRAPH_WORK:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.work_bound",
            message=(
                f"face-generation work bound {face_work_bound} exceeds "
                f"{MAX_CUBICAL_GRAPH_WORK}"
            ),
        )
    if vertex_count_bound > MAX_CUBICAL_GRAPH_VERTICES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.vertex_bound",
            message=(
                f"conservative vertex bound {vertex_count_bound} exceeds "
                f"{MAX_CUBICAL_GRAPH_VERTICES}"
            ),
        )
    if edge_count_bound > MAX_CUBICAL_GRAPH_EDGES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.edge_bound",
            message=(
                f"conservative edge bound {edge_count_bound} exceeds "
                f"{MAX_CUBICAL_GRAPH_EDGES}"
            ),
        )
    closure_count_bound = min(MAX_FACE_CELLS, face_work_bound)
    cell_bytes = ambient_dimension * (2 * maximum_coordinate_digits + 8) + 64
    vertex_bytes = ambient_dimension * (2 * maximum_coordinate_digits + 8) + 48
    estimated_output_bytes = (
        closure_count_bound * cell_bytes
        + vertex_count_bound * vertex_bytes
        + edge_count_bound * 32
        + 4_096
    )
    if estimated_output_bytes > MAX_CUBICAL_GRAPH_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.output_bytes",
            message=(
                f"conservative output estimate {estimated_output_bytes} exceeds "
                f"{MAX_CUBICAL_GRAPH_RESULT_BYTES} bytes"
            ),
        )

    complex_, _canonical_sources = _canonical_complex(
        source_cells, face_output_limit=MAX_FACE_CELLS
    )
    vertex_cells = tuple(cell for cell in complex_.cells if cell.dimension == 0)
    vertex_index = {cell: index for index, cell in enumerate(vertex_cells)}
    graph_edges: set[tuple[int, int]] = set()
    for edge in complex_.cells:
        if edge.dimension != 1:
            continue
        varying_axis = next(
            index for index, (left, right) in enumerate(edge.intervals) if right > left
        )
        low = list(edge.intervals)
        high = list(edge.intervals)
        left_endpoint, right_endpoint = edge.intervals[varying_axis]
        low[varying_axis] = (left_endpoint, left_endpoint)
        high[varying_axis] = (right_endpoint, right_endpoint)
        left_vertex = vertex_index[CubicalCell(intervals=tuple(low))]
        right_vertex = vertex_index[CubicalCell(intervals=tuple(high))]
        graph_edges.add(tuple(sorted((left_vertex, right_vertex))))
    if len(vertex_cells) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.vertex_bound",
            message="one-skeleton exceeds the indexed graph vertex limit",
        )
    if len(graph_edges) > MAX_INDEXED_SIMPLE_GRAPH_EDGES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.one_skeleton.edge_bound",
            message="one-skeleton exceeds the indexed graph edge limit",
        )
    graph = IndexedSimpleUndirectedGraph(
        vertex_count=len(vertex_cells), edges=tuple(sorted(graph_edges))
    )
    return CubicalOneSkeletonResult._from_kernel(
        complex_=complex_, graph=graph, vertex_cells=vertex_cells
    )


def skeleton(
    cells: tuple[CubicalCell, ...], dimension_bound: int
) -> CubicalSkeletonResult:
    """Return the cells of dimension at most ``dimension_bound``.

    Input cells are generators: their complete face closure defines the source
    complex. Since every face of a cell of dimension at most k also has
    dimension at most k, filtering that canonical closure produces a subcomplex.
    """
    if type(dimension_bound) is not int or not 0 <= dimension_bound <= MAX_DIM:
        raise OperationDomainValidationError(
            location=("dimension_bound",),
            code="cubical_complex.skeleton_dimension_bound",
            message=f"dimension_bound must be an integer in [0, {MAX_DIM}]",
        )
    if type(cells) is not tuple or not cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.skeleton_cells_shape",
            message="at least one cubical generator is required",
        )
    if len(cells) > MAX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.skeleton_source_cell_budget",
            message=f"source exceeds the {MAX_CELLS}-cell input limit",
        )
    try:
        validated_cells = tuple(
            CubicalCell.model_validate(cell.model_dump(mode="python"))
            for cell in cells
            if isinstance(cell, CubicalCell)
        )
        if len(validated_cells) != len(cells):
            raise TypeError("every generator must be a cubical cell")
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.skeleton_invalid_cell",
            message="generators must satisfy the canonical cubical-cell contract",
        ) from exc
    if any(len(cell.intervals) != len(validated_cells[0].intervals) for cell in validated_cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.skeleton_ambient_axis",
            message="all generators must use one ambient coordinate axis",
        )
    coordinate_digits = max(
        _coordinate_digit_count(coordinate)
        for cell in validated_cells
        for interval in cell.intervals
        for coordinate in interval
    )
    face_bound = sum(3**cell.dimension for cell in set(validated_cells))
    encoded_cell_bound = len(set(validated_cells)) + face_bound
    estimated_bytes = encoded_cell_bound * len(validated_cells[0].intervals) * (2 * coordinate_digits + 8) + 512
    if face_bound > MAX_FACE_CELLS or estimated_bytes > MAX_CUBICAL_SKELETON_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.skeleton_result_size",
            message="skeleton closure and duplicated result exceed the admitted output bound",
        )
    complex_, _source_cells = _canonical_complex(validated_cells)
    retained = tuple(
        cell for cell in complex_.cells if cell.dimension <= dimension_bound
    )
    # Every nonempty cubical complex has vertices, so the 0-skeleton is nonempty.
    skeleton_complex = CubicalComplex(
        ambient_dimension=complex_.ambient_dimension,
        cells=retained,
    )
    return CubicalSkeletonResult(
        complex=complex_,
        skeleton=skeleton_complex,
        dimension_bound=dimension_bound,
    )


def product(
    left_cells: tuple[CubicalCell, ...], right_cells: tuple[CubicalCell, ...]
) -> CubicalProductResult:
    """Construct the finite Cartesian product of two cubical complexes.

    Coordinates of the left factor precede coordinates of the right factor.
    Each factor is first normalized to its full face closure, so the Cartesian
    product of those cells is already face closed.
    """
    if not left_cells or not right_cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.product_empty_factor",
            message="both cubical product factors must contain at least one cell",
        )
    left_dimension = len(left_cells[0].intervals)
    right_dimension = len(right_cells[0].intervals)
    if left_dimension + right_dimension > MAX_DIM:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.product_ambient_dimension",
            message=(
                "the product ambient dimension exceeds the supported "
                f"dimension {MAX_DIM}"
            ),
        )

    left, _ = _canonical_complex(left_cells)
    right, _ = _canonical_complex(right_cells)
    product_cell_count = len(left.cells) * len(right.cells)
    if product_cell_count > MAX_CUBICAL_CHAIN_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.product_cell_budget",
            message=(
                "the cubical product exceeds the "
                f"{MAX_CUBICAL_CHAIN_CELLS}-cell output bound"
            ),
        )

    def coordinate_digit_bound(cell: CubicalCell) -> int:
        return sum(
            (abs(coordinate).bit_length() * 30103) // 100000 + 1 + (coordinate < 0)
            for interval in cell.intervals
            for coordinate in interval
        )

    output_bytes_bound = (
        128
        + product_cell_count * (64 + 8 * (left_dimension + right_dimension))
        + len(right.cells) * sum(coordinate_digit_bound(cell) for cell in left.cells)
        + len(left.cells) * sum(coordinate_digit_bound(cell) for cell in right.cells)
    )
    if output_bytes_bound > MAX_CUBICAL_PRODUCT_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.product_result_size",
            message=(
                "the cubical product exceeds the "
                f"{MAX_CUBICAL_PRODUCT_RESULT_BYTES}-byte result bound"
            ),
        )

    cells = tuple(
        CubicalCell(intervals=left_cell.intervals + right_cell.intervals)
        for left_cell in left.cells
        for right_cell in right.cells
    )
    cells = tuple(sorted(cells, key=lambda cell: cell.intervals))
    return CubicalProductResult(
        complex=CubicalComplex(
            ambient_dimension=left_dimension + right_dimension,
            cells=cells,
        ),
        left_ambient_dimension=left_dimension,
        right_ambient_dimension=right_dimension,
    )


def verify_f_vector(claim: FVectorResult) -> bool:
    """Verify f-vector and Euler claims against retained source cells."""
    try:
        return f_vector(claim.source_cells) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_face_closure(claim: FaceClosureResult) -> bool:
    """Verify the canonical face closure and count summary."""
    try:
        return face_closure(claim.source_cells) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def _cells_by_dimension(
    complex_: CubicalComplex,
) -> tuple[tuple[CubicalCell, ...], ...]:
    """Partition the canonical face-closed cells by dimension.

    Face closure guarantees at least one cell in every dimension from zero
    through the top cell dimension, so the resulting groups form a
    contiguous degree axis.
    """
    top = max(cell.dimension for cell in complex_.cells)
    groups: list[list[CubicalCell]] = [[] for _ in range(top + 1)]
    for cell in complex_.cells:
        groups[cell.dimension].append(cell)
    return tuple(tuple(group) for group in groups)


def _boundary_terms(cell: CubicalCell) -> tuple[tuple[CubicalCell, int], ...]:
    """Return the signed codimension-one face terms of one elementary cube.

    With nondegenerate axes indexed in increasing ambient order, the oriented
    cellular boundary is ``sum_j (-1)^(j-1) (Q_j^+ - Q_j^-)`` where the first
    nondegenerate axis uses the ``+`` sign.  Each returned coefficient is
    relative to the face's own canonical orientation.
    """
    intervals = cell.intervals
    nondegenerate = [axis for axis, (a, b) in enumerate(intervals) if b > a]
    terms: list[tuple[CubicalCell, int]] = []
    for position, axis in enumerate(nondegenerate):
        lower, upper = intervals[axis]
        upper_sign = 1 if position % 2 == 0 else -1
        upper_intervals = list(intervals)
        upper_intervals[axis] = (upper, upper)
        lower_intervals = list(intervals)
        lower_intervals[axis] = (lower, lower)
        terms.append((CubicalCell(intervals=tuple(upper_intervals)), upper_sign))
        terms.append((CubicalCell(intervals=tuple(lower_intervals)), -upper_sign))
    return tuple(terms)


def _boundary_matrices(
    groups: tuple[tuple[CubicalCell, ...], ...],
    *,
    prime: int | None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Assemble the dense cubical boundary matrices in canonical basis order."""
    matrices: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(1, len(groups)):
        source = groups[degree]
        target = groups[degree - 1]
        row_for = {cell: index for index, cell in enumerate(target)}
        dense = [[0] * len(source) for _ in target]
        for column, cell in enumerate(source):
            for face, coefficient in _boundary_terms(cell):
                dense[row_for[face]][column] += coefficient
        matrices.append(
            tuple(
                tuple(value % prime if prime is not None else value for value in row)
                for row in dense
            )
        )
    return tuple(matrices)


def chain_complex(
    cells: tuple[CubicalCell, ...],
    coefficient_ring: CubicalChainCoefficient = CubicalChainCoefficient.INTEGER,
    prime: int | None = None,
) -> CubicalChainComplexResult:
    """Build the exact based cubical chain complex of a face-closed complex.

    The elementary cubes are closed under every cubical face once during
    admission, partitioned into canonical per-dimension bases, and assembled
    into the oriented cubical boundary ``d Q = sum_j (-1)^(j-1) (Q_j^+ -
    Q_j^-)`` over ``ZZ`` or ``GF(p)``.  The differentials are delegated to the
    shared exact based chain-complex kernel, which replays ``d^2 = 0`` before
    the result is returned.
    """
    complex_, _source_cells = _canonical_complex(cells)
    return _chain_complex_from_canonical(complex_, coefficient_ring, prime)


def _chain_complex_from_canonical(
    complex_: CubicalComplex,
    coefficient_ring: CubicalChainCoefficient,
    prime: int | None,
    *,
    groups: tuple[tuple[CubicalCell, ...], ...] | None = None,
    prime_admitted: bool = False,
    chain_bounds_admitted: bool = False,
) -> CubicalChainComplexResult:
    """Build cubical chains from one already-admitted canonical complex."""
    groups = groups or _cells_by_dimension(complex_)
    if not chain_bounds_admitted:
        if any(len(group) > MAX_CUBICAL_CHAIN_GROUP for group in groups):
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.chain_group_budget",
                message=(
                    "a cubical chain group exceeds the "
                    f"{MAX_CUBICAL_CHAIN_GROUP}-cell per-degree bound"
                ),
            )
        aggregate_cells = sum(
            len(groups[degree]) * len(groups[degree + 1])
            for degree in range(len(groups) - 1)
        )
        if aggregate_cells > MAX_OPERATION_MATRIX_CELLS:
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.chain_cell_budget",
                message=(
                    "the cubical boundary matrices exceed the "
                    f"{MAX_OPERATION_MATRIX_CELLS}-cell aggregate bound"
                ),
            )
    ring = _CHAIN_RING[coefficient_ring]
    modulus = prime if ring is CoefficientRing.PRIME_FIELD else None
    if not prime_admitted:
        try:
            require_prime_field_admission(ring, prime)
        except ValueError as exc:
            raise OperationDomainValidationError(
                location=("coefficient_ring", "prime"),
                code="cubical_complex.chain_coefficient_invalid",
                message=str(exc),
            ) from exc
    basis_sizes = tuple(len(group) for group in groups)
    matrices = _boundary_matrices(groups, prime=modulus)
    value = construct_chain_complex(
        basis_sizes,
        matrices,
        coefficient_ring=ring,
        prime=modulus,
    )
    ledger = tuple(
        CubicalSquareLedgerEntry(
            upper_dimension=degree,
            product_rows=len(groups[degree - 2]) if degree >= 2 else 0,
            product_columns=len(groups[degree]),
        )
        for degree in range(1, len(groups))
    )
    return CubicalChainComplexResult._from_kernel(
        complex=complex_,
        coefficient_ring=coefficient_ring,
        prime=prime,
        cell_bases=tuple(
            CubicalCellBasis(dimension=degree, cells=groups[degree])
            for degree in range(len(groups))
        ),
        value=value,
        differential_squared_zero=ledger,
    )


def _coordinate_digit_count(coordinate: int) -> int:
    if abs(coordinate).bit_length() > 4 * MAX_LOWER_STAR_COORDINATE_DIGITS:
        return MAX_LOWER_STAR_COORDINATE_DIGITS + 1
    return len(str(abs(coordinate)))


def _lower_star_vertices(cell: CubicalCell) -> tuple[CubicalCell, ...]:
    choices = tuple(
        (lower,) if lower == upper else (lower, upper)
        for lower, upper in cell.intervals
    )
    return tuple(
        CubicalCell(intervals=tuple((coordinate, coordinate) for coordinate in vertex))
        for vertex in cartesian_product(*choices)
    )


def _admit_lower_star_request(
    request: CubicalLowerStarRequest,
) -> tuple[
    int, tuple[CubicalCell, ...], dict[CubicalCell, Fraction], tuple[Fraction, ...]
]:
    if not isinstance(request, CubicalLowerStarRequest):
        raise OperationDomainValidationError(
            location=(),
            code="cubical_complex.lower_star_request_type_invalid",
            message="lower-star filtration requires a canonical request",
        )
    try:
        require_prime_field_admission(CoefficientRing.PRIME_FIELD, request.prime)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prime",),
            code="cubical_complex.lower_star_prime_invalid",
            message=str(exc),
        ) from exc
    if not request.cells:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.lower_star_empty_source",
            message="lower-star filtration requires at least one source cell",
        )
    ambient_dimension = len(request.cells[0].intervals)
    if any(len(cell.intervals) != ambient_dimension for cell in request.cells):
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.lower_star_ambient_dimension_mismatch",
            message="all source cells must use one ordered ambient coordinate axis",
        )

    for cell in (*request.cells, *(entry.vertex for entry in request.vertex_values)):
        if any(
            _coordinate_digit_count(coordinate) > MAX_LOWER_STAR_COORDINATE_DIGITS
            for interval in cell.intervals
            for coordinate in interval
        ):
            raise OperationResourceAdmissionError(
                location=("cells",),
                code="cubical_complex.lower_star_coordinate_digit_budget",
                message=(
                    "lower-star coordinates are limited to "
                    f"{MAX_LOWER_STAR_COORDINATE_DIGITS} decimal digits"
                ),
            )
    for entry in request.vertex_values:
        if len(entry.vertex.intervals) != ambient_dimension:
            raise OperationDomainValidationError(
                location=("vertex_values",),
                code="cubical_complex.lower_star_vertex_axis_mismatch",
                message="every vertex value must use the source ambient axes",
            )
        try:
            require_bounded_rational(
                entry.value,
                max_digits=MAX_LOWER_STAR_VALUE_DIGITS,
                label="lower-star vertex value",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("vertex_values",),
                code="cubical_complex.lower_star_value_digit_budget",
                message=str(exc),
            ) from exc

    input_vertices = tuple(
        sorted(
            (entry.vertex for entry in request.vertex_values),
            key=lambda vertex: vertex.intervals,
        )
    )
    if len(set(input_vertices)) != len(input_vertices):
        raise OperationDomainValidationError(
            location=("vertex_values",),
            code="cubical_complex.lower_star_duplicate_vertex",
            message="each source vertex must receive exactly one filtration value",
        )
    if len(input_vertices) > MAX_LOWER_STAR_VERTICES:
        raise OperationResourceAdmissionError(
            location=("vertex_values",),
            code="cubical_complex.lower_star_vertex_budget",
            message=(
                f"lower-star input exceeds the {MAX_LOWER_STAR_VERTICES}-vertex bound"
            ),
        )

    value_by_vertex = {
        entry.vertex: entry.value.as_fraction() for entry in request.vertex_values
    }
    critical = tuple(sorted(set(value_by_vertex.values())))
    if len(critical) > MAX_FILTER_LEVELS:
        raise OperationResourceAdmissionError(
            location=("vertex_values",),
            code="cubical_complex.lower_star_level_budget",
            message=(
                "the filtered-chain representation admits at most "
                f"{MAX_FILTER_LEVELS} distinct lower-star values"
            ),
        )
    return ambient_dimension, input_vertices, value_by_vertex, critical


def _admit_lower_star_output(
    complex_: CubicalComplex,
    vertices: tuple[CubicalCell, ...],
    critical: tuple[Fraction, ...],
    ambient_dimension: int,
) -> tuple[tuple[CubicalCell, ...], ...]:
    groups = _cells_by_dimension(complex_)
    if any(len(group) > MAX_FILTER_AMBIENT_DIMENSION for group in groups):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.lower_star_chain_group_budget",
            message=(
                "each filtered cubical chain group is limited to "
                f"{MAX_FILTER_AMBIENT_DIMENSION} cells"
            ),
        )
    matrix_cells = sum(
        len(groups[degree]) * len(groups[degree + 1])
        for degree in range(len(groups) - 1)
    )
    if matrix_cells > MAX_OPERATION_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.lower_star_boundary_matrix_budget",
            message=(
                "lower-star cubical boundary matrices exceed the "
                f"{MAX_OPERATION_MATRIX_CELLS}-cell bound"
            ),
        )

    vertex_cell_incidences = sum(1 << cell.dimension for cell in complex_.cells)
    if vertex_cell_incidences > MAX_LOWER_STAR_INCIDENCES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.lower_star_incidence_budget",
            message=(
                "lower-star cell-to-vertex incidence exceeds the "
                f"{MAX_LOWER_STAR_INCIDENCES}-pair bound"
            ),
        )
    filter_vector_entries = len(critical) * sum(
        len(group) * len(group) for group in groups
    )
    if filter_vector_entries > MAX_LOWER_STAR_FILTER_VECTOR_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.lower_star_filter_vector_budget",
            message=(
                "lower-star filtration vectors exceed the "
                f"{MAX_LOWER_STAR_FILTER_VECTOR_ENTRIES}-entry bound"
            ),
        )
    output_bytes_bound = (
        1024
        + len(complex_.cells)
        * (192 + ambient_dimension * (2 * MAX_LOWER_STAR_COORDINATE_DIGITS + 16))
        + vertex_cell_incidences
        * ambient_dimension
        * (2 * MAX_LOWER_STAR_COORDINATE_DIGITS + 12)
        + len(vertices)
        * ambient_dimension
        * (2 * MAX_LOWER_STAR_COORDINATE_DIGITS + 32)
        + len(complex_.cells) * 2 * MAX_LOWER_STAR_VALUE_DIGITS
        + matrix_cells * 16
        + filter_vector_entries * 4
    )
    if output_bytes_bound > MAX_LOWER_STAR_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.lower_star_result_size",
            message=(
                "lower-star output exceeds the "
                f"{MAX_LOWER_STAR_RESULT_BYTES}-byte result bound"
            ),
        )
    return groups


def _lower_star_births(
    complex_: CubicalComplex,
    value_by_vertex: dict[CubicalCell, Fraction],
) -> tuple[tuple[CubicalCell, Fraction, tuple[CubicalCell, ...]], ...]:
    births = []
    for cell in complex_.cells:
        cell_vertices = _lower_star_vertices(cell)
        value = max(value_by_vertex[vertex] for vertex in cell_vertices)
        maximizers = tuple(
            vertex for vertex in cell_vertices if value_by_vertex[vertex] == value
        )
        births.append((cell, value, maximizers))
    return tuple(births)


def _filtered_lower_star_levels(
    groups: tuple[tuple[CubicalCell, ...], ...],
    birth_by_cell: dict[CubicalCell, Fraction],
    critical: tuple[Fraction, ...],
) -> tuple[FiltrationLevel, ...]:
    return tuple(
        FiltrationLevel(
            subspaces=tuple(
                FilteredSubspace(
                    vectors=tuple(
                        tuple(
                            1 if index == basis_index else 0
                            for index in range(len(group))
                        )
                        for basis_index, cell in enumerate(group)
                        if birth_by_cell[cell] <= level
                    )
                )
                for group in groups
            )
        )
        for level in critical
    )


def lower_star_from_vertices(
    request: CubicalLowerStarRequest,
) -> FilteredCubicalComplex:
    """Build exact vertex lower-star values and filtered cubical chains."""
    ambient_dimension, input_vertices, value_by_vertex, critical = (
        _admit_lower_star_request(request)
    )
    complex_, _ = _canonical_complex(
        request.cells, face_output_limit=MAX_LOWER_STAR_CELLS
    )
    vertices = tuple(cell for cell in complex_.cells if cell.dimension == 0)
    if vertices != input_vertices:
        raise OperationDomainValidationError(
            location=("vertex_values",),
            code="cubical_complex.lower_star_vertex_domain_mismatch",
            message=(
                "vertex values must cover exactly the distinct vertices in the "
                "source face closure"
            ),
        )
    groups = _admit_lower_star_output(complex_, vertices, critical, ambient_dimension)
    births_data = _lower_star_births(complex_, value_by_vertex)
    birth_by_cell = {cell: value for cell, value, _ in births_data}
    chains = _chain_complex_from_canonical(
        complex_,
        CubicalChainCoefficient.PRIME_FIELD,
        request.prime,
        groups=groups,
        prime_admitted=True,
        chain_bounds_admitted=True,
    )
    filtered_chain = FilteredChainComplexRequest(
        complex=chains.value,
        filtration=_filtered_lower_star_levels(groups, birth_by_cell, critical),
    )
    return FilteredCubicalComplex(
        complex=complex_,
        vertex_values=tuple(
            sorted(request.vertex_values, key=lambda entry: entry.vertex.intervals)
        ),
        cell_bases=chains.cell_bases,
        cell_births=tuple(
            CubicalCellBirth(
                cell=cell,
                value=CanonicalRational.from_fraction(value),
                maximizing_vertices=maximizers,
            )
            for cell, value, maximizers in births_data
        ),
        critical_values=tuple(
            CanonicalRational.from_fraction(value) for value in critical
        ),
        filtered_chain_complex=filtered_chain,
    )


def from_top_cell_values(
    request: CubicalTopCellFiltrationRequest,
) -> FilteredCubicalComplexFromTopCells:
    """Build a filtration from exact values on the maximal supplied cells.

    A face is born at the minimum value among the supplied inclusion-maximal
    cells containing it. Thus a sublevel consists exactly of the face closure
    of active top cells and is a cubical subcomplex.
    """
    try:
        require_prime_field_admission(CoefficientRing.PRIME_FIELD, request.prime)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("prime",),
            code="cubical_complex.top_cell_prime_invalid",
            message=str(exc),
        ) from exc
    if any(
        _coordinate_digit_count(coordinate) > MAX_LOWER_STAR_COORDINATE_DIGITS
        for cell in request.cells
        for interval in cell.intervals
        for coordinate in interval
    ):
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="cubical_complex.top_cell_coordinate_digit_budget",
            message=(
                "top-cell filtration coordinates are limited to "
                f"{MAX_LOWER_STAR_COORDINATE_DIGITS} decimal digits"
            ),
        )
    for entry in request.top_cell_values:
        try:
            require_bounded_rational(
                entry.value,
                max_digits=MAX_LOWER_STAR_VALUE_DIGITS,
                label="top-cell filtration value",
            )
        except ValueError as exc:
            raise OperationResourceAdmissionError(
                location=("top_cell_values",),
                code="cubical_complex.top_cell_value_digit_budget",
                message=str(exc),
            ) from exc

    complex_, source = _canonical_complex(
        request.cells, face_output_limit=MAX_LOWER_STAR_CELLS
    )
    maximal = tuple(
        cell
        for cell in source
        if not any(
            candidate != cell
            and candidate.dimension > cell.dimension
            and all(
                outer_start <= inner_start and inner_end <= outer_end
                for (outer_start, outer_end), (inner_start, inner_end) in zip(
                    candidate.intervals, cell.intervals, strict=True
                )
            )
            for candidate in source
        )
    )
    values = tuple(
        sorted(request.top_cell_values, key=lambda entry: entry.cell.intervals)
    )
    if tuple(entry.cell for entry in values) != maximal:
        raise OperationDomainValidationError(
            location=("top_cell_values",),
            code="cubical_complex.top_cell_value_domain_mismatch",
            message="top-cell values must cover exactly the inclusion-maximal supplied cells",
        )
    value_by_cell = {entry.cell: entry.value.as_fraction() for entry in values}
    critical = tuple(sorted(set(value_by_cell.values())))
    if len(critical) > MAX_FILTER_LEVELS:
        raise OperationResourceAdmissionError(
            location=("top_cell_values",),
            code="cubical_complex.top_cell_level_budget",
            message=(
                "the filtered-chain representation admits at most "
                f"{MAX_FILTER_LEVELS} distinct top-cell values"
            ),
        )
    groups = _admit_lower_star_output(
        complex_, (), critical, complex_.ambient_dimension
    )
    birth_data = []
    for cell in complex_.cells:
        containing = tuple(
            top
            for top in maximal
            if all(
                outer_start <= inner_start and inner_end <= outer_end
                for (outer_start, outer_end), (inner_start, inner_end) in zip(
                    top.intervals, cell.intervals, strict=True
                )
            )
        )
        birth = min(value_by_cell[top] for top in containing)
        witnesses = tuple(top for top in containing if value_by_cell[top] == birth)
        birth_data.append((cell, birth, witnesses))
    birth_by_cell = {cell: value for cell, value, _ in birth_data}
    chains = _chain_complex_from_canonical(
        complex_,
        CubicalChainCoefficient.PRIME_FIELD,
        request.prime,
        groups=groups,
        prime_admitted=True,
        chain_bounds_admitted=True,
    )
    filtered_chain = FilteredChainComplexRequest(
        complex=chains.value,
        filtration=_filtered_lower_star_levels(groups, birth_by_cell, critical),
    )
    return FilteredCubicalComplexFromTopCells(
        complex=complex_,
        top_cell_values=values,
        cell_bases=chains.cell_bases,
        cell_births=tuple(
            CubicalTopCellBirth(
                cell=cell,
                value=CanonicalRational.from_fraction(value),
                minimizing_top_cells=witnesses,
            )
            for cell, value, witnesses in birth_data
        ),
        critical_values=tuple(
            CanonicalRational.from_fraction(value) for value in critical
        ),
        filtered_chain_complex=filtered_chain,
    )


__all__ = [
    "chain_complex",
    "f_vector",
    "face_closure",
    "from_top_cell_values",
    "lower_star_from_vertices",
    "skeleton",
    "verify_f_vector",
    "verify_face_closure",
]
