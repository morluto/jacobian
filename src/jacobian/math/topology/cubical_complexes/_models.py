"""Typed wire contracts for cubical complex operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_LEVELS,
    FilteredChainComplexRequest,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

MAX_DIM = 10
MAX_CELLS = 5000
MAX_FACE_CELLS = 3**MAX_DIM
"""Enough distinct faces for a full cube at every supported ambient dimension."""

MAX_CUBICAL_CHAIN_GROUP = 64
MAX_CUBICAL_CHAIN_CELLS = 16384
MAX_CUBICAL_PRODUCT_RESULT_BYTES = 8 * 1024 * 1024
MAX_CUBICAL_BITMAP_SIDE = 256
MAX_CUBICAL_BITMAP_PIXELS = MAX_CUBICAL_BITMAP_SIDE**2
MAX_CUBICAL_BITMAP_RESULT_BYTES = 8 * 1024 * 1024
MAX_CUBICAL_GRAPH_VERTICES = 1024
MAX_CUBICAL_GRAPH_EDGES = 65_536
MAX_CUBICAL_GRAPH_WORK = 2_000_000
MAX_CUBICAL_GRAPH_RESULT_BYTES = 10 * 1024 * 1024
MAX_CUBICAL_GRAPH_COORDINATE_DIGITS = 1024
MAX_CUBICAL_FACE_POSET_CANDIDATES = MAX_POSET_ELEMENTS * 3**3
MAX_CUBICAL_FACE_POSET_COVER_CANDIDATES = MAX_POSET_ELEMENTS * 6
MAX_CUBICAL_FACE_POSET_COORDINATE_DIGITS = 1024
MAX_CUBICAL_FACE_POSET_RESULT_BYTES = 8 * 1024 * 1024
MAX_CUBICAL_CLOSED_STAR_RESULT_BYTES = 8 * 1024 * 1024
MAX_CUBICAL_CLOSED_STAR_COORDINATE_DIGITS = 64
MAX_CUBICAL_CLOSED_STAR_WORK = 2_000_000
MAX_CUBICAL_BOUNDARY_SUBCOMPLEX_WORK = 8_000_000
MAX_CUBICAL_BOUNDARY_SUBCOMPLEX_RESULT_BYTES = 8 * 1024 * 1024
MAX_CUBICAL_BOUNDARY_SUBCOMPLEX_COORDINATE_DIGITS = 64
MAX_CUBICAL_PRIME = 1000003
MAX_TRIANGULATION_POINTS = 4096
MAX_TRIANGULATION_SIMPLICES = 16384
MAX_TRIANGULATION_CELL_SIMPLICES = 720
MAX_LOWER_STAR_CELLS = 256
MAX_LOWER_STAR_VERTICES = 256
MAX_LOWER_STAR_INCIDENCES = 1024
MAX_LOWER_STAR_COORDINATE_DIGITS = 64
MAX_LOWER_STAR_VALUE_DIGITS = 128
MAX_LOWER_STAR_RESULT_BYTES = 8 * 1024 * 1024
MAX_LOWER_STAR_FILTER_VECTOR_ENTRIES = 131072


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by cubical-complex contracts."""

    return PydanticCustomError(f"cubical_complex.{reason}", message)


class CubicalCell(StrictModel):
    """An elementary cube: a tuple of intervals [a_i, b_i] on integer lattice."""

    intervals: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid_intervals(self) -> Self:
        for a, b in self.intervals:
            if a > b:
                raise _validation_error(
                    "interval_order",
                    "each interval must have a <= b (interval is [a, b])",
                )
            if b - a > 1:
                raise _validation_error(
                    "interval_length",
                    "each interval must have length 0 or 1 (b <= a + 1)",
                )
        return self

    @property
    def dimension(self) -> int:
        return sum(1 for a, b in self.intervals if b > a)


class CubicalComplexRequest(StrictModel):
    """A finite cubical complex: a set of elementary cubes."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class CubicalComplex(StrictModel):
    """Canonical cubical complex with an explicit ambient coordinate axis.

    ``cells`` is a sorted family of distinct cells.  Operations establish that
    it is face closed before constructing this value; the empty family is the
    void subcomplex and retains the declared ambient dimension. Decoding checks
    only the bounded cell and axis representation.
    """

    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    cells: tuple[CubicalCell, ...] = Field(max_length=MAX_FACE_CELLS)

    @model_validator(mode="after")
    def require_structural_cells(self) -> Self:
        if any(len(cell.intervals) != self.ambient_dimension for cell in self.cells):
            raise _validation_error(
                "ambient_dimension_mismatch",
                "every cell must use the declared ambient coordinate axis",
            )
        if tuple(sorted(self.cells, key=lambda cell: cell.intervals)) != self.cells:
            raise _validation_error(
                "cells_not_canonical", "cells must be sorted canonically"
            )
        if len(set(self.cells)) != len(self.cells):
            raise _validation_error("duplicate_cells", "cells must be distinct")
        return self


class CubicalBoundarySubcomplexResult(StrictModel):
    """A pure cubical complex and its source-bound exposed-facet subcomplex.

    The boundary may be empty; its ambient dimension remains the same as the
    source complex so the void subcomplex has an unambiguous cubical context.
    """

    complex: CubicalComplex
    boundary: CubicalComplex
    exposed_facets: tuple[CubicalCell, ...] = Field(max_length=MAX_FACE_CELLS)

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        source_cells = set(self.complex.cells)
        boundary_cells = set(self.boundary.cells)
        if (
            self.boundary.ambient_dimension != self.complex.ambient_dimension
            or not boundary_cells.issubset(source_cells)
            or not set(self.exposed_facets).issubset(boundary_cells)
            or tuple(
                sorted(self.exposed_facets, key=lambda cell: cell.intervals)
            )
            != self.exposed_facets
            or len(set(self.exposed_facets)) != len(self.exposed_facets)
        ):
            raise _validation_error(
                "boundary_subcomplex_source_binding",
                "the boundary and exposed facets must be bound to the source complex",
            )
        return self


class CubicalCellPosetElement(StrictModel):
    """One canonical face-poset label bound to its cubical cell and dimension."""

    element: ElementLabel
    cell: CubicalCell
    dimension: StrictInt = Field(ge=0, le=MAX_DIM)


class CubicalFacePosetResult(StrictModel):
    """The finite poset of cubical cells with an exact label-to-cell axis."""

    complex: CubicalComplex
    poset: FinitePoset
    cell_elements: tuple[CubicalCellPosetElement, ...] = Field(
        min_length=1, max_length=MAX_POSET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_cell_axis_binding(self) -> Self:
        if (
            tuple(entry.cell for entry in self.cell_elements) != self.complex.cells
            or tuple(entry.element for entry in self.cell_elements)
            != self.poset.elements
            or any(
                entry.dimension != entry.cell.dimension for entry in self.cell_elements
            )
        ):
            raise _validation_error(
                "face_poset_axis_binding",
                "face-poset labels and dimensions must bind the canonical cell axis",
            )
        return self


class FVector(StrictModel):
    """An f-vector whose entries are indexed by explicit cell dimension."""

    dimension_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)
    counts: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        if self.dimension_axis != tuple(range(len(self.dimension_axis))):
            raise _validation_error(
                "dimension_axis_not_canonical",
                "dimension axis must enumerate dimensions from zero",
            )
        if len(self.counts) != len(self.dimension_axis) or any(
            count < 0 for count in self.counts
        ):
            raise _validation_error(
                "f_vector_shape", "f-vector counts must match its dimension axis"
            )
        return self


class FVectorResult(StrictModel):
    """The f-vector and Euler characteristic bound to a cubical complex."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    f_vector: FVector
    euler_characteristic: int


class FaceClosureRequest(StrictModel):
    """Compute the full face closure of a set of cells."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class FaceClosureResult(StrictModel):
    """A face-closed complex and its dimensional cell-count summary."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    original_cells: int
    total_cells: int
    cells_by_dimension: FVector


class CubicalClosedStarRequest(StrictModel):
    """A cubical complex generator family and a cell whose closed star is requested."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    cell: CubicalCell


class CubicalClosedStarResult(StrictModel):
    """A source-bound cubical cell and its closed star subcomplex."""

    complex: CubicalComplex
    cell: CubicalCell
    closed_star: CubicalComplex

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        source_cells = set(self.complex.cells)
        if (
            len(self.cell.intervals) != self.complex.ambient_dimension
            or self.closed_star.ambient_dimension != self.complex.ambient_dimension
            or self.cell not in source_cells
            or self.cell not in self.closed_star.cells
            or any(cell not in source_cells for cell in self.closed_star.cells)
        ):
            raise _validation_error(
                "closed_star_source_binding",
                "the selected cell and closed star must be bound to the source complex",
            )
        return self


class CubicalOneSkeletonResult(StrictModel):
    """The cubical complex's one-skeleton with its indexed vertex-cell map."""

    complex: CubicalComplex
    graph: IndexedSimpleUndirectedGraph
    vertex_cells: tuple[CubicalCell, ...] = Field(max_length=MAX_CUBICAL_GRAPH_VERTICES)

    @model_validator(mode="after")
    def require_vertex_axis_map(self) -> Self:
        if len(self.vertex_cells) != self.graph.vertex_count:
            raise _validation_error(
                "one_skeleton_vertex_map_shape",
                "vertex cells must align with every indexed graph vertex",
            )
        if self.vertex_cells != tuple(
            sorted(set(self.vertex_cells), key=lambda cell: cell.intervals)
        ):
            raise _validation_error(
                "one_skeleton_vertex_map_order",
                "vertex cells must be unique and sorted by cubical coordinates",
            )
        source_cells = set(self.complex.cells)
        if any(
            cell.dimension != 0 or cell not in source_cells
            for cell in self.vertex_cells
        ):
            raise _validation_error(
                "one_skeleton_vertex_map_source",
                "every indexed vertex must map to a source zero-cell",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        complex_: CubicalComplex,
        graph: IndexedSimpleUndirectedGraph,
        vertex_cells: tuple[CubicalCell, ...],
    ) -> Self:
        """Build the admitted projection without replaying its cell relations."""

        return cls.model_construct(
            complex=complex_, graph=graph, vertex_cells=vertex_cells
        )


class CubicalSkeletonRequest(StrictModel):
    """A finite family of elementary cubes and the requested dimension bound."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    dimension_bound: int = Field(ge=0, le=MAX_DIM)


class CubicalSkeletonResult(StrictModel):
    """The canonical face-closed source and its k-dimensional skeleton."""

    complex: CubicalComplex
    skeleton: CubicalComplex
    dimension_bound: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_source_bound_skeleton(self) -> Self:
        if self.skeleton.ambient_dimension != self.complex.ambient_dimension:
            raise _validation_error(
                "skeleton_ambient_dimension_mismatch",
                "the skeleton and source must use the same ambient axes",
            )
        if any(
            cell not in self.complex.cells or cell.dimension > self.dimension_bound
            for cell in self.skeleton.cells
        ):
            raise _validation_error(
                "skeleton_source_binding_invalid",
                "every retained cell must belong to the source and meet the dimension bound",
            )
        return self


class CubicalVertexFiltrationValue(StrictModel):
    """An exact rational value attached to one integer-lattice vertex."""

    vertex: CubicalCell
    value: CanonicalRational

    @model_validator(mode="after")
    def require_vertex_cell(self) -> Self:
        if self.vertex.dimension != 0:
            raise _validation_error(
                "lower_star_input_not_vertex",
                "lower-star values must be attached to zero-dimensional cells",
            )
        return self


class CubicalLowerStarRequest(StrictModel):
    """Cubical generators and exact filtration values on every source vertex."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    vertex_values: tuple[CubicalVertexFiltrationValue, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_VERTICES
    )
    prime: StrictInt = Field(default=2, ge=2, le=MAX_CUBICAL_PRIME)


class CubicalTopCellFiltrationValue(StrictModel):
    """An exact rational value attached to one inclusion-maximal source cell."""

    cell: CubicalCell
    value: CanonicalRational


class CubicalTopCellFiltrationRequest(StrictModel):
    """A cubical complex with one exact value on each maximal cell."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    top_cell_values: tuple[CubicalTopCellFiltrationValue, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_CELLS
    )
    prime: StrictInt = Field(default=2, ge=2, le=MAX_CUBICAL_PRIME)


class CubicalCellBirth(StrictModel):
    """One cell's lower-star birth and the vertices attaining that maximum."""

    cell: CubicalCell
    value: CanonicalRational
    maximizing_vertices: tuple[CubicalCell, ...] = Field(
        min_length=1, max_length=2**MAX_DIM
    )


class CubicalProductRequest(StrictModel):
    """Two finite elementary-cube families whose Cartesian product is requested."""

    left_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    right_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class CubicalProductResult(StrictModel):
    """The cubical product, with the concatenated axes' factor split retained."""

    complex: CubicalComplex
    left_ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    right_ambient_dimension: int = Field(ge=1, le=MAX_DIM)

    @model_validator(mode="after")
    def require_factor_axis_split(self) -> Self:
        if (
            self.left_ambient_dimension + self.right_ambient_dimension
            != self.complex.ambient_dimension
        ):
            raise _validation_error(
                "product_axis_split_invalid",
                "the product ambient axis must concatenate the two factor axes",
            )
        return self


BitmapRow = Annotated[
    tuple[StrictBool, ...],
    Field(min_length=1, max_length=MAX_CUBICAL_BITMAP_SIDE),
]


class CubicalBitmapRequest(StrictModel):
    """A rectangular binary raster whose true pixels denote closed top squares.

    Rows run from top to bottom and columns from left to right.  The raster
    dimensions are bounded to 256 by 256 before any cubical faces are made.
    """

    pixels: tuple[BitmapRow, ...] = Field(
        min_length=1, max_length=MAX_CUBICAL_BITMAP_SIDE
    )

    @model_validator(mode="after")
    def require_rectangular_bitmap(self) -> Self:
        width = len(self.pixels[0])
        if any(len(row) != width for row in self.pixels):
            raise _validation_error(
                "bitmap_not_rectangular", "binary bitmap rows must have equal width"
            )
        if len(self.pixels) * width > MAX_CUBICAL_BITMAP_PIXELS:
            raise _validation_error(
                "bitmap_pixel_budget",
                f"binary bitmap dimensions exceed {MAX_CUBICAL_BITMAP_PIXELS} pixels",
            )
        return self


class CubicalBitmapPixelCell(StrictModel):
    """The exact image of one foreground ``(row, column)`` pixel as a 2-cell."""

    row: int = Field(ge=0, le=MAX_CUBICAL_BITMAP_SIDE - 1)
    column: int = Field(ge=0, le=MAX_CUBICAL_BITMAP_SIDE - 1)
    cell: CubicalCell

    @model_validator(mode="after")
    def require_pixel_square(self) -> Self:
        if self.cell.intervals != (
            (self.column, self.column + 1),
            (self.row, self.row + 1),
        ):
            raise _validation_error(
                "bitmap_pixel_image_invalid",
                "a pixel map entry must be its ordered closed unit square",
            )
        return self


class CubicalBitmapResult(StrictModel):
    """The cubical complex, raster dimensions, and foreground-pixel image.

    In the returned complex, coordinate axis 0 is column (x) and axis 1 is row
    (y, increasing downward).  A foreground pixel at ``(r, c)`` is the closed
    unit square ``([c,c+1], [r,r+1])``.
    """

    complex: CubicalComplex
    row_count: int = Field(ge=1, le=MAX_CUBICAL_BITMAP_SIDE)
    column_count: int = Field(ge=1, le=MAX_CUBICAL_BITMAP_SIDE)
    pixel_to_cell: tuple[CubicalBitmapPixelCell, ...] = Field(max_length=MAX_CELLS)

    @model_validator(mode="after")
    def require_bitmap_axis_and_bounds(self) -> Self:
        if self.complex.ambient_dimension != 2:
            raise _validation_error(
                "bitmap_ambient_dimension_invalid",
                "binary bitmap complexes must have the ordered (column, row) 2D axis",
            )
        pixel_indices = tuple((entry.row, entry.column) for entry in self.pixel_to_cell)
        if pixel_indices != tuple(sorted(set(pixel_indices))):
            raise _validation_error(
                "bitmap_pixel_map_not_canonical",
                "pixel map entries must be unique and sorted by (row, column)",
            )
        mapped_cells = {entry.cell for entry in self.pixel_to_cell}
        top_cells = {cell for cell in self.complex.cells if cell.dimension == 2}
        if mapped_cells != top_cells:
            raise _validation_error(
                "bitmap_pixel_map_incomplete",
                "pixel map cells must equal the complex's full set of unit squares",
            )
        if any(
            a < 0 or b > bound
            for cell in self.complex.cells
            for (a, b), bound in zip(
                cell.intervals, (self.column_count, self.row_count), strict=True
            )
        ) or any(
            entry.row >= self.row_count
            or entry.column >= self.column_count
            or entry.cell not in self.complex.cells
            for entry in self.pixel_to_cell
        ):
            raise _validation_error(
                "bitmap_cell_out_of_bounds",
                "bitmap cells must lie within the retained row and column dimensions",
            )
        return self


class CubicalChainCoefficient(StrEnum):
    """Exact coefficient rings supported by cubical chain complexes."""

    INTEGER = "ZZ"
    PRIME_FIELD = "GF_p"


class CubicalChainComplexRequest(StrictModel):
    """A finite elementary cubical complex with exact chain coefficients.

    The kernel closes the supplied cells under faces once during admission;
    the request itself need only use one ambient coordinate axis.
    """

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    coefficient_ring: CubicalChainCoefficient = CubicalChainCoefficient.INTEGER
    prime: int | None = Field(default=None, ge=2, le=MAX_CUBICAL_PRIME)


class CubicalCellBasis(StrictModel):
    """The canonically ordered cells spanning one cubical chain group."""

    dimension: int = Field(ge=0, le=MAX_DIM)
    cells: tuple[CubicalCell, ...] = Field(
        min_length=1,
        max_length=MAX_CUBICAL_CHAIN_GROUP,
        description=(
            "Canonically ordered cells of one dimension; the order is the "
            "implicit basis of the dense boundary matrices."
        ),
    )


class FilteredCubicalComplex(StrictModel):
    """Source-bound lower-star values plus their filtered based chain complex."""

    complex: CubicalComplex
    vertex_values: tuple[CubicalVertexFiltrationValue, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_VERTICES
    )
    cell_bases: tuple[CubicalCellBasis, ...] = Field(
        min_length=1, max_length=MAX_DIM + 1
    )
    cell_births: tuple[CubicalCellBirth, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_CELLS
    )
    critical_values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_FILTER_LEVELS
    )
    filtered_chain_complex: FilteredChainComplexRequest

    @model_validator(mode="after")
    def require_filtered_chain_axes(self) -> Self:
        basis_sizes = tuple(len(basis.cells) for basis in self.cell_bases)
        if tuple(basis.dimension for basis in self.cell_bases) != tuple(
            range(len(self.cell_bases))
        ):
            raise _validation_error(
                "basis_axis_invalid",
                "cell bases must cover contiguous degrees starting at zero",
            )
        if any(size > MAX_FILTER_AMBIENT_DIMENSION for size in basis_sizes):
            raise _validation_error(
                "basis_size_invalid",
                "cubical chain bases must fit the filtered-chain ambient bound",
            )
        if any(
            tuple(sorted(basis.cells, key=lambda cell: cell.intervals)) != basis.cells
            or any(cell.dimension != basis.dimension for cell in basis.cells)
            for basis in self.cell_bases
        ):
            raise _validation_error(
                "basis_cells_invalid",
                "each cell basis must be canonical and match its degree",
            )
        if self.filtered_chain_complex.complex.basis_sizes != basis_sizes:
            raise _validation_error(
                "chain_basis_mismatch",
                "filtered chain groups must retain the cubical cell-basis sizes",
            )
        if tuple(entry.cell for entry in self.cell_births) != self.complex.cells:
            raise _validation_error(
                "cell_axis_invalid",
                "cell birth entries must follow the canonical source cell axis",
            )
        if {cell for basis in self.cell_bases for cell in basis.cells} != set(
            self.complex.cells
        ):
            raise _validation_error(
                "cell_basis_incomplete",
                "degree cell bases must partition the source cubical cells",
            )
        source_vertices = tuple(
            cell for cell in self.complex.cells if cell.dimension == 0
        )
        if tuple(entry.vertex for entry in self.vertex_values) != source_vertices:
            raise _validation_error(
                "vertex_axis_invalid",
                "vertex values must follow the complete canonical source vertex axis",
            )
        critical = tuple(value.as_fraction() for value in self.critical_values)
        if critical != tuple(sorted(set(critical))):
            raise _validation_error(
                "critical_values_invalid",
                "critical values must be strictly increasing and duplicate-free",
            )
        if any(
            tuple(
                sorted(
                    entry.maximizing_vertices,
                    key=lambda vertex: vertex.intervals,
                )
            )
            != entry.maximizing_vertices
            or any(
                vertex.dimension != 0 or vertex not in source_vertices
                for vertex in entry.maximizing_vertices
            )
            for entry in self.cell_births
        ):
            raise _validation_error(
                "maximizing_vertices_invalid",
                "cell birth witnesses must be canonical source vertices",
            )
        if self.filtered_chain_complex.complex.prime is None:
            raise _validation_error(
                "chain_field_missing",
                "filtered cubical chains must retain their finite-field modulus",
            )
        return self


class CubicalTopCellBirth(StrictModel):
    """Birth of a cell, witnessed by its earliest maximal source coface."""

    cell: CubicalCell
    value: CanonicalRational
    minimizing_top_cells: tuple[CubicalCell, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_CELLS
    )


class FilteredCubicalComplexFromTopCells(StrictModel):
    """Filtered cubical chains from values on inclusion-maximal source cells."""

    complex: CubicalComplex
    top_cell_values: tuple[CubicalTopCellFiltrationValue, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_CELLS
    )
    cell_bases: tuple[CubicalCellBasis, ...] = Field(
        min_length=1, max_length=MAX_DIM + 1
    )
    cell_births: tuple[CubicalTopCellBirth, ...] = Field(
        min_length=1, max_length=MAX_LOWER_STAR_CELLS
    )
    critical_values: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_FILTER_LEVELS
    )
    filtered_chain_complex: FilteredChainComplexRequest

    @model_validator(mode="after")
    def require_top_cell_axes(self) -> Self:
        if tuple(entry.cell for entry in self.cell_births) != self.complex.cells:
            raise _validation_error(
                "top_cell_birth_axis_invalid",
                "cell births must follow the canonical complex cell axis",
            )
        if any(entry.cell not in self.complex.cells for entry in self.top_cell_values):
            raise _validation_error(
                "top_cell_value_source_invalid",
                "top-cell values must be attached to source-complex cells",
            )
        if any(
            coface not in self.complex.cells
            for birth in self.cell_births
            for coface in birth.minimizing_top_cells
        ):
            raise _validation_error(
                "top_cell_witness_invalid",
                "birth witnesses must belong to the source complex",
            )
        basis_sizes = tuple(len(basis.cells) for basis in self.cell_bases)
        if self.filtered_chain_complex.complex.basis_sizes != basis_sizes:
            raise _validation_error(
                "top_cell_chain_basis_mismatch",
                "filtered chains must retain the cubical degree-basis sizes",
            )
        if self.filtered_chain_complex.complex.prime is None:
            raise _validation_error(
                "top_cell_chain_field_missing",
                "filtered cubical chains must retain their finite-field modulus",
            )
        return self


class CubicalSquareLedgerEntry(StrictModel):
    """One replayed d^2 = 0 product between adjacent cubical degrees."""

    upper_dimension: int = Field(ge=1, le=MAX_DIM)
    product_rows: int = Field(ge=0)
    product_columns: int = Field(ge=0)
    nonzero_entries: Literal[0] = 0


class CubicalChainComplexResult(StrictModel):
    """The based cubical chain complex with its replayed square-zero ledger."""

    complex: CubicalComplex
    coefficient_ring: CubicalChainCoefficient
    prime: int | None = Field(default=None, ge=2, le=MAX_CUBICAL_PRIME)
    cell_bases: tuple[CubicalCellBasis, ...] = Field(min_length=1)
    value: ChainComplexValue
    differential_squared_zero: tuple[CubicalSquareLedgerEntry, ...] = ()

    @model_validator(mode="after")
    def require_structural_chain_contract(self) -> Self:
        dimensions = tuple(basis.dimension for basis in self.cell_bases)
        if dimensions != tuple(range(len(self.cell_bases))):
            raise _validation_error(
                "cell_basis_coverage_invalid",
                "cell bases must cover contiguous dimensions from zero",
            )
        expected_sizes = tuple(len(basis.cells) for basis in self.cell_bases)
        if self.value.basis_sizes != expected_sizes:
            raise _validation_error(
                "chain_basis_binding_invalid",
                "canonical chain basis sizes must match the cell bases",
            )
        if self.value.degree_min != 0:
            raise _validation_error(
                "chain_degree_binding_invalid",
                "cubical chain complexes are concentrated in degrees 0..d",
            )
        expected_ring = (
            "ZZ" if self.coefficient_ring is CubicalChainCoefficient.INTEGER else "GF_p"
        )
        if (
            self.value.coefficient_ring.value != expected_ring
            or self.value.prime != self.prime
        ):
            raise _validation_error(
                "chain_coefficient_binding_invalid",
                "canonical chain coefficients must match the requested ring",
            )
        if tuple(entry.upper_dimension for entry in self.differential_squared_zero) != (
            tuple(range(1, len(self.cell_bases)))
        ):
            raise _validation_error(
                "square_ledger_incomplete",
                "the square-zero ledger must cover every adjacent degree pair",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_CELLS",
    "MAX_CUBICAL_BITMAP_PIXELS",
    "MAX_CUBICAL_BITMAP_RESULT_BYTES",
    "MAX_CUBICAL_BITMAP_SIDE",
    "MAX_CUBICAL_CHAIN_CELLS",
    "MAX_CUBICAL_CHAIN_GROUP",
    "MAX_CUBICAL_CLOSED_STAR_COORDINATE_DIGITS",
    "MAX_CUBICAL_CLOSED_STAR_RESULT_BYTES",
    "MAX_CUBICAL_CLOSED_STAR_WORK",
    "MAX_CUBICAL_FACE_POSET_CANDIDATES",
    "MAX_CUBICAL_FACE_POSET_COORDINATE_DIGITS",
    "MAX_CUBICAL_FACE_POSET_COVER_CANDIDATES",
    "MAX_CUBICAL_FACE_POSET_RESULT_BYTES",
    "MAX_CUBICAL_PRIME",
    "MAX_CUBICAL_PRODUCT_RESULT_BYTES",
    "MAX_DIM",
    "MAX_LOWER_STAR_CELLS",
    "MAX_LOWER_STAR_COORDINATE_DIGITS",
    "MAX_LOWER_STAR_FILTER_VECTOR_ENTRIES",
    "MAX_LOWER_STAR_INCIDENCES",
    "MAX_LOWER_STAR_RESULT_BYTES",
    "MAX_LOWER_STAR_VALUE_DIGITS",
    "MAX_LOWER_STAR_VERTICES",
    "MAX_TRIANGULATION_CELL_SIMPLICES",
    "MAX_TRIANGULATION_POINTS",
    "MAX_TRIANGULATION_SIMPLICES",
    "CubicalBitmapPixelCell",
    "CubicalBitmapRequest",
    "CubicalBitmapResult",
    "CubicalBoundarySubcomplexResult",
    "CubicalCell",
    "CubicalCellBasis",
    "CubicalCellBirth",
    "CubicalCellPosetElement",
    "CubicalChainCoefficient",
    "CubicalChainComplexRequest",
    "CubicalChainComplexResult",
    "CubicalClosedStarRequest",
    "CubicalClosedStarResult",
    "CubicalComplex",
    "CubicalComplexRequest",
    "CubicalFacePosetResult",
    "CubicalLowerStarRequest",
    "CubicalProductRequest",
    "CubicalProductResult",
    "CubicalSkeletonRequest",
    "CubicalSkeletonResult",
    "CubicalSquareLedgerEntry",
    "CubicalTopCellBirth",
    "CubicalTopCellFiltrationRequest",
    "CubicalTopCellFiltrationValue",
    "CubicalVertexFiltrationValue",
    "FVector",
    "FVectorResult",
    "FaceClosureRequest",
    "FaceClosureResult",
    "FilteredCubicalComplex",
    "FilteredCubicalComplexFromTopCells",
]
