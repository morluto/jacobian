"""Cubical complex operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.topology.chain_complexes._filtered_models import MAX_FILTER_LEVELS
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CUBICAL_CHAIN_CELLS,
    MAX_CUBICAL_CHAIN_PRODUCT_RESULT_BYTES,
    MAX_CUBICAL_CHAIN_PRODUCT_TERMS,
    MAX_CUBICAL_CHAIN_VALUE_COEFFICIENT_DIGITS,
    MAX_CUBICAL_CHAIN_VALUE_COORDINATE_DIGITS,
    MAX_CUBICAL_CHAIN_VALUE_TERMS,
    MAX_CUBICAL_PRODUCT_RESULT_BYTES,
    MAX_DIM,
    MAX_LOWER_STAR_CELLS,
    MAX_LOWER_STAR_VERTICES,
    CubicalChainComplexRequest,
    CubicalChainComplexResult,
    CubicalChainProductRequest,
    CubicalChainValue,
    CubicalClosedStarRequest,
    CubicalClosedStarResult,
    CubicalComplexRequest,
    CubicalFacePosetResult,
    CubicalLowerStarRequest,
    CubicalOneSkeletonResult,
    CubicalProductRequest,
    CubicalProductResult,
    CubicalSkeletonRequest,
    CubicalSkeletonResult,
    CubicalTopCellFiltrationRequest,
    FaceClosureRequest,
    FaceClosureResult,
    FilteredCubicalComplex,
    FilteredCubicalComplexFromTopCells,
    FVectorResult,
)
from jacobian.math.topology.cubical_complexes.extensions_tools import (
    TOOLS as EXTENSION_TOOLS,
)
from jacobian.math.topology.cubical_complexes.operations import (
    chain_complex,
    chain_product,
    closed_star,
    f_vector,
    face_closure,
    face_poset,
    from_top_cell_values,
    lower_star_from_vertices,
    one_skeleton,
    product,
    skeleton,
)


def _f_vector(request: CubicalComplexRequest) -> FVectorResult:
    return f_vector(request.cells)


def _face_closure(request: FaceClosureRequest) -> FaceClosureResult:
    return face_closure(request.cells)


def _closed_star(request: CubicalClosedStarRequest) -> CubicalClosedStarResult:
    return closed_star(request)


def _face_poset(request: CubicalComplexRequest) -> CubicalFacePosetResult:
    return face_poset(request)


def _chain_complex(request: CubicalChainComplexRequest) -> CubicalChainComplexResult:
    return chain_complex(request.cells, request.coefficient_ring, request.prime)


def _chain_product(request: CubicalChainProductRequest) -> CubicalChainValue:
    return chain_product(request)


def _product(request: CubicalProductRequest) -> CubicalProductResult:
    return product(request.left_cells, request.right_cells)


def _skeleton(request: CubicalSkeletonRequest) -> CubicalSkeletonResult:
    return skeleton(request.cells, request.dimension_bound)


def _lower_star(request: CubicalLowerStarRequest) -> FilteredCubicalComplex:
    return lower_star_from_vertices(request)


def _top_cell_filtration(
    request: CubicalTopCellFiltrationRequest,
) -> Any:
    return from_top_cell_values(request)


def _one_skeleton(request: CubicalComplexRequest) -> CubicalOneSkeletonResult:
    return one_skeleton(request.cells)


# A single 2D square: [(0,1),(0,1)] + [(0,1),(1,2)] + [(1,2),(0,1)] + [(1,2),(1,2)]
_CELLS = {
    "cells": [
        {"intervals": [[0, 1], [0, 1]]},
        {"intervals": [[0, 1], [1, 2]]},
        {"intervals": [[1, 2], [0, 1]]},
        {"intervals": [[1, 2], [1, 2]]},
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    *EXTENSION_TOOLS,
    MathTool(
        operation_id="topology.cubical_complex.closed_star.compute",
        title="Compute a cubical cell's closed star",
        description=(
            "Return a face-closed source complex and the union of the closures "
            "of all source cells containing the selected cell. The selected "
            "cell must occur in the generated face closure. Closure work, "
            "coordinate digits, and both serialized complexes are admitted "
            "before expansion."
        ),
        request_type=CubicalClosedStarRequest,
        result_type=CubicalClosedStarResult,
        run=_closed_star,
        tags=("topology", "cubical", "closed-star", "exact"),
        discovery_terms=(
            "closed star of a cubical cell",
            "cofaces of a cell and their faces",
            "cubical neighborhood of a cell",
        ),
        examples=(
            OperationExample(
                name="endpoint_star_in_a_path",
                description=(
                    "The closed star of vertex 0 in the path with edges [0,1] "
                    "and [1,2] is the first edge and its two vertices."
                ),
                input={
                    "cells": [
                        {"intervals": [[0, 1]]},
                        {"intervals": [[1, 2]]},
                    ],
                    "cell": {"intervals": [[0, 0]]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_complex.face_poset.compute",
        title="Compute the finite poset of cubical cells",
        description=(
            "Close a bounded family of elementary integer-lattice cubes under "
            "faces and return the finite poset ordered by cell inclusion. The "
            "result reuses the canonical finite-poset value and binds each "
            "poset label to its exact cubical cell and dimension. Admission "
            "limits the face closure and poset to 64 cells, with bounded face "
            "candidate work, coordinate digits, relations, and result bytes. "
            "The current cubical-complex carrier excludes the void complex."
        ),
        request_type=CubicalComplexRequest,
        result_type=CubicalFacePosetResult,
        run=_face_poset,
        tags=("topology", "cubical", "face-poset", "exact"),
        discovery_terms=(
            "cubical face poset",
            "cubical cells ordered by inclusion",
            "finite poset of elementary cubes",
        ),
        examples=(
            OperationExample(
                name="unit_square_face_poset",
                description=(
                    "Compute the inclusion order of the square and its four edges "
                    "and four vertices."
                ),
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_complex.one_skeleton.compute",
        title="Project a finite cubical complex to its one-skeleton graph",
        description=(
            "Close the supplied lattice cubes under faces, then return the "
            "indexed graph whose vertices are zero-cells and whose edges are "
            "one-cells. The result retains the canonical cubical complex and "
            "the coordinate cell aligned with every graph vertex index. Face "
            "work, graph size, and serialized result size are admitted before "
            "closure and graph construction."
        ),
        request_type=CubicalComplexRequest,
        result_type=CubicalOneSkeletonResult,
        run=_one_skeleton,
        tags=("topology", "cubical", "one-skeleton", "graph", "exact"),
        discovery_terms=(
            "one-skeleton of a cubical complex",
            "cubical complex graph vertices edges",
        ),
        examples=(
            OperationExample(
                name="unit_square_one_skeleton_graph",
                description=(
                    "Project a unit square to its four vertex and four boundary-edge graph."
                ),
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
    MathTool(
        operation_id=(
            "topology.filtered_cubical_complex.lower_star_from_vertices.compute"
        ),
        title="Build a cubical vertex lower-star filtration",
        description=(
            "Close the supplied elementary cubes under faces, require one exact "
            "QQ value on every source vertex, and assign each cell the maximum "
            "of its vertex values. Return exact critical levels and maximizing "
            "vertex provenance together with the source-bound cubical bases and "
            "a composable filtered chain complex over GF(p). The filtered-chain "
            f"envelope is limited to {MAX_LOWER_STAR_CELLS} cells, "
            f"{MAX_LOWER_STAR_VERTICES} vertices, {MAX_FILTER_LEVELS} critical "
            "levels, and 32 basis cells per degree."
        ),
        request_type=CubicalLowerStarRequest,
        result_type=FilteredCubicalComplex,
        run=_lower_star,
        tags=("topology", "cubical", "filtration", "filtered-chain", "exact"),
        discovery_terms=(
            "cubical lower-star filtration",
            "filtered cubical chain complex",
        ),
        examples=(
            OperationExample(
                name="unit_square_vertex_lower_star",
                description=(
                    "Assign exact values to the four vertices of a square; the "
                    "square is born at the largest vertex value."
                ),
                input={
                    "cells": [{"intervals": [[0, 1], [0, 1]]}],
                    "vertex_values": [
                        {
                            "vertex": {"intervals": [[0, 0], [0, 0]]},
                            "value": {"num": "0", "den": "1"},
                        },
                        {
                            "vertex": {"intervals": [[0, 0], [1, 1]]},
                            "value": {"num": "1", "den": "1"},
                        },
                        {
                            "vertex": {"intervals": [[1, 1], [0, 0]]},
                            "value": {"num": "1", "den": "1"},
                        },
                        {
                            "vertex": {"intervals": [[1, 1], [1, 1]]},
                            "value": {"num": "2", "den": "1"},
                        },
                    ],
                    "prime": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.filtered_cubical_complex.from_top_cells.compute",
        title="Build a cubical filtration from top-cell values",
        description=(
            "Close the supplied elementary cubes under faces and require one "
            "exact rational value on every inclusion-maximal supplied cell. "
            "Each face is born at the least value among its maximal source "
            "cofaces, so every sublevel is exactly the face closure of active "
            "top cells. Return the finite-field filtered chain complex and "
            "source-bound minimizing-coface witnesses under the bounded "
            "filtered-chain envelope."
        ),
        request_type=CubicalTopCellFiltrationRequest,
        result_type=FilteredCubicalComplexFromTopCells,
        run=_top_cell_filtration,
        tags=("topology", "cubical", "filtration", "filtered-chain", "exact"),
        discovery_terms=(
            "cubical top-cell sublevel filtration",
            "cubical filtration from maximal cells",
            "filtered cubical chain complex from top-cell values",
        ),
        examples=(
            OperationExample(
                name="two_squares_top_cell_filtration",
                description=(
                    "Assign values to the two maximal squares; their shared "
                    "edge is born with the earlier square."
                ),
                input={
                    "cells": [
                        {"intervals": [[0, 1], [0, 1]]},
                        {"intervals": [[1, 2], [0, 1]]},
                    ],
                    "top_cell_values": [
                        {
                            "cell": {"intervals": [[0, 1], [0, 1]]},
                            "value": {"num": "0", "den": "1"},
                        },
                        {
                            "cell": {"intervals": [[1, 2], [0, 1]]},
                            "value": {"num": "2", "den": "1"},
                        },
                    ],
                    "prime": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_complex.skeleton.compute",
        title="Compute the cubical skeleton",
        description=(
            "Close the supplied elementary lattice cubes under faces and retain "
            "exactly cells of dimension at most the requested bound. The result "
            "contains the canonical source closure and its cubical subcomplex. "
            "Face-closure growth is admitted before construction."
        ),
        request_type=CubicalSkeletonRequest,
        result_type=CubicalSkeletonResult,
        run=_skeleton,
        tags=("topology", "cubical", "skeleton", "exact"),
        examples=(
            OperationExample(
                name="square_one_skeleton",
                description=(
                    "Return the four boundary edges and four vertices of a unit square."
                ),
                input={
                    "cells": [{"intervals": [[0, 1], [0, 1]]}],
                    "dimension_bound": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.product.compute",
        title="Compute a finite cubical product",
        description=(
            "Compute the Cartesian product of two finite cubical complexes. "
            "Each factor is closed under faces, then product cells concatenate "
            "the left factor's integer-lattice coordinates before the right "
            f"factor's. The complete result is bounded to "
            f"{MAX_CUBICAL_CHAIN_CELLS} cells and ambient dimension "
            f"{MAX_DIM}."
            f" Estimated result encoding is bounded to "
            f"{MAX_CUBICAL_PRODUCT_RESULT_BYTES} bytes."
        ),
        request_type=CubicalProductRequest,
        result_type=CubicalProductResult,
        run=_product,
        tags=("topology", "cubical", "product", "exact"),
        examples=(
            OperationExample(
                name="interval_times_point",
                description=(
                    "Form the product of one unit interval with one point; the "
                    "output is an interval on the concatenated coordinate axes."
                ),
                input={
                    "left_cells": [{"intervals": [[0, 1]]}],
                    "right_cells": [{"intervals": [[5, 5]]}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.f_vector.compute",
        title="Compute the f-vector of a cubical complex",
        description="Compute the f-vector (cell counts by dimension) and Euler "
        "characteristic of a finite cubical complex composed of "
        "elementary unit lattice cubes.",
        request_type=CubicalComplexRequest,
        result_type=FVectorResult,
        run=_f_vector,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="four_squares",
                description="Compute the f-vector of four unit squares forming a 2x2 grid; "
                "each interval must be unit length (b = a + 1).",
                input=_CELLS,
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.face_closure.compute",
        title="Compute the face closure of a cubical complex",
        description="Compute the full face closure (all proper faces) of a set "
        "of elementary cubes, returning total cell count and "
        "cells by dimension.",
        request_type=FaceClosureRequest,
        result_type=FaceClosureResult,
        run=_face_closure,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="single_square_closure",
                description="Compute the face closure of a single unit square; "
                "each interval must be unit length (b = a + 1).",
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_complex.chain_complex.compute",
        title="Compute the oriented cubical chain complex",
        description="Close a finite family of elementary integer-lattice cubes "
        "under every cubical face, then construct the exact based chain complex "
        "with oriented cubical boundary "
        "dQ = sum_j (-1)^(j-1) (Q_j^+ - Q_j^-) over ZZ or GF(p). Returns the "
        "canonical per-dimension cell bases, the chain complex value, and a "
        "replayed d^2 = 0 ledger; each chain group is bounded and the shared "
        "chain-complex kernel verifies the square-zero identity.",
        request_type=CubicalChainComplexRequest,
        result_type=CubicalChainComplexResult,
        run=_chain_complex,
        tags=(
            "topology",
            "cubical-complex",
            "chain-complex",
            "boundary-matrix",
            "exact",
        ),
        discovery_terms=(
            "cubical chain complex",
            "cubical boundary",
        ),
        examples=(
            OperationExample(
                name="unit_square_integer_chain_complex",
                description="Build the integer cubical chain complex of one unit "
                "square (4 vertices, 4 edges, 1 square); faces are closed "
                "automatically and intervals must be unit length (b = a + 1).",
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
            OperationExample(
                name="unit_square_mod_two_chain_complex",
                description="Build the GF(2) cubical chain complex of the same "
                "unit square with boundary coefficients reduced modulo two.",
                input={
                    "cells": [{"intervals": [[0, 1], [0, 1]]}],
                    "coefficient_ring": "GF_p",
                    "prime": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_chain.external_product.compute",
        title="Compute the external product of cubical chains",
        description=(
            "Multiply two finite homogeneous integral cubical chains by "
            "concatenating each left cell's coordinates before each right "
            "cell's coordinates and multiplying coefficients. This uses the "
            "product orientation induced by factor axis order. Inputs admit "
            f"at most {MAX_CUBICAL_CHAIN_VALUE_TERMS:,} terms, "
            f"{MAX_CUBICAL_CHAIN_VALUE_COORDINATE_DIGITS}-digit coordinates, "
            f"and {MAX_CUBICAL_CHAIN_VALUE_COEFFICIENT_DIGITS}-digit "
            "coefficients; product expansion is bounded to "
            f"{MAX_CUBICAL_CHAIN_PRODUCT_TERMS:,} terms and "
            f"{MAX_CUBICAL_CHAIN_PRODUCT_RESULT_BYTES // 1024**2} MiB."
        ),
        request_type=CubicalChainProductRequest,
        result_type=CubicalChainValue,
        run=_chain_product,
        tags=("topology", "cubical", "chain", "product", "exact"),
        discovery_terms=(
            "external product of cubical chains",
            "cubical chain cross product",
            "tensor product of cubical chains",
        ),
        examples=(
            OperationExample(
                name="product_of_two_oriented_edges",
                description=(
                    "Multiply two integral oriented edges to obtain the "
                    "product-oriented square."
                ),
                input={
                    "left": {
                        "ambient_dimension": 1,
                        "degree": 1,
                        "terms": [
                            {
                                "cell": {"intervals": [["0", "1"]]},
                                "coefficient": "2",
                            }
                        ],
                    },
                    "right": {
                        "ambient_dimension": 1,
                        "degree": 1,
                        "terms": [
                            {
                                "cell": {"intervals": [["3", "4"]]},
                                "coefficient": "-3",
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
