"""Root system operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    CartanMatrixRequest,
    CartanTypeRequest,
    CartanTypeResult,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    SimpleReflectionsResult,
    WeylDescentsResult,
    WeylElementLengthResult,
    WeylElementRequest,
    WeylGroupOrderResult,
    WeylLongestElementResult,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_matrix_from_type,
    root_system_data,
    simple_reflection,
    simple_reflections,
    weyl_element_descents,
    weyl_element_length,
    weyl_group_order,
    weyl_longest_element,
)


def _run_cartan_matrix_from_type(request: CartanTypeRequest) -> CartanTypeResult:
    return cartan_matrix_from_type(request.cartan_type, request.rank)


def _run_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    return root_system_data(request.matrix)


def _run_simple_reflections(request: CartanMatrixRequest) -> SimpleReflectionsResult:
    return simple_reflections(request.matrix)


def _run_simple_reflection(request: SimpleReflectionRequest) -> SimpleReflectionResult:
    return simple_reflection(request.matrix, request.vector, request.simple_index)


def _run_weyl_group_order(request: CartanMatrixRequest) -> WeylGroupOrderResult:
    return weyl_group_order(request.matrix)


def _run_weyl_element_length(request: WeylElementRequest) -> WeylElementLengthResult:
    return weyl_element_length(request.matrix, request.word)


def _run_weyl_element_descents(
    request: WeylElementRequest,
) -> WeylDescentsResult:
    return weyl_element_descents(request.matrix, request.word)


def _run_weyl_longest_element(
    request: CartanMatrixRequest,
) -> WeylLongestElementResult:
    return weyl_longest_element(request.matrix)


_A2 = {
    "matrix": {
        "matrix": {
            "domain": "ZZ",
            "row_count": 2,
            "column_count": 2,
            "entries": [["2", "-1"], ["-1", "2"]],
        },
        "simple_root_axis": [0, 1],
    }
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="root_system.cartan_matrix.from_type.compute",
        title="Build a Cartan matrix from a finite Dynkin type and rank",
        description="Build the exact Cartan matrix of a finite irreducible root "
        "system from its Dynkin type (A_n, B_n, C_n, D_n, E_6/E_7/E_8, F_4, "
        "G_2) and rank. The kernel asserts its own output is finite-type; "
        "the returned canonical matrix feeds root-system and Weyl-group "
        "operations directly.",
        request_type=CartanTypeRequest,
        result_type=CartanTypeResult,
        run=_run_cartan_matrix_from_type,
        tags=("algebra", "root-system", "cartan-matrix", "exact"),
        discovery_terms=(
            "cartan matrix by Lie type",
            "dynkin type constructor",
            "finite root system Cartan matrix",
        ),
        examples=(
            OperationExample(
                name="a2_cartan_from_type",
                description="Build the A2 Cartan matrix [[2, -1], [-1, 2]]; "
                "the (type, rank) pair must be a valid finite Dynkin type.",
                input={"cartan_type": "A", "rank": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.positive_roots.compute",
        title="Compute positive roots from a Cartan matrix",
        description="Compute all positive roots of a finite crystallographic root "
        "system from its Cartan matrix, using closure under simple "
        "reflections. Returns simple and positive roots plus highest-root "
        "and Coxeter data for each irreducible component.",
        request_type=CartanMatrixRequest,
        result_type=RootSystemDataResult,
        run=_run_root_system_data,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_cartan",
                description="Compute root system data for A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.simple_reflection.compute",
        title="Apply a simple reflection to a root lattice vector",
        description="Apply the simple reflection s_i to a vector in the root lattice "
        "of a finite crystallographic root system defined by its Cartan "
        f"matrix of rank at most {MAX_RANK}. The vector uses that same "
        "simple-root axis and the index is zero-based.",
        request_type=SimpleReflectionRequest,
        result_type=SimpleReflectionResult,
        run=_run_simple_reflection,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_reflection",
                description="Apply s_0 to the simple root alpha_0 in A2. The matrix is "
                "a finite-type generalized Cartan matrix, the vector has its "
                "two simple-root coordinates, and index 0 is below its rank.",
                input={"matrix": _A2["matrix"], "vector": [1, 0], "simple_index": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.simple_reflections.compute",
        title="Compute all simple-reflection matrices of a Weyl group",
        description=(
            "Compute the exact action matrix of every simple reflection s_i on "
            "the root, coroot, weight, and coweight lattices of a finite "
            "crystallographic root system from its Cartan matrix of rank at "
            f"most {MAX_RANK}. Each returned reflection squares to the "
            "identity, replayed by the kernel before construction."
        ),
        request_type=CartanMatrixRequest,
        result_type=SimpleReflectionsResult,
        run=_run_simple_reflections,
        tags=("algebra", "weyl-group", "root-system", "reflection", "exact"),
        discovery_terms=(
            "simple reflection matrices",
            "Weyl group generators",
            "reflection action on weights",
            "coroot reflection",
        ),
        examples=(
            OperationExample(
                name="a2_simple_reflections",
                description="Compute the two simple-reflection matrices of A2 on all "
                "four lattices; the matrix must be a valid finite-type Cartan "
                "matrix of rank at most 8.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_group_order.compute",
        title="Compute the exact order of a Weyl group",
        description="Compute the exact order of the Weyl group of a finite "
        "crystallographic root system from its Cartan matrix. The kernel "
        "constructs the bounded complete signed-root action and uses SymPy's "
        "Schreier-Sims order algorithm; it never enumerates Weyl-group elements.",
        request_type=CartanMatrixRequest,
        result_type=WeylGroupOrderResult,
        run=_run_weyl_group_order,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_weyl_group_order",
                description="Compute the order of the A2 Weyl group, which is 6; "
                "the matrix must be a finite-type generalized Cartan "
                "matrix of rank at most 8.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.length.compute",
        title="Compute the length and inversion set of a Weyl-group word",
        description="Apply a word in the simple reflections to every positive "
        "root of a finite crystallographic root system and count the "
        "inversions: positive roots sent negative. That count is the "
        "element's length, and the word is reduced exactly when the count "
        "equals its factor count.",
        request_type=WeylElementRequest,
        result_type=WeylElementLengthResult,
        run=_run_weyl_element_length,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "weyl group element length",
            "inversion set of a weyl word",
            "reduced word check",
        ),
        examples=(
            OperationExample(
                name="a2_longest_word_length",
                description="Compute length 3 and the full inversion set of "
                "s_0 s_1 s_0 in A2; each word index must be below the "
                "Cartan rank and the word within the length budget.",
                input={"matrix": _A2["matrix"], "word": [0, 1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.longest_element.compute",
        title="Compute a reduced word for the longest Weyl-group element",
        description="Search the weak order for the longest element of a "
        "finite Weyl group by greedy ascent, returning a reduced word "
        "whose length equals the positive-root count. The kernel asserts "
        "that maximality before construction.",
        request_type=CartanMatrixRequest,
        result_type=WeylLongestElementResult,
        run=_run_weyl_longest_element,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "longest weyl group element",
            "longest reduced word",
            "weak order maximum",
        ),
        examples=(
            OperationExample(
                name="a2_longest_element",
                description="Compute a reduced longest word of length 3 in A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.descents.compute",
        title="Compute the left and right descent sets of a Weyl-group word",
        description="Find the simple reflections that lower a Weyl-group "
        "word's length when prepended (right descents) or appended (left "
        "descents) by testing the word and its reversal on simple roots.",
        request_type=WeylElementRequest,
        result_type=WeylDescentsResult,
        run=_run_weyl_element_descents,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "weyl group descent sets",
            "left right descents",
            "length lowering reflections",
        ),
        examples=(
            OperationExample(
                name="a2_asymmetric_descents",
                description="Compute left {1} and right {0} descents of s_1 s_0 "
                "in A2; each word index must be below the Cartan rank and "
                "the word within the length budget.",
                input={"matrix": _A2["matrix"], "word": [0, 1]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
