from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.group_lattice import (
    AffineGroupLattice,
    compute_group_lattice,
)
from jacobian.math.affine_semigroups.group_lattice_models import (
    AffineGroupLatticeRequest,
)
from jacobian.math.affine_semigroups.holes import (
    AffineSemigroupHoleProfile,
    AffineSemigroupHolesRequest,
    holes_through_degree,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineFiber,
    AffineFiberGraph,
    AffineHilbertBasis,
    AffineMembershipResult,
    PositiveAffineSemigroup,
    PositiveGradingResult,
    construct,
    fiber,
    fiber_graph,
    hilbert_basis,
    membership,
    positive_grading,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineFiberGraphRequest,
    AffineFiberRequest,
    AffineHilbertBasisRequest,
    AffineMembershipRequest,
    AffineSemigroupRequest,
    PositiveGradingRequest,
)


def _group_lattice(r: AffineGroupLatticeRequest) -> AffineGroupLattice:
    return compute_group_lattice(r.configuration)


def _holes_through_degree(
    request: AffineSemigroupHolesRequest,
) -> AffineSemigroupHoleProfile:
    try:
        return holes_through_degree(request.semigroup, request.max_degree)
    except (OperationResourceAdmissionError, OperationDomainValidationError):
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.hole_profile",
            message=str(exc),
        ) from exc


def _grading(r: PositiveGradingRequest) -> PositiveGradingResult:
    try:
        return positive_grading(r.configuration)
    except OperationResourceAdmissionError:
        raise
    except ValueError as e:
        raise OperationDomainValidationError(
            location=("configuration",), code="affine_semigroup.grading", message=str(e)
        ) from e


def _construct(r: AffineSemigroupRequest) -> PositiveAffineSemigroup:
    try:
        return construct(r.configuration, r.grading)
    except ValueError as e:
        raise OperationDomainValidationError(
            location=("grading",), code="affine_semigroup.grading", message=str(e)
        ) from e


def _fiber(r: AffineFiberRequest) -> AffineFiber:
    try:
        return fiber(r.semigroup, r.target)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("target",), code="affine_semigroup.fiber", message=str(e)
        ) from e


def _membership(r: AffineMembershipRequest) -> AffineMembershipResult:
    try:
        return membership(r.semigroup, r.target)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("target",), code="affine_semigroup.membership", message=str(e)
        ) from e


def _fiber_graph(r: AffineFiberGraphRequest) -> AffineFiberGraph:
    try:
        return fiber_graph(r.semigroup, r.target, r.moves)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("moves",), code="affine_semigroup.fiber_graph", message=str(e)
        ) from e


def _hilbert_basis(r: AffineHilbertBasisRequest) -> AffineHilbertBasis:
    try:
        return hilbert_basis(r.configuration)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.hilbert_basis",
            message=str(e),
        ) from e


TOOLS = (
    MathTool(
        operation_id="affine_semigroup.holes_through_degree.compute",
        title="Enumerate affine-semigroup holes through a positive degree",
        description=(
            "Return every point h in cone(S) intersect gp(S) with the retained "
            "positive grading at most max_degree that is not in S. The operation "
            "currently admits full-rank pointed cones in two ambient dimensions; "
            "it enumerates a bounded containing box in generated-lattice coordinates "
            "and closes the finite semigroup reachability set. Candidate points, "
            "work, scalar heights, and the complete exact output are preflighted. "
            "A degree-bounded profile is not the global set of holes."
        ),
        request_type=AffineSemigroupHolesRequest,
        result_type=AffineSemigroupHoleProfile,
        run=_holes_through_degree,
        tags=("affine-semigroup", "holes", "normalization", "exact"),
        discovery_terms=(
            "affine semigroup holes through degree",
            "bounded holes in a positive affine semigroup",
            "lattice points in the normalization missing from the semigroup",
            "degree bounded nonnormality profile",
        ),
        examples=(
            OperationExample(
                name="parity_holes_through_degree_two",
                description=(
                    "For generators (2,0), (0,2), (1,1), (1,0), the generated "
                    "lattice is Z^2 and the only hole of x+y degree at most two "
                    "is (0,1)."
                ),
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["x", "y"],
                            "generator_labels": ["g0", "g1", "g2", "g3"],
                            "entries": [
                                ["2", "0", "1", "1"],
                                ["0", "2", "1", "0"],
                            ],
                        },
                        "grading": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "max_degree": "2",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.group_lattice.compute",
        title="Compute the generated ambient lattice",
        description=(
            "Return the canonical exact integer lattice generated by the labelled "
            "columns of a bounded affine configuration, retaining its source "
            "configuration and ambient coordinate dimension."
        ),
        request_type=AffineGroupLatticeRequest,
        result_type=AffineGroupLattice,
        run=_group_lattice,
        tags=("affine-semigroup", "lattice", "integer", "exact"),
        examples=(
            OperationExample(
                name="even_axis_lattice",
                description=(
                    "Compute the subgroup generated by (2,0), (0,2), and (2,2)."
                ),
                input={
                    "configuration": {
                        "row_labels": ["x", "y"],
                        "generator_labels": ["a", "b", "c"],
                        "entries": [["2", "0", "2"], ["0", "2", "2"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.hilbert_basis.compute",
        title="Compute a complete two-dimensional affine Hilbert basis",
        description=(
            "Return the complete Hilbert basis of the cone generated by a "
            "labelled configuration in the ambient lattice Z^2. The cone must "
            "be full-dimensional and pointed; the primitive extreme-ray "
            "determinant is at most 1,000. This is the Hilbert basis of the "
            "cone, not a basis of the possibly smaller semigroup generated by "
            "the supplied columns."
        ),
        request_type=AffineHilbertBasisRequest,
        result_type=AffineHilbertBasis,
        run=_hilbert_basis,
        discovery_terms=(
            "Hilbert basis of a pointed rational cone in Z^2",
            "minimal additive generators of a two-dimensional saturated affine semigroup",
            "indecomposable lattice points in a rational cone",
        ),
        tags=("affine-semigroup", "hilbert-basis", "cone", "exact"),
        examples=(
            OperationExample(
                name="determinant_three_cone",
                description=(
                    "Compute the Hilbert basis of the cone with primitive rays "
                    "(1,0) and (1,3)."
                ),
                input={
                    "configuration": {
                        "row_labels": ["x", "y"],
                        "generator_labels": ["u", "v"],
                        "entries": [["1", "1"], ["0", "3"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="integer_configuration.positive_grading.compute",
        title="Find an exact positive grading",
        description="Find an exact rational covector positive on every labelled generator; bounded search exhaustion is an admission failure, not a nonmembership claim.",
        request_type=PositiveGradingRequest,
        result_type=PositiveGradingResult,
        run=_grading,
        tags=("affine-semigroup", "grading", "exact"),
        examples=(
            OperationExample(
                name="positive_ray",
                description="Find a positive grading for generators (1,0) and (0,1).",
                input={
                    "configuration": {
                        "row_labels": ["x", "y"],
                        "generator_labels": ["a", "b"],
                        "entries": [["1", "0"], ["0", "1"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.from_generators.compute",
        title="Construct a positive affine semigroup",
        description="Construct the canonical positive affine semigroup from labelled integer generators and an exact positive grading.",
        request_type=AffineSemigroupRequest,
        result_type=PositiveAffineSemigroup,
        run=_construct,
        tags=("affine-semigroup", "exact"),
        examples=(
            OperationExample(
                name="quadrant",
                description="Construct the positive semigroup generated by (1,0),(0,1); the grading must be positive on both generators.",
                input={
                    "configuration": {
                        "row_labels": ["x", "y"],
                        "generator_labels": ["a", "b"],
                        "entries": [["1", "0"], ["0", "1"]],
                    },
                    "grading": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.factorizations.compute",
        title="Compute a complete affine-semigroup fiber",
        description="Return every nonnegative integer factorization Au=b for one positive labelled affine semigroup.",
        request_type=AffineFiberRequest,
        result_type=AffineFiber,
        run=_fiber,
        tags=("affine-semigroup", "fiber", "exact"),
        examples=(
            OperationExample(
                name="fiber_2",
                description="Compute all factorizations of (2,0) in the quadrant semigroup; the target uses the labelled row axis.",
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["x", "y"],
                            "generator_labels": ["a", "b"],
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                        "grading": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                    "target": ["2", "0"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.membership.compute",
        title="Classify affine-semigroup membership",
        description="Distinguish exact cone, ambient lattice, and nonnegative semigroup membership and return a factorization when present.",
        request_type=AffineMembershipRequest,
        result_type=AffineMembershipResult,
        run=_membership,
        tags=("affine-semigroup", "membership", "exact"),
        examples=(
            OperationExample(
                name="member",
                description="Classify (2,1) in the quadrant semigroup; the target uses the labelled row axis.",
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["x", "y"],
                            "generator_labels": ["a", "b"],
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                        "grading": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                    "target": ["2", "1"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="affine_semigroup.fiber_graph.compute",
        title="Compute a bounded affine fiber graph",
        description=(
            "Return the complete factorization vertices and connected components "
            "of one admitted fiber under at most 16 supplied exact kernel moves. "
            "This classifies only the selected fiber and makes no global Markov-basis claim."
        ),
        request_type=AffineFiberGraphRequest,
        result_type=AffineFiberGraph,
        run=_fiber_graph,
        tags=("affine-semigroup", "fiber", "moves", "graph", "exact"),
        examples=(
            OperationExample(
                name="three_factorizations_one_move",
                description=(
                    "Build the graph on factorizations of 2 for the generators "
                    "1,1,1, with the move (1,-1,0)."
                ),
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["degree"],
                            "generator_labels": ["a", "b", "c"],
                            "entries": [["1", "1", "1"]],
                        },
                        "grading": [{"num": "1", "den": "1"}],
                    },
                    "target": ["2"],
                    "moves": [["1", "-1", "0"]],
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
