from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineFactorization,
    AffineFiber,
    AffineFiberGraph,
    AffineHilbertBasis,
    AffineMembershipResult,
    AffineSemigroupNormalization,
    PositiveAffineSemigroup,
    PositiveGradingResult,
    _evaluate_factorization,
    construct,
    fiber,
    fiber_graph,
    hilbert_basis,
    membership,
    normalization,
    positive_grading,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineFactorizationRequest,
    AffineFiberGraphRequest,
    AffineFiberRequest,
    AffineHilbertBasisRequest,
    AffineMembershipRequest,
    AffineSemigroupNormalizationRequest,
    AffineSemigroupRequest,
    PositiveGradingRequest,
)


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


def _factorization(r: AffineFactorizationRequest) -> AffineFactorization:
    try:
        return _evaluate_factorization(r.semigroup, r.coordinates, validate_parent=True)
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("coordinates",),
            code="affine_semigroup.factorization",
            message=str(e),
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


def _normalization(
    r: AffineSemigroupNormalizationRequest,
) -> AffineSemigroupNormalization:
    try:
        return normalization(r.semigroup)
    except OperationDomainValidationError:
        # The native boundary already exposes the stable owner diagnostic;
        # preserve it so direct Python and catalog invocations agree.
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.normalization",
            message=str(e),
        ) from e


TOOLS = (
    MathTool(
        operation_id="affine_semigroup.factorization.evaluate",
        title="Evaluate a parent-bound affine-semigroup factorization",
        description=(
            "Evaluate one nonnegative coefficient vector on the retained "
            "generator axis and return an AffineFactorization carrying its "
            "positive semigroup parent and exact ambient target. This is a "
            "single O(rows x generators) matrix product; it does not enumerate "
            "a fiber or claim that the factorization is unique. Admission limits "
            "coefficients to 32 decimal digits and preflights arithmetic and output size."
        ),
        request_type=AffineFactorizationRequest,
        result_type=AffineFactorization,
        run=_factorization,
        discovery_terms=(
            "evaluate affine semigroup factorization coordinates",
            "map a nonnegative generator vector to its exact semigroup element",
            "parent-bound affine semigroup element from factorization",
        ),
        tags=("affine-semigroup", "factorization", "exact"),
        examples=(
            OperationExample(
                name="quadrant_factorization",
                description=(
                    "Evaluate (2,3) on generators (1,0),(0,1), returning "
                    "the parent-bound element (2,3)."
                ),
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["x", "y"],
                            "generator_labels": ["a", "b"],
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                        "grading": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                    },
                    "coordinates": ["2", "3"],
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
        operation_id="affine_semigroup.normalization.compute",
        title="Compute the normalization of a two-dimensional affine semigroup",
        description=(
            "Return the minimal generators of cone(S) intersect gp(S) for a "
            "positive, full-rank affine semigroup in Z^2. Computation is in the "
            "lattice generated by S, then transported back to the retained "
            "ambient row axes. The cone Hilbert determinant is at most 1,000."
        ),
        request_type=AffineSemigroupNormalizationRequest,
        result_type=AffineSemigroupNormalization,
        run=_normalization,
        discovery_terms=(
            "normalization of a two-dimensional affine semigroup",
            "integral closure in the group lattice",
            "minimal generators of cone(S) intersect gp(S)",
        ),
        tags=("affine-semigroup", "normalization", "hilbert-basis", "exact"),
        examples=(
            OperationExample(
                name="diagonal_submonoid",
                description=(
                    "Normalize the semigroup generated by (2,0), (0,2), and "
                    "(2,2). Its generated group is 2Z^2, so the normalization "
                    "generators are (0,2) and (2,0), in canonical order."
                ),
                input={
                    "semigroup": {
                        "configuration": {
                            "row_labels": ["x", "y"],
                            "generator_labels": ["a", "b", "c"],
                            "entries": [["2", "0", "2"], ["0", "2", "2"]],
                        },
                        "grading": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
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
