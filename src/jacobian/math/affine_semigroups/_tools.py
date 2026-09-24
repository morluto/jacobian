"""Public declaration for exact integer relation lattices."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.affine_semigroups._models import (
    IntegerConfigurationCircuitsRequest,
    IntegerConfigurationCircuitsResult,
    RelationLatticeRequest,
    RelationLatticeResult,
)
from jacobian.math.affine_semigroups.graver import (
    graver_basis,
    markov_basis,
    toric_ideal,
)
from jacobian.math.affine_semigroups.graver_models import (
    IntegerConfigurationGraverBasis,
    IntegerConfigurationGraverRequest,
    IntegerConfigurationMarkovBasis,
    IntegerConfigurationMarkovBasisRequest,
    IntegerConfigurationToricIdealRequest,
)
from jacobian.math.affine_semigroups.operations import (
    integer_configuration_circuits,
    relation_lattice,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS as _SEMIGROUP_TOOLS
from jacobian.math.polynomials.values import RationalPolynomialIdeal


def _run_relation_lattice(request: RelationLatticeRequest) -> RelationLatticeResult:
    return relation_lattice(request.configuration)


def _run_circuits(
    request: IntegerConfigurationCircuitsRequest,
) -> IntegerConfigurationCircuitsResult:
    return integer_configuration_circuits(request.configuration)


_RELATION_TOOLS = (
    MathTool(
        operation_id="integer_configuration.circuits.compute",
        title="Compute primitive circuits of an integer configuration",
        description=(
            "Return every primitive support-minimal integer relation z with "
            "A z = 0, in the retained generator coordinates. Circuits are "
            "sign-normalized and distinct from a Graver basis or a Markov basis. "
            "Admission bounds the configuration to 12 rows and columns, 8 digits "
            "per entry, all support-rank work, and worst-case exact output size."
        ),
        request_type=IntegerConfigurationCircuitsRequest,
        result_type=IntegerConfigurationCircuitsResult,
        run=_run_circuits,
        discovery_terms=(
            "primitive circuit vectors of an integer matrix",
            "support-minimal integer kernel relations",
            "primitive partition identities of a generator configuration",
            "circuits of the represented rational matroid",
        ),
        tags=("integer-configuration", "circuits", "integer-kernel", "exact"),
        examples=(
            OperationExample(
                name="one_row_three_columns",
                description=(
                    "Compute the primitive circuits of [1 2 3]; the output vectors "
                    "use the matrix's three-column generator axis."
                ),
                input={"configuration": {"entries": [["1", "2", "3"]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="integer_configuration.relation_lattice.compute",
        title="Compute the exact integer relation lattice of a configuration",
        description=(
            "Return the complete integer kernel lattice ker_Z(A) of one bounded "
            "integer configuration A in ZZ^(d x n): the canonical row-Hermite "
            "basis in ZZ^n, its unimodular transformation, rank and nullity, "
            "the configuration Smith invariant factors, and the saturation "
            "profile of the embedding. The returned basis satisfies "
            "A B^T = 0 exactly. Admission limits the configuration to 12 rows "
            "and columns and 8 digits per scalar; no solution search, Hilbert "
            "basis, or toric-ideal claim is made."
        ),
        request_type=RelationLatticeRequest,
        result_type=RelationLatticeResult,
        run=_run_relation_lattice,
        discovery_terms=(
            "integer kernel of a generator matrix",
            "integer relation lattice of a configuration",
            "canonical HNF basis of integer linear relations",
            "rank and nullity of an integer matrix over ZZ",
            "Smith normal form kernel generators",
        ),
        tags=(
            "lattice",
            "integer",
            "kernel",
            "smith-normal-form",
            "hermite-normal-form",
            "exact",
        ),
        examples=(
            OperationExample(
                name="one_by_three_configuration",
                description=(
                    "Compute ker_Z([1 2 3]), a rank-two sublattice of ZZ^3 whose "
                    "basis replays [1 2 3] B^T = 0."
                ),
                input={"configuration": {"entries": [["1", "2", "3"]]}},
            ),
        ),
    ),
)


def _run_graver(
    request: IntegerConfigurationGraverRequest,
) -> IntegerConfigurationGraverBasis:
    return graver_basis(request.configuration)


def _run_toric_ideal(
    request: IntegerConfigurationToricIdealRequest,
) -> RationalPolynomialIdeal:
    return toric_ideal(request.configuration)


def _run_markov_basis(
    request: IntegerConfigurationMarkovBasisRequest,
) -> IntegerConfigurationMarkovBasis:
    return markov_basis(request.configuration)


_GRAVER_TOOLS = (
    MathTool(
        operation_id="integer_configuration.graver_basis.compute",
        title="Compute a complete one-row integer Graver basis",
        description=(
            "Return every sign-normalized conformally indecomposable relation of a "
            "one-row integer configuration. Completeness is exact for the accepted "
            "input; admission limits the matrix to five columns and preflights the "
            "complete coordinate-box and candidate-pair minimality work. This operation "
            "does not claim a Markov basis."
        ),
        request_type=IntegerConfigurationGraverRequest,
        result_type=IntegerConfigurationGraverBasis,
        run=_run_graver,
        discovery_terms=(
            "complete Graver basis of a one-row integer matrix",
            "conformally indecomposable primitive partition identities",
            "one-dimensional integer kernel move basis",
        ),
        tags=("integer-configuration", "graver-basis", "exact"),
        examples=(
            OperationExample(
                name="one_row_partition_identities",
                description="Compute all primitive moves for [1 2 3].",
                input={"configuration": {"entries": [["1", "2", "3"]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="integer_configuration.toric_ideal.compute",
        title="Compute a bounded one-row toric ideal",
        description=(
            "Return a Graver-generated presentation of the kernel of the monomial "
            "map QQ[x_1,...,x_n] -> QQ[t], x_j -> t^a_j, for one-row nonnegative "
            "integer weights. Polynomial variables retain the configuration's "
            "ordered generator labels. One- and two-column cases use the exact "
            "primitive-kernel formula; three-to-five columns use the complete "
            "Graver enumeration after preflighting its l1 generator envelope, "
            "100,000,000 candidate-pair work limit, polynomial exponent limit, "
            "64-generator ideal-carrier limit, and serialized output size. This "
            "is a generating presentation, not a minimal one."
        ),
        request_type=IntegerConfigurationToricIdealRequest,
        result_type=RationalPolynomialIdeal,
        run=_run_toric_ideal,
        discovery_terms=(
            "toric ideal of a one-row nonnegative integer matrix",
            "integer configuration binomial ideal",
            "kernel of a monomial map to a univariate polynomial ring",
        ),
        tags=("integer-configuration", "toric-ideal", "graver-basis", "exact"),
        examples=(
            OperationExample(
                name="weights_one_one_two",
                description=(
                    "Compute the QQ toric ideal for weights (1,1,2); the returned "
                    "polynomials use the exact ordered variables (u,v,w)."
                ),
                input={
                    "configuration": {
                        "row_labels": ["degree"],
                        "generator_labels": ["u", "v", "w"],
                        "entries": [["1", "1", "2"]],
                    }
                },
            ),
        ),
    ),
)

_MARKOV_TOOLS = (
    MathTool(
        operation_id="integer_configuration.markov_basis.compute",
        title="Compute a global Markov basis of a one-row configuration",
        description=(
            "Return a complete move family that generates the integer "
            "configuration's toric ideal and connects every nonnegative fiber. "
            "The bounded implementation returns the complete Graver basis on "
            "the same generator axis; this is a global guarantee, unlike "
            "connectivity of any one materialized fiber. Admission uses the "
            "Graver work, coordinate, and result-size limits."
        ),
        request_type=IntegerConfigurationMarkovBasisRequest,
        result_type=IntegerConfigurationMarkovBasis,
        run=_run_markov_basis,
        discovery_terms=(
            "global Markov basis for an integer configuration",
            "moves connecting every nonnegative factorization fiber",
            "generators of a toric ideal as integer kernel moves",
        ),
        tags=("integer-configuration", "markov-basis", "toric-ideal", "exact"),
        examples=(
            OperationExample(
                name="one_row_three_weights",
                description=(
                    "Compute globally connecting moves for weights (1,2,3); "
                    "move coordinates use the retained generator axis."
                ),
                input={"configuration": {"entries": [["1", "2", "3"]]}},
            ),
        ),
    ),
)

TOOLS: MathTools = _RELATION_TOOLS + _SEMIGROUP_TOOLS + _GRAVER_TOOLS + _MARKOV_TOOLS

__all__ = ["TOOLS"]
