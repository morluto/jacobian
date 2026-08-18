"""Explicit immutable inventory of built-in mathematical tools."""

from __future__ import annotations

from jacobian.catalog.admission import OperationAdmission, curate_public_tools
from jacobian.catalog.models import MathTools

# Admission rows co-located with each owning math domain
from jacobian.math.additive_combinatorics._admission import (
    ADMISSIONS as ADDITIVE_COMBINATORICS_ADMISSIONS,
)
from jacobian.math.additive_combinatorics._tools import (
    TOOLS as ADDITIVE_COMBINATORICS_TOOLS,
)
from jacobian.math.algebraic_combinatorics._admission import (
    ADMISSIONS as ALGEBRAIC_COMBINATORICS_ADMISSIONS,
)
from jacobian.math.algebraic_combinatorics._tools import (
    TOOLS as ALGEBRAIC_COMBINATORICS_TOOLS,
)
from jacobian.math.analysis._admission import ADMISSIONS as ANALYSIS_ADMISSIONS
from jacobian.math.analysis._tools import TOOLS as ANALYSIS_TOOLS
from jacobian.math.arithmetic._admission import ADMISSIONS as ARITHMETIC_ADMISSIONS
from jacobian.math.arithmetic._tools import TOOLS as ARITHMETIC_TOOLS
from jacobian.math.arithmetic_counting._admission import (
    ADMISSIONS as ARITHMETIC_COUNTING_ADMISSIONS,
)
from jacobian.math.arithmetic_counting._tools import TOOLS as ARITHMETIC_COUNTING_TOOLS
from jacobian.math.arithmetic_dynamics._admission import (
    ADMISSIONS as ARITHMETIC_DYNAMICS_ADMISSIONS,
)
from jacobian.math.arithmetic_dynamics._tools import (
    TOOLS as ARITHMETIC_DYNAMICS_TOOLS,
)
from jacobian.math.arithmetic_functions._admission import (
    ADMISSIONS as ARITHMETIC_FUNCTIONS_ADMISSIONS,
)
from jacobian.math.arithmetic_functions._tools import (
    TOOLS as ARITHMETIC_FUNCTIONS_TOOLS,
)
from jacobian.math.boolean._admission import ADMISSIONS as BOOLEAN_ADMISSIONS
from jacobian.math.boolean._tools import TOOLS as BOOLEAN_TOOLS
from jacobian.math.boolean_analysis._admission import (
    ADMISSIONS as BOOLEAN_ANALYSIS_ADMISSIONS,
)
from jacobian.math.boolean_analysis._tools import TOOLS as BOOLEAN_ANALYSIS_TOOLS
from jacobian.math.code_theory._admission import ADMISSIONS as CODE_THEORY_ADMISSIONS
from jacobian.math.code_theory._tools import TOOLS as CODE_THEORY_TOOLS
from jacobian.math.combinatorics._admission import (
    ADMISSIONS as COMBINATORICS_ADMISSIONS,
)
from jacobian.math.combinatorics._tools import TOOLS as COMBINATORICS_TOOLS
from jacobian.math.convex_analysis._admission import (
    ADMISSIONS as CONVEX_ANALYSIS_ADMISSIONS,
)
from jacobian.math.convex_analysis._tools import TOOLS as CONVEX_ANALYSIS_TOOLS
from jacobian.math.diophantine_approximation._admission import (
    ADMISSIONS as DIOPHANTINE_APPROXIMATION_ADMISSIONS,
)
from jacobian.math.diophantine_approximation._tools import (
    TOOLS as DIOPHANTINE_APPROXIMATION_TOOLS,
)
from jacobian.math.discrepancy_theory._admission import (
    ADMISSIONS as DISCREPANCY_THEORY_ADMISSIONS,
)
from jacobian.math.discrepancy_theory._tools import TOOLS as DISCREPANCY_THEORY_TOOLS
from jacobian.math.electrical_networks._admission import (
    ADMISSIONS as ELECTRICAL_NETWORKS_ADMISSIONS,
)
from jacobian.math.electrical_networks._tools import (
    TOOLS as ELECTRICAL_NETWORKS_TOOLS,
)
from jacobian.math.finite_fields._admission import (
    ADMISSIONS as FINITE_FIELDS_ADMISSIONS,
)
from jacobian.math.finite_fields._tools import TOOLS as FINITE_FIELDS_TOOLS
from jacobian.math.finite_game_theory._admission import (
    ADMISSIONS as FINITE_GAME_THEORY_ADMISSIONS,
)
from jacobian.math.finite_game_theory._tools import TOOLS as FINITE_GAME_THEORY_TOOLS
from jacobian.math.finite_metric_spaces._admission import (
    ADMISSIONS as FINITE_METRIC_SPACES_ADMISSIONS,
)
from jacobian.math.finite_metric_spaces._tools import (
    TOOLS as FINITE_METRIC_SPACES_TOOLS,
)
from jacobian.math.finite_sets._admission import ADMISSIONS as FINITE_SETS_ADMISSIONS
from jacobian.math.finite_sets._tools import TOOLS as FINITE_SETS_TOOLS
from jacobian.math.finite_state_transducers._admission import (
    ADMISSIONS as FINITE_STATE_TRANSDUCERS_ADMISSIONS,
)
from jacobian.math.finite_state_transducers._tools import (
    TOOLS as FINITE_STATE_TRANSDUCER_TOOLS,
)
from jacobian.math.finite_topology._admission import (
    ADMISSIONS as FINITE_TOPOLOGY_ADMISSIONS,
)
from jacobian.math.finite_topology._tools import TOOLS as FINITE_TOPOLOGY_TOOLS
from jacobian.math.formal_power_series._admission import (
    ADMISSIONS as FORMAL_POWER_SERIES_ADMISSIONS,
)
from jacobian.math.formal_power_series._tools import TOOLS as FORMAL_POWER_SERIES_TOOLS
from jacobian.math.geometry._admission import ADMISSIONS as GEOMETRY_ADMISSIONS
from jacobian.math.geometry._tools import TOOLS as GEOMETRY_TOOLS
from jacobian.math.geometry.euclidean._admission import (
    ADMISSIONS as GEOMETRY_EUCLIDEAN_ADMISSIONS,
)
from jacobian.math.geometry.euclidean._tools import TOOLS as EUCLIDEAN_GEOMETRY_TOOLS
from jacobian.math.geometry.exact._admission import (
    ADMISSIONS as GEOMETRY_EXACT_ADMISSIONS,
)
from jacobian.math.geometry.exact._tools import TOOLS as EXACT_GEOMETRY_TOOLS
from jacobian.math.geometry.projective._admission import (
    ADMISSIONS as GEOMETRY_PROJECTIVE_ADMISSIONS,
)
from jacobian.math.geometry.projective._tools import TOOLS as PROJECTIVE_GEOMETRY_TOOLS
from jacobian.math.graphical_models._admission import (
    ADMISSIONS as GRAPHICAL_MODELS_ADMISSIONS,
)
from jacobian.math.graphical_models._tools import (
    TOOLS as GRAPHICAL_MODEL_TOOLS,
)
from jacobian.math.graphs._admission import ADMISSIONS as GRAPHS_ADMISSIONS
from jacobian.math.graphs._tools import TOOLS as GRAPHS_TOOLS
from jacobian.math.graphs.coloring._admission import (
    ADMISSIONS as GRAPHS_COLORING_ADMISSIONS,
)
from jacobian.math.graphs.coloring._tools import TOOLS as GRAPH_COLORING_OPS_TOOLS
from jacobian.math.graphs.decomposition._admission import (
    ADMISSIONS as GRAPHS_DECOMPOSITION_ADMISSIONS,
)
from jacobian.math.graphs.decomposition._tools import TOOLS as GRAPH_DECOMPOSITION_TOOLS
from jacobian.math.graphs.directed._admission import (
    ADMISSIONS as GRAPHS_DIRECTED_ADMISSIONS,
)
from jacobian.math.graphs.directed._tools import TOOLS as DIRECTED_GRAPH_TOOLS
from jacobian.math.graphs.flow._admission import ADMISSIONS as GRAPHS_FLOW_ADMISSIONS
from jacobian.math.graphs.flow._tools import TOOLS as GRAPH_FLOW_TOOLS
from jacobian.math.graphs.isomorphism._admission import (
    ADMISSIONS as GRAPHS_ISOMORPHISM_ADMISSIONS,
)
from jacobian.math.graphs.isomorphism._tools import TOOLS as GRAPH_ISOMORPHISM_TOOLS
from jacobian.math.graphs.optimization._admission import (
    ADMISSIONS as GRAPHS_OPTIMIZATION_ADMISSIONS,
)
from jacobian.math.graphs.optimization._tools import TOOLS as GRAPH_OPTIMIZATION_TOOLS
from jacobian.math.graphs.polynomials._admission import (
    ADMISSIONS as GRAPHS_POLYNOMIALS_ADMISSIONS,
)
from jacobian.math.graphs.polynomials._tools import TOOLS as GRAPH_POLYNOMIALS_TOOLS
from jacobian.math.graphs.realization._admission import (
    ADMISSIONS as GRAPHS_REALIZATION_ADMISSIONS,
)
from jacobian.math.graphs.realization._tools import TOOLS as GRAPH_REALIZATION_TOOLS
from jacobian.math.graphs.spectral._admission import (
    ADMISSIONS as GRAPHS_SPECTRAL_ADMISSIONS,
)
from jacobian.math.graphs.spectral._tools import TOOLS as GRAPH_SPECTRAL_TOOLS
from jacobian.math.graphs.symmetry._admission import (
    ADMISSIONS as GRAPHS_SYMMETRY_ADMISSIONS,
)
from jacobian.math.graphs.symmetry._tools import TOOLS as GRAPH_SYMMETRY_TOOLS
from jacobian.math.graphs.transforms._admission import (
    ADMISSIONS as GRAPHS_TRANSFORMS_ADMISSIONS,
)
from jacobian.math.graphs.transforms._tools import TOOLS as GRAPH_TRANSFORMS_TOOLS
from jacobian.math.group._admission import ADMISSIONS as GROUP_ADMISSIONS
from jacobian.math.group._tools import TOOLS as GROUP_TOOLS
from jacobian.math.impartial_games._admission import (
    ADMISSIONS as IMPARTIAL_GAMES_ADMISSIONS,
)
from jacobian.math.impartial_games._tools import TOOLS as IMPARTIAL_GAMES_TOOLS
from jacobian.math.lattices._admission import ADMISSIONS as LATTICES_ADMISSIONS
from jacobian.math.lattices._tools import TOOLS as LATTICES_TOOLS
from jacobian.math.logic._admission import ADMISSIONS as LOGIC_ADMISSIONS
from jacobian.math.logic._tools import TOOLS as LOGIC_TOOLS
from jacobian.math.markov_chain._admission import ADMISSIONS as MARKOV_CHAIN_ADMISSIONS
from jacobian.math.markov_chain._tools import TOOLS as MARKOV_CHAIN_TOOLS
from jacobian.math.matrices._admission import ADMISSIONS as MATRICES_ADMISSIONS
from jacobian.math.matrices._tools import TOOLS as MATRICES_TOOLS
from jacobian.math.matrices.analysis._admission import (
    ADMISSIONS as MATRICES_ANALYSIS_ADMISSIONS,
)
from jacobian.math.matrices.analysis._tools import TOOLS as MATRIX_ANALYSIS_TOOLS
from jacobian.math.matrices.canonical_forms._admission import (
    ADMISSIONS as MATRICES_CANONICAL_FORMS_ADMISSIONS,
)
from jacobian.math.matrices.canonical_forms._tools import TOOLS as CANONICAL_FORMS_TOOLS
from jacobian.math.matrices.certified_snf._admission import (
    ADMISSIONS as MATRICES_CERTIFIED_SNF_ADMISSIONS,
)
from jacobian.math.matrices.certified_snf._tools import TOOLS as CERTIFIED_SNF_TOOLS
from jacobian.math.matrices.rational_linear._admission import (
    ADMISSIONS as MATRICES_RATIONAL_LINEAR_ADMISSIONS,
)
from jacobian.math.matrices.rational_linear._tools import TOOLS as RATIONAL_LINEAR_TOOLS
from jacobian.math.matrices.symbolic._admission import (
    ADMISSIONS as MATRICES_SYMBOLIC_ADMISSIONS,
)
from jacobian.math.matrices.symbolic._tools import TOOLS as SYMBOLIC_MATRIX_TOOLS
from jacobian.math.multiple_testing._admission import (
    ADMISSIONS as MULTIPLE_TESTING_ADMISSIONS,
)
from jacobian.math.multiple_testing._tools import TOOLS as MULTIPLE_TESTING_TOOLS
from jacobian.math.number_field._admission import ADMISSIONS as NUMBER_FIELD_ADMISSIONS
from jacobian.math.number_field._tools import TOOLS as NUMBER_FIELD_TOOLS
from jacobian.math.number_theory._admission import (
    ADMISSIONS as NUMBER_THEORY_ADMISSIONS,
)
from jacobian.math.number_theory._tools import TOOLS as NUMBER_THEORY_TOOLS
from jacobian.math.numerical_semigroups._admission import (
    ADMISSIONS as NUMERICAL_SEMIGROUPS_ADMISSIONS,
)
from jacobian.math.numerical_semigroups._tools import (
    TOOLS as NUMERICAL_SEMIGROUPS_TOOLS,
)
from jacobian.math.optimization._admission import ADMISSIONS as OPTIMIZATION_ADMISSIONS
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS
from jacobian.math.petri_nets._admission import ADMISSIONS as PETRI_NETS_ADMISSIONS
from jacobian.math.petri_nets._tools import TOOLS as PETRI_NET_TOOLS
from jacobian.math.polynomials._admission import ADMISSIONS as POLYNOMIALS_ADMISSIONS
from jacobian.math.polynomials._tools import TOOLS as POLYNOMIAL_TOOLS
from jacobian.math.polynomials.maps._admission import (
    ADMISSIONS as POLYNOMIALS_MAPS_ADMISSIONS,
)
from jacobian.math.polynomials.maps._tools import TOOLS as POLYNOMIAL_MAPS_TOOLS
from jacobian.math.polynomials.multivariate._admission import (
    ADMISSIONS as POLYNOMIALS_MULTIVARIATE_ADMISSIONS,
)
from jacobian.math.polynomials.multivariate._tools import (
    TOOLS as MULTIVARIATE_POLYNOMIAL_TOOLS,
)
from jacobian.math.polynomials.real_algebra._admission import (
    ADMISSIONS as POLYNOMIALS_REAL_ALGEBRA_ADMISSIONS,
)
from jacobian.math.polynomials.real_algebra._tools import TOOLS as REAL_ALGEBRA_TOOLS
from jacobian.math.posets._admission import ADMISSIONS as POSETS_ADMISSIONS
from jacobian.math.posets._tools import TOOLS as POSETS_TOOLS
from jacobian.math.probability._admission import ADMISSIONS as PROBABILITY_ADMISSIONS
from jacobian.math.probability._tools import TOOLS as PROBABILITY_TOOLS
from jacobian.math.recurrence_solving._admission import (
    ADMISSIONS as RECURRENCE_SOLVING_ADMISSIONS,
)
from jacobian.math.recurrence_solving._tools import TOOLS as RECURRENCE_SOLVING_TOOLS
from jacobian.math.regular_languages._admission import (
    ADMISSIONS as REGULAR_LANGUAGES_ADMISSIONS,
)
from jacobian.math.regular_languages._tools import TOOLS as REGULAR_LANGUAGES_TOOLS
from jacobian.math.root_isolation._admission import (
    ADMISSIONS as ROOT_ISOLATION_ADMISSIONS,
)
from jacobian.math.root_isolation._tools import TOOLS as ROOT_ISOLATION_TOOLS
from jacobian.math.sequences._admission import ADMISSIONS as SEQUENCES_ADMISSIONS
from jacobian.math.sequences._tools import TOOLS as SEQUENCES_TOOLS
from jacobian.math.submodular_opt._admission import (
    ADMISSIONS as SUBMODULAR_OPT_ADMISSIONS,
)
from jacobian.math.submodular_opt._tools import TOOLS as SUBMODULAR_OPT_TOOLS
from jacobian.math.symbolic_dynamics._admission import (
    ADMISSIONS as SYMBOLIC_DYNAMICS_ADMISSIONS,
)
from jacobian.math.symbolic_dynamics._tools import TOOLS as SYMBOLIC_DYNAMICS_TOOLS
from jacobian.math.term_rewriting._admission import (
    ADMISSIONS as TERM_REWRITING_ADMISSIONS,
)
from jacobian.math.term_rewriting._tools import TOOLS as TERM_REWRITING_TOOLS
from jacobian.math.topology._admission import ADMISSIONS as TOPOLOGY_ADMISSIONS
from jacobian.math.topology._tools import TOOLS as TOPOLOGY_TOOLS
from jacobian.math.tree_automata._admission import (
    ADMISSIONS as TREE_AUTOMATA_ADMISSIONS,
)
from jacobian.math.tree_automata._tools import TOOLS as TREE_AUTOMATA_TOOLS
from jacobian.math.words._admission import ADMISSIONS as WORDS_ADMISSIONS
from jacobian.math.words._tools import TOOLS as WORDS_TOOLS

_BUILTIN_CANDIDATES: MathTools = (
    *BOOLEAN_TOOLS,
    *GROUP_TOOLS,
    *GRAPH_COLORING_OPS_TOOLS,
    *GRAPH_SPECTRAL_TOOLS,
    *GRAPH_FLOW_TOOLS,
    *GRAPH_DECOMPOSITION_TOOLS,
    *GRAPH_ISOMORPHISM_TOOLS,
    *ROOT_ISOLATION_TOOLS,
    *RECURRENCE_SOLVING_TOOLS,
    *CODE_THEORY_TOOLS,
    *NUMBER_FIELD_TOOLS,
    *MARKOV_CHAIN_TOOLS,
    *ARITHMETIC_TOOLS,
    *ARITHMETIC_DYNAMICS_TOOLS,
    *NUMBER_THEORY_TOOLS,
    *DIOPHANTINE_APPROXIMATION_TOOLS,
    *COMBINATORICS_TOOLS,
    *FINITE_SETS_TOOLS,
    *FINITE_FIELDS_TOOLS,
    *LOGIC_TOOLS,
    *SEQUENCES_TOOLS,
    *GEOMETRY_TOOLS,
    *PROJECTIVE_GEOMETRY_TOOLS,
    *GRAPH_OPTIMIZATION_TOOLS,
    *GRAPHS_TOOLS,
    *GRAPH_SYMMETRY_TOOLS,
    *GRAPH_TRANSFORMS_TOOLS,
    *CERTIFIED_SNF_TOOLS,
    *MATRICES_TOOLS,
    *CANONICAL_FORMS_TOOLS,
    *SYMBOLIC_MATRIX_TOOLS,
    *RATIONAL_LINEAR_TOOLS,
    *LATTICES_TOOLS,
    *FORMAL_POWER_SERIES_TOOLS,
    *POLYNOMIAL_TOOLS,
    *MULTIVARIATE_POLYNOMIAL_TOOLS,
    *ANALYSIS_TOOLS,
    *PROBABILITY_TOOLS,
    *OPTIMIZATION_TOOLS,
    *TOPOLOGY_TOOLS,
    *POSETS_TOOLS,
    *DIRECTED_GRAPH_TOOLS,
    *DISCREPANCY_THEORY_TOOLS,
    *GRAPH_POLYNOMIALS_TOOLS,
    *NUMERICAL_SEMIGROUPS_TOOLS,
    *MULTIPLE_TESTING_TOOLS,
    *EXACT_GEOMETRY_TOOLS,
    *ARITHMETIC_COUNTING_TOOLS,
    *GRAPH_REALIZATION_TOOLS,
    *BOOLEAN_ANALYSIS_TOOLS,
    *ARITHMETIC_FUNCTIONS_TOOLS,
    *ADDITIVE_COMBINATORICS_TOOLS,
    *MATRIX_ANALYSIS_TOOLS,
    *CONVEX_ANALYSIS_TOOLS,
    *SUBMODULAR_OPT_TOOLS,
    *POLYNOMIAL_MAPS_TOOLS,
    *EUCLIDEAN_GEOMETRY_TOOLS,
    *FINITE_GAME_THEORY_TOOLS,
    *IMPARTIAL_GAMES_TOOLS,
    *WORDS_TOOLS,
    *SYMBOLIC_DYNAMICS_TOOLS,
    *FINITE_TOPOLOGY_TOOLS,
    *ELECTRICAL_NETWORKS_TOOLS,
    *REGULAR_LANGUAGES_TOOLS,
    *ALGEBRAIC_COMBINATORICS_TOOLS,
    *REAL_ALGEBRA_TOOLS,
    *FINITE_METRIC_SPACES_TOOLS,
    *PETRI_NET_TOOLS,
    *TERM_REWRITING_TOOLS,
    *TREE_AUTOMATA_TOOLS,
    *GRAPHICAL_MODEL_TOOLS,
    *FINITE_STATE_TRANSDUCER_TOOLS,
)

_RAW_ADMISSIONS: tuple[OperationAdmission, ...] = (
    *ADDITIVE_COMBINATORICS_ADMISSIONS,
    *ALGEBRAIC_COMBINATORICS_ADMISSIONS,
    *ANALYSIS_ADMISSIONS,
    *ARITHMETIC_ADMISSIONS,
    *ARITHMETIC_COUNTING_ADMISSIONS,
    *ARITHMETIC_DYNAMICS_ADMISSIONS,
    *ARITHMETIC_FUNCTIONS_ADMISSIONS,
    *BOOLEAN_ADMISSIONS,
    *BOOLEAN_ANALYSIS_ADMISSIONS,
    *CODE_THEORY_ADMISSIONS,
    *COMBINATORICS_ADMISSIONS,
    *CONVEX_ANALYSIS_ADMISSIONS,
    *DIOPHANTINE_APPROXIMATION_ADMISSIONS,
    *DISCREPANCY_THEORY_ADMISSIONS,
    *ELECTRICAL_NETWORKS_ADMISSIONS,
    *FINITE_FIELDS_ADMISSIONS,
    *FINITE_GAME_THEORY_ADMISSIONS,
    *FINITE_METRIC_SPACES_ADMISSIONS,
    *FINITE_SETS_ADMISSIONS,
    *FINITE_STATE_TRANSDUCERS_ADMISSIONS,
    *FINITE_TOPOLOGY_ADMISSIONS,
    *FORMAL_POWER_SERIES_ADMISSIONS,
    *GEOMETRY_ADMISSIONS,
    *GEOMETRY_EUCLIDEAN_ADMISSIONS,
    *GEOMETRY_EXACT_ADMISSIONS,
    *GEOMETRY_PROJECTIVE_ADMISSIONS,
    *GRAPHICAL_MODELS_ADMISSIONS,
    *GRAPHS_ADMISSIONS,
    *GRAPHS_COLORING_ADMISSIONS,
    *GRAPHS_DECOMPOSITION_ADMISSIONS,
    *GRAPHS_DIRECTED_ADMISSIONS,
    *GRAPHS_FLOW_ADMISSIONS,
    *GRAPHS_ISOMORPHISM_ADMISSIONS,
    *GRAPHS_OPTIMIZATION_ADMISSIONS,
    *GRAPHS_POLYNOMIALS_ADMISSIONS,
    *GRAPHS_REALIZATION_ADMISSIONS,
    *GRAPHS_SPECTRAL_ADMISSIONS,
    *GRAPHS_SYMMETRY_ADMISSIONS,
    *GRAPHS_TRANSFORMS_ADMISSIONS,
    *GROUP_ADMISSIONS,
    *IMPARTIAL_GAMES_ADMISSIONS,
    *LATTICES_ADMISSIONS,
    *LOGIC_ADMISSIONS,
    *MARKOV_CHAIN_ADMISSIONS,
    *MATRICES_ADMISSIONS,
    *MATRICES_ANALYSIS_ADMISSIONS,
    *MATRICES_CANONICAL_FORMS_ADMISSIONS,
    *MATRICES_CERTIFIED_SNF_ADMISSIONS,
    *MATRICES_RATIONAL_LINEAR_ADMISSIONS,
    *MATRICES_SYMBOLIC_ADMISSIONS,
    *MULTIPLE_TESTING_ADMISSIONS,
    *NUMBER_FIELD_ADMISSIONS,
    *NUMBER_THEORY_ADMISSIONS,
    *NUMERICAL_SEMIGROUPS_ADMISSIONS,
    *OPTIMIZATION_ADMISSIONS,
    *PETRI_NETS_ADMISSIONS,
    *POLYNOMIALS_ADMISSIONS,
    *POLYNOMIALS_MAPS_ADMISSIONS,
    *POLYNOMIALS_MULTIVARIATE_ADMISSIONS,
    *POLYNOMIALS_REAL_ALGEBRA_ADMISSIONS,
    *POSETS_ADMISSIONS,
    *PROBABILITY_ADMISSIONS,
    *RECURRENCE_SOLVING_ADMISSIONS,
    *REGULAR_LANGUAGES_ADMISSIONS,
    *ROOT_ISOLATION_ADMISSIONS,
    *SEQUENCES_ADMISSIONS,
    *SUBMODULAR_OPT_ADMISSIONS,
    *SYMBOLIC_DYNAMICS_ADMISSIONS,
    *TERM_REWRITING_ADMISSIONS,
    *TOPOLOGY_ADMISSIONS,
    *TREE_AUTOMATA_ADMISSIONS,
    *WORDS_ADMISSIONS,
)
_ALL_ADMISSIONS: tuple[OperationAdmission, ...] = tuple(
    sorted(_RAW_ADMISSIONS, key=lambda r: r.operation_id)
)

BUILTIN_TOOLS: MathTools = curate_public_tools(_BUILTIN_CANDIDATES, _ALL_ADMISSIONS)

__all__ = ["BUILTIN_TOOLS"]
