"""Explicit immutable inventory of built-in mathematical tools."""

from __future__ import annotations

from jacobian.catalog.models import MathTools
from jacobian.math.additive_combinatorics._tools import (
    TOOLS as ADDITIVE_COMBINATORICS_TOOLS,
)
from jacobian.math.algebraic_combinatorics._tools import (
    TOOLS as ALGEBRAIC_COMBINATORICS_TOOLS,
)
from jacobian.math.analysis._tools import TOOLS as ANALYSIS_TOOLS
from jacobian.math.arithmetic._tools import TOOLS as ARITHMETIC_TOOLS
from jacobian.math.arithmetic_counting._tools import TOOLS as ARITHMETIC_COUNTING_TOOLS
from jacobian.math.arithmetic_functions._tools import (
    TOOLS as ARITHMETIC_FUNCTIONS_TOOLS,
)
from jacobian.math.boolean._tools import TOOLS as BOOLEAN_TOOLS
from jacobian.math.boolean_analysis._tools import TOOLS as BOOLEAN_ANALYSIS_TOOLS
from jacobian.math.canonical_forms._tools import TOOLS as CANONICAL_FORMS_TOOLS
from jacobian.math.certified_snf._tools import TOOLS as CERTIFIED_SNF_TOOLS
from jacobian.math.code_theory._tools import TOOLS as CODE_THEORY_TOOLS
from jacobian.math.combinatorics._tools import TOOLS as COMBINATORICS_TOOLS
from jacobian.math.convex_analysis._tools import TOOLS as CONVEX_ANALYSIS_TOOLS
from jacobian.math.diophantine_approximation._tools import (
    TOOLS as DIOPHANTINE_APPROXIMATION_TOOLS,
)
from jacobian.math.directed_graph._tools import TOOLS as DIRECTED_GRAPH_TOOLS
from jacobian.math.discrepancy_theory._tools import TOOLS as DISCREPANCY_THEORY_TOOLS
from jacobian.math.electrical_networks._tools import (
    TOOLS as ELECTRICAL_NETWORKS_TOOLS,
)
from jacobian.math.euclidean_geometry._tools import TOOLS as EUCLIDEAN_GEOMETRY_TOOLS
from jacobian.math.exact_geometry._tools import TOOLS as EXACT_GEOMETRY_TOOLS
from jacobian.math.finite_fields._tools import TOOLS as FINITE_FIELDS_TOOLS
from jacobian.math.finite_game_theory._tools import TOOLS as FINITE_GAME_THEORY_TOOLS
from jacobian.math.finite_metric_spaces._tools import (
    TOOLS as FINITE_METRIC_SPACES_TOOLS,
)
from jacobian.math.finite_sets._tools import TOOLS as FINITE_SETS_TOOLS
from jacobian.math.formal_power_series._tools import TOOLS as FORMAL_POWER_SERIES_TOOLS
from jacobian.math.geometry._tools import TOOLS as GEOMETRY_TOOLS
from jacobian.math.graph_coloring_ops._tools import TOOLS as GRAPH_COLORING_OPS_TOOLS
from jacobian.math.graph_decomposition._tools import TOOLS as GRAPH_DECOMPOSITION_TOOLS
from jacobian.math.graph_flow._tools import TOOLS as GRAPH_FLOW_TOOLS
from jacobian.math.graph_isomorphism._tools import TOOLS as GRAPH_ISOMORPHISM_TOOLS
from jacobian.math.graph_optimization._tools import TOOLS as GRAPH_OPTIMIZATION_TOOLS
from jacobian.math.graph_polynomials._tools import TOOLS as GRAPH_POLYNOMIALS_TOOLS
from jacobian.math.graph_realization._tools import TOOLS as GRAPH_REALIZATION_TOOLS
from jacobian.math.graph_spectral._tools import TOOLS as GRAPH_SPECTRAL_TOOLS
from jacobian.math.graph_symmetry._tools import TOOLS as GRAPH_SYMMETRY_TOOLS
from jacobian.math.graph_transforms._tools import TOOLS as GRAPH_TRANSFORMS_TOOLS
from jacobian.math.graphs._tools import TOOLS as GRAPHS_TOOLS
from jacobian.math.group._tools import TOOLS as GROUP_TOOLS
from jacobian.math.lattices._tools import TOOLS as LATTICES_TOOLS
from jacobian.math.logic._tools import TOOLS as LOGIC_TOOLS
from jacobian.math.markov_chain._tools import TOOLS as MARKOV_CHAIN_TOOLS
from jacobian.math.matrices._tools import TOOLS as MATRICES_TOOLS
from jacobian.math.matrix_analysis._tools import TOOLS as MATRIX_ANALYSIS_TOOLS
from jacobian.math.multiple_testing._tools import TOOLS as MULTIPLE_TESTING_TOOLS
from jacobian.math.multivariate_polynomial._tools import (
    TOOLS as MULTIVARIATE_POLYNOMIAL_TOOLS,
)
from jacobian.math.number_field._tools import TOOLS as NUMBER_FIELD_TOOLS
from jacobian.math.number_theory._tools import TOOLS as NUMBER_THEORY_TOOLS
from jacobian.math.numerical_semigroups._tools import (
    TOOLS as NUMERICAL_SEMIGROUPS_TOOLS,
)
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS
from jacobian.math.polynomial._tools import TOOLS as POLYNOMIAL_TOOLS
from jacobian.math.polynomial_maps._tools import TOOLS as POLYNOMIAL_MAPS_TOOLS
from jacobian.math.posets._tools import TOOLS as POSETS_TOOLS
from jacobian.math.probability._tools import TOOLS as PROBABILITY_TOOLS
from jacobian.math.projective_geometry._tools import TOOLS as PROJECTIVE_GEOMETRY_TOOLS
from jacobian.math.rational_linear._tools import TOOLS as RATIONAL_LINEAR_TOOLS
from jacobian.math.real_algebra._tools import TOOLS as REAL_ALGEBRA_TOOLS
from jacobian.math.recurrence_solving._tools import TOOLS as RECURRENCE_SOLVING_TOOLS
from jacobian.math.regular_languages._tools import TOOLS as REGULAR_LANGUAGES_TOOLS
from jacobian.math.root_isolation._tools import TOOLS as ROOT_ISOLATION_TOOLS
from jacobian.math.sequences._tools import TOOLS as SEQUENCES_TOOLS
from jacobian.math.submodular_opt._tools import TOOLS as SUBMODULAR_OPT_TOOLS
from jacobian.math.symbolic_matrix._tools import TOOLS as SYMBOLIC_MATRIX_TOOLS
from jacobian.math.topology._tools import TOOLS as TOPOLOGY_TOOLS

BUILTIN_TOOLS: MathTools = (
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
    *ELECTRICAL_NETWORKS_TOOLS,
    *REGULAR_LANGUAGES_TOOLS,
    *ALGEBRAIC_COMBINATORICS_TOOLS,
    *REAL_ALGEBRA_TOOLS,
    *FINITE_METRIC_SPACES_TOOLS,
)

__all__ = ["BUILTIN_TOOLS"]
