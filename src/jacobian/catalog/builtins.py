"""Explicit immutable inventory of built-in mathematical domains."""

from __future__ import annotations

from jacobian.catalog.admission import OperationAdmission, curate_public_tools
from jacobian.catalog.models import MathTools
from jacobian.math.additive_combinatorics._admission import (
    REGISTRATION as ADDITIVE_COMBINATORICS_REGISTRATION,
)
from jacobian.math.algebraic_combinatorics._admission import (
    REGISTRATION as ALGEBRAIC_COMBINATORICS_REGISTRATION,
)
from jacobian.math.analysis._admission import REGISTRATION as ANALYSIS_REGISTRATION
from jacobian.math.arithmetic._admission import REGISTRATION as ARITHMETIC_REGISTRATION
from jacobian.math.arithmetic_counting._admission import (
    REGISTRATION as ARITHMETIC_COUNTING_REGISTRATION,
)
from jacobian.math.arithmetic_dynamics._admission import (
    REGISTRATION as ARITHMETIC_DYNAMICS_REGISTRATION,
)
from jacobian.math.arithmetic_functions._admission import (
    REGISTRATION as ARITHMETIC_FUNCTIONS_REGISTRATION,
)
from jacobian.math.boolean._admission import REGISTRATION as BOOLEAN_REGISTRATION
from jacobian.math.boolean_analysis._admission import (
    REGISTRATION as BOOLEAN_ANALYSIS_REGISTRATION,
)
from jacobian.math.code_theory._admission import (
    REGISTRATION as CODE_THEORY_REGISTRATION,
)
from jacobian.math.combinatorial_maps._admission import (
    REGISTRATION as COMBINATORIAL_MAPS_REGISTRATION,
)
from jacobian.math.combinatorics._admission import (
    REGISTRATION as COMBINATORICS_REGISTRATION,
)
from jacobian.math.commutative_algebra_ops._admission import (
    REGISTRATION as COMMUTATIVE_ALGEBRA_OPS_REGISTRATION,
)
from jacobian.math.convex_analysis._admission import (
    REGISTRATION as CONVEX_ANALYSIS_REGISTRATION,
)
from jacobian.math.diophantine_approximation._admission import (
    REGISTRATION as DIOPHANTINE_APPROXIMATION_REGISTRATION,
)
from jacobian.math.discrepancy_theory._admission import (
    REGISTRATION as DISCREPANCY_THEORY_REGISTRATION,
)
from jacobian.math.electrical_networks._admission import (
    REGISTRATION as ELECTRICAL_NETWORKS_REGISTRATION,
)
from jacobian.math.finite_categories._admission import (
    REGISTRATION as FINITE_CATEGORIES_REGISTRATION,
)
from jacobian.math.finite_fields._admission import (
    REGISTRATION as FINITE_FIELDS_REGISTRATION,
)
from jacobian.math.finite_game_theory._admission import (
    REGISTRATION as FINITE_GAME_THEORY_REGISTRATION,
)
from jacobian.math.finite_metric_spaces._admission import (
    REGISTRATION as FINITE_METRIC_SPACES_REGISTRATION,
)
from jacobian.math.finite_semigroups._admission import (
    REGISTRATION as FINITE_SEMIGROUPS_REGISTRATION,
)
from jacobian.math.finite_sets._admission import (
    REGISTRATION as FINITE_SETS_REGISTRATION,
)
from jacobian.math.finite_state_transducers._admission import (
    REGISTRATION as FINITE_STATE_TRANSDUCERS_REGISTRATION,
)
from jacobian.math.finite_topology._admission import (
    REGISTRATION as FINITE_TOPOLOGY_REGISTRATION,
)
from jacobian.math.formal_power_series._admission import (
    REGISTRATION as FORMAL_POWER_SERIES_REGISTRATION,
)
from jacobian.math.geometry._admission import REGISTRATION as GEOMETRY_REGISTRATION
from jacobian.math.geometry.euclidean._admission import (
    REGISTRATION as GEOMETRY_EUCLIDEAN_REGISTRATION,
)
from jacobian.math.geometry.exact._admission import (
    REGISTRATION as GEOMETRY_EXACT_REGISTRATION,
)
from jacobian.math.geometry.projective._admission import (
    REGISTRATION as GEOMETRY_PROJECTIVE_REGISTRATION,
)
from jacobian.math.graphical_models._admission import (
    REGISTRATION as GRAPHICAL_MODELS_REGISTRATION,
)
from jacobian.math.graphs._admission import REGISTRATION as GRAPHS_REGISTRATION
from jacobian.math.graphs.coloring._admission import (
    REGISTRATION as GRAPHS_COLORING_REGISTRATION,
)
from jacobian.math.graphs.decomposition._admission import (
    REGISTRATION as GRAPHS_DECOMPOSITION_REGISTRATION,
)
from jacobian.math.graphs.directed._admission import (
    REGISTRATION as GRAPHS_DIRECTED_REGISTRATION,
)
from jacobian.math.graphs.flow._admission import (
    REGISTRATION as GRAPHS_FLOW_REGISTRATION,
)
from jacobian.math.graphs.isomorphism._admission import (
    REGISTRATION as GRAPHS_ISOMORPHISM_REGISTRATION,
)
from jacobian.math.graphs.morphisms._admission import (
    REGISTRATION as GRAPHS_MORPHISMS_REGISTRATION,
)
from jacobian.math.graphs.optimization._admission import (
    REGISTRATION as GRAPHS_OPTIMIZATION_REGISTRATION,
)
from jacobian.math.graphs.polynomials._admission import (
    REGISTRATION as GRAPHS_POLYNOMIALS_REGISTRATION,
)
from jacobian.math.graphs.realization._admission import (
    REGISTRATION as GRAPHS_REALIZATION_REGISTRATION,
)
from jacobian.math.graphs.spectral._admission import (
    REGISTRATION as GRAPHS_SPECTRAL_REGISTRATION,
)
from jacobian.math.graphs.symmetry._admission import (
    REGISTRATION as GRAPHS_SYMMETRY_REGISTRATION,
)
from jacobian.math.graphs.transforms._admission import (
    REGISTRATION as GRAPHS_TRANSFORMS_REGISTRATION,
)
from jacobian.math.group._admission import REGISTRATION as GROUP_REGISTRATION
from jacobian.math.impartial_games._admission import (
    REGISTRATION as IMPARTIAL_GAMES_REGISTRATION,
)
from jacobian.math.lattices._admission import REGISTRATION as LATTICES_REGISTRATION
from jacobian.math.logic._admission import REGISTRATION as LOGIC_REGISTRATION
from jacobian.math.markov_chain._admission import (
    REGISTRATION as MARKOV_CHAIN_REGISTRATION,
)
from jacobian.math.matrices._admission import REGISTRATION as MATRICES_REGISTRATION
from jacobian.math.matrices.analysis._admission import (
    REGISTRATION as MATRICES_ANALYSIS_REGISTRATION,
)
from jacobian.math.matrices.canonical_forms._admission import (
    REGISTRATION as MATRICES_CANONICAL_FORMS_REGISTRATION,
)
from jacobian.math.matrices.certified_snf._admission import (
    REGISTRATION as MATRICES_CERTIFIED_SNF_REGISTRATION,
)
from jacobian.math.matrices.rational_linear._admission import (
    REGISTRATION as MATRICES_RATIONAL_LINEAR_REGISTRATION,
)
from jacobian.math.matrices.symbolic._admission import (
    REGISTRATION as MATRICES_SYMBOLIC_REGISTRATION,
)
from jacobian.math.multiple_testing._admission import (
    REGISTRATION as MULTIPLE_TESTING_REGISTRATION,
)
from jacobian.math.number_field._admission import (
    REGISTRATION as NUMBER_FIELD_REGISTRATION,
)
from jacobian.math.number_theory._admission import (
    REGISTRATION as NUMBER_THEORY_REGISTRATION,
)
from jacobian.math.numerical_semigroups._admission import (
    REGISTRATION as NUMERICAL_SEMIGROUPS_REGISTRATION,
)
from jacobian.math.optimization._admission import (
    REGISTRATION as OPTIMIZATION_REGISTRATION,
)
from jacobian.math.petri_nets._admission import REGISTRATION as PETRI_NETS_REGISTRATION
from jacobian.math.polynomials._admission import (
    REGISTRATION as POLYNOMIALS_REGISTRATION,
)
from jacobian.math.polynomials.maps._admission import (
    REGISTRATION as POLYNOMIALS_MAPS_REGISTRATION,
)
from jacobian.math.polynomials.multivariate._admission import (
    REGISTRATION as POLYNOMIALS_MULTIVARIATE_REGISTRATION,
)
from jacobian.math.polynomials.real_algebra._admission import (
    REGISTRATION as POLYNOMIALS_REAL_ALGEBRA_REGISTRATION,
)
from jacobian.math.posets._admission import REGISTRATION as POSETS_REGISTRATION
from jacobian.math.probability._admission import (
    REGISTRATION as PROBABILITY_REGISTRATION,
)
from jacobian.math.projective_coords_ops._admission import (
    REGISTRATION as PROJECTIVE_COORDS_OPS_REGISTRATION,
)
from jacobian.math.recurrence_solving._admission import (
    REGISTRATION as RECURRENCE_SOLVING_REGISTRATION,
)
from jacobian.math.regular_languages._admission import (
    REGISTRATION as REGULAR_LANGUAGES_REGISTRATION,
)
from jacobian.math.root_isolation._admission import (
    REGISTRATION as ROOT_ISOLATION_REGISTRATION,
)
from jacobian.math.root_systems._admission import (
    REGISTRATION as ROOT_SYSTEMS_REGISTRATION,
)
from jacobian.math.sequences._admission import REGISTRATION as SEQUENCES_REGISTRATION
from jacobian.math.submodular_opt._admission import (
    REGISTRATION as SUBMODULAR_OPT_REGISTRATION,
)
from jacobian.math.symbolic_dynamics._admission import (
    REGISTRATION as SYMBOLIC_DYNAMICS_REGISTRATION,
)
from jacobian.math.term_rewriting._admission import (
    REGISTRATION as TERM_REWRITING_REGISTRATION,
)
from jacobian.math.topology._admission import REGISTRATION as TOPOLOGY_REGISTRATION
from jacobian.math.tree_automata._admission import (
    REGISTRATION as TREE_AUTOMATA_REGISTRATION,
)
from jacobian.math.words._admission import REGISTRATION as WORDS_REGISTRATION

_BUILTIN_REGISTRATIONS = (
    ADDITIVE_COMBINATORICS_REGISTRATION,
    ALGEBRAIC_COMBINATORICS_REGISTRATION,
    ANALYSIS_REGISTRATION,
    ARITHMETIC_REGISTRATION,
    ARITHMETIC_COUNTING_REGISTRATION,
    ARITHMETIC_DYNAMICS_REGISTRATION,
    ARITHMETIC_FUNCTIONS_REGISTRATION,
    BOOLEAN_REGISTRATION,
    BOOLEAN_ANALYSIS_REGISTRATION,
    CODE_THEORY_REGISTRATION,
    COMBINATORIAL_MAPS_REGISTRATION,
    COMBINATORICS_REGISTRATION,
    COMMUTATIVE_ALGEBRA_OPS_REGISTRATION,
    CONVEX_ANALYSIS_REGISTRATION,
    DIOPHANTINE_APPROXIMATION_REGISTRATION,
    DISCREPANCY_THEORY_REGISTRATION,
    ELECTRICAL_NETWORKS_REGISTRATION,
    FINITE_CATEGORIES_REGISTRATION,
    FINITE_FIELDS_REGISTRATION,
    FINITE_GAME_THEORY_REGISTRATION,
    FINITE_METRIC_SPACES_REGISTRATION,
    FINITE_SEMIGROUPS_REGISTRATION,
    FINITE_SETS_REGISTRATION,
    FINITE_STATE_TRANSDUCERS_REGISTRATION,
    FINITE_TOPOLOGY_REGISTRATION,
    FORMAL_POWER_SERIES_REGISTRATION,
    GEOMETRY_REGISTRATION,
    GEOMETRY_EUCLIDEAN_REGISTRATION,
    GEOMETRY_EXACT_REGISTRATION,
    GEOMETRY_PROJECTIVE_REGISTRATION,
    GRAPHICAL_MODELS_REGISTRATION,
    GRAPHS_REGISTRATION,
    GRAPHS_COLORING_REGISTRATION,
    GRAPHS_DECOMPOSITION_REGISTRATION,
    GRAPHS_DIRECTED_REGISTRATION,
    GRAPHS_FLOW_REGISTRATION,
    GRAPHS_ISOMORPHISM_REGISTRATION,
    GRAPHS_MORPHISMS_REGISTRATION,
    GRAPHS_OPTIMIZATION_REGISTRATION,
    GRAPHS_POLYNOMIALS_REGISTRATION,
    GRAPHS_REALIZATION_REGISTRATION,
    GRAPHS_SPECTRAL_REGISTRATION,
    GRAPHS_SYMMETRY_REGISTRATION,
    GRAPHS_TRANSFORMS_REGISTRATION,
    GROUP_REGISTRATION,
    IMPARTIAL_GAMES_REGISTRATION,
    LATTICES_REGISTRATION,
    LOGIC_REGISTRATION,
    MARKOV_CHAIN_REGISTRATION,
    MATRICES_REGISTRATION,
    MATRICES_ANALYSIS_REGISTRATION,
    MATRICES_CANONICAL_FORMS_REGISTRATION,
    MATRICES_CERTIFIED_SNF_REGISTRATION,
    MATRICES_RATIONAL_LINEAR_REGISTRATION,
    MATRICES_SYMBOLIC_REGISTRATION,
    MULTIPLE_TESTING_REGISTRATION,
    NUMBER_FIELD_REGISTRATION,
    NUMBER_THEORY_REGISTRATION,
    NUMERICAL_SEMIGROUPS_REGISTRATION,
    OPTIMIZATION_REGISTRATION,
    PETRI_NETS_REGISTRATION,
    POLYNOMIALS_REGISTRATION,
    POLYNOMIALS_MAPS_REGISTRATION,
    POLYNOMIALS_MULTIVARIATE_REGISTRATION,
    POLYNOMIALS_REAL_ALGEBRA_REGISTRATION,
    POSETS_REGISTRATION,
    PROBABILITY_REGISTRATION,
    PROJECTIVE_COORDS_OPS_REGISTRATION,
    RECURRENCE_SOLVING_REGISTRATION,
    REGULAR_LANGUAGES_REGISTRATION,
    ROOT_ISOLATION_REGISTRATION,
    ROOT_SYSTEMS_REGISTRATION,
    SEQUENCES_REGISTRATION,
    SUBMODULAR_OPT_REGISTRATION,
    SYMBOLIC_DYNAMICS_REGISTRATION,
    TERM_REWRITING_REGISTRATION,
    TOPOLOGY_REGISTRATION,
    TREE_AUTOMATA_REGISTRATION,
    WORDS_REGISTRATION,
)

_BUILTIN_CANDIDATES: MathTools = tuple(
    tool for registration in _BUILTIN_REGISTRATIONS for tool in registration.candidates
)
_RAW_ADMISSIONS: tuple[OperationAdmission, ...] = tuple(
    admission
    for registration in _BUILTIN_REGISTRATIONS
    for admission in registration.admissions
)
_ALL_ADMISSIONS: tuple[OperationAdmission, ...] = tuple(
    sorted(_RAW_ADMISSIONS, key=lambda admission: admission.operation_id)
)

BUILTIN_TOOLS: MathTools = curate_public_tools(_BUILTIN_CANDIDATES, _ALL_ADMISSIONS)

__all__ = ["BUILTIN_TOOLS"]
