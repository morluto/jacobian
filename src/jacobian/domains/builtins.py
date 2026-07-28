"""Explicit built-in domain portfolio."""

from jacobian.domains.analysis import REAL_ANALYSIS_BUNDLE
from jacobian.domains.arithmetic import ARITHMETIC_BUNDLE
from jacobian.domains.combinatorics import COMBINATORICS_BUNDLE
from jacobian.domains.finite_sets import FINITE_SET_BUNDLE
from jacobian.domains.geometry import GEOMETRY_BUNDLE
from jacobian.domains.graph_optimization import (
    GRAPH_INVARIANT_BUNDLE,
    GRAPH_OPTIMIZATION_BUNDLE,
)
from jacobian.domains.matrix_lattice import LATTICE_BUNDLE, MATRIX_BUNDLE
from jacobian.domains.number_theory import NUMBER_THEORY_BUNDLE
from jacobian.domains.optimization import RATIONAL_OPTIMIZATION_BUNDLE
from jacobian.domains.polynomial import POLYNOMIAL_BUNDLE
from jacobian.domains.probability import FINITE_PROBABILITY_BUNDLE
from jacobian.domains.projective_geometry import PROJECTIVE_GEOMETRY_BUNDLE
from jacobian.domains.sequences import SEQUENCE_BUNDLE

BUILTIN_DOMAIN_BUNDLES = (
    ARITHMETIC_BUNDLE,
    NUMBER_THEORY_BUNDLE,
    COMBINATORICS_BUNDLE,
    FINITE_SET_BUNDLE,
    SEQUENCE_BUNDLE,
    GEOMETRY_BUNDLE,
    PROJECTIVE_GEOMETRY_BUNDLE,
    GRAPH_OPTIMIZATION_BUNDLE,
    GRAPH_INVARIANT_BUNDLE,
    MATRIX_BUNDLE,
    LATTICE_BUNDLE,
    POLYNOMIAL_BUNDLE,
    REAL_ANALYSIS_BUNDLE,
    FINITE_PROBABILITY_BUNDLE,
    RATIONAL_OPTIMIZATION_BUNDLE,
)

__all__ = ["BUILTIN_DOMAIN_BUNDLES"]
