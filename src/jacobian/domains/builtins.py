"""Explicit factories for the built-in domain portfolio.

This is the only module allowed to know every built-in mathematical domain.
The tuple is deliberately literal and ordered: adding a domain changes this
composition root, without package scanning or import-time registration.
"""

from collections.abc import Callable

from jacobian.domains.analysis import build_real_analysis_bundle
from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.certified_snf import build_certified_snf_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.formal_datasets import build_formal_dataset_bundle
from jacobian.domains.geometry import build_geometry_bundle
from jacobian.domains.graph_optimization import (
    build_graph_invariant_bundle,
    build_graph_optimization_bundle,
)
from jacobian.domains.graph_symmetry import build_graph_symmetry_bundle
from jacobian.domains.matrix_lattice import build_lattice_bundle, build_matrix_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.optimization import build_rational_optimization_bundle
from jacobian.domains.polynomial import build_polynomial_bundle
from jacobian.domains.polynomial_nullstellensatz import (
    build_nullstellensatz_core_bundle,
    build_nullstellensatz_singular_bundle,
)
from jacobian.domains.posets import build_finite_poset_bundle
from jacobian.domains.probability import build_finite_probability_bundle
from jacobian.domains.projective_geometry import build_projective_geometry_bundle
from jacobian.domains.rational_linear import build_rational_linear_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.domains.topology import build_topology_bundle
from jacobian.operations import DomainBundle

type DomainBundleFactory = Callable[[], DomainBundle]

BUILTIN_DOMAIN_BUNDLE_FACTORIES: tuple[DomainBundleFactory, ...] = (
    build_arithmetic_bundle,
    build_number_theory_bundle,
    build_combinatorics_bundle,
    build_finite_set_bundle,
    build_formal_dataset_bundle,
    build_sequence_bundle,
    build_geometry_bundle,
    build_projective_geometry_bundle,
    build_graph_optimization_bundle,
    build_graph_invariant_bundle,
    build_graph_symmetry_bundle,
    build_certified_snf_bundle,
    build_matrix_bundle,
    build_rational_linear_bundle,
    build_lattice_bundle,
    build_polynomial_bundle,
    build_nullstellensatz_core_bundle,
    build_nullstellensatz_singular_bundle,
    build_real_analysis_bundle,
    build_finite_probability_bundle,
    build_rational_optimization_bundle,
    build_topology_bundle,
    build_finite_poset_bundle,
)


def build_builtin_domain_bundles() -> tuple[DomainBundle, ...]:
    """Construct the ordered built-in portfolio without import-time instances."""

    return tuple(factory() for factory in BUILTIN_DOMAIN_BUNDLE_FACTORIES)


__all__ = [
    "BUILTIN_DOMAIN_BUNDLE_FACTORIES",
    "DomainBundleFactory",
    "build_builtin_domain_bundles",
]
