"""The explicit built-in Jacobian mathematical portfolio."""

from __future__ import annotations

from collections.abc import Callable

from jacobian.domains.analysis import build_real_analysis_bundle
from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.certified_snf import build_certified_snf_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_fields import build_finite_field_bundle
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
from jacobian.domains.posets import build_finite_poset_bundle
from jacobian.domains.probability import build_finite_probability_bundle
from jacobian.domains.projective_geometry import build_projective_geometry_bundle
from jacobian.domains.rational_linear import build_rational_linear_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.domains.topology import build_topology_bundle
from jacobian.portfolio.model import PortfolioComponent, PortfolioPlan
from jacobian.portfolio.nullstellensatz_installation import (
    build_nullstellensatz_core_component,
    build_nullstellensatz_singular_component,
)

type PortfolioComponentFactory = Callable[[], PortfolioComponent]

BUILTIN_PORTFOLIO_COMPONENT_FACTORIES: tuple[PortfolioComponentFactory, ...] = (
    build_arithmetic_bundle,
    build_number_theory_bundle,
    build_combinatorics_bundle,
    build_finite_set_bundle,
    build_finite_field_bundle,
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
    build_nullstellensatz_core_component,
    build_nullstellensatz_singular_component,
    build_real_analysis_bundle,
    build_finite_probability_bundle,
    build_rational_optimization_bundle,
    build_topology_bundle,
    build_finite_poset_bundle,
)


def build_builtin_portfolio_components() -> tuple[PortfolioComponent, ...]:
    """Construct the ordered built-in portfolio without dynamic discovery."""

    return tuple(factory() for factory in BUILTIN_PORTFOLIO_COMPONENT_FACTORIES)


def build_builtin_portfolio() -> PortfolioPlan:
    """Build one fresh ordered portfolio from explicit component factories."""

    return PortfolioPlan(components=build_builtin_portfolio_components())


__all__ = [
    "BUILTIN_PORTFOLIO_COMPONENT_FACTORIES",
    "PortfolioComponentFactory",
    "build_builtin_portfolio",
    "build_builtin_portfolio_components",
]
