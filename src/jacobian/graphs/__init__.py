"""Graph construction, property, and composition operations."""

from jacobian.graphs.composition import (
    GraphCompositionInstallation,
    install_graph_composition_operations,
)
from jacobian.graphs.installation import GraphInstallation, install_graph_operations
from jacobian.graphs.isomorphism import (
    GraphIsomorphismInstallation,
    install_graph_isomorphism,
)

__all__ = [
    "GraphCompositionInstallation",
    "GraphInstallation",
    "GraphIsomorphismInstallation",
    "install_graph_composition_operations",
    "install_graph_isomorphism",
    "install_graph_operations",
]
