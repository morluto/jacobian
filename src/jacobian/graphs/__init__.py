"""Graph construction, property, coloring, and composition capabilities."""

from jacobian.graphs.coloring import (
    GraphColoringInstallation,
    install_graph_coloring_capabilities,
)
from jacobian.graphs.composition import (
    GraphCompositionInstallation,
    install_graph_composition_capabilities,
)
from jacobian.graphs.installation import GraphInstallation, install_graph_capabilities
from jacobian.graphs.isomorphism import (
    GraphIsomorphismInstallation,
    install_graph_isomorphism,
)

__all__ = [
    "GraphColoringInstallation",
    "GraphCompositionInstallation",
    "GraphInstallation",
    "GraphIsomorphismInstallation",
    "install_graph_capabilities",
    "install_graph_coloring_capabilities",
    "install_graph_composition_capabilities",
    "install_graph_isomorphism",
]
