"""Electrical network operations."""

from jacobian.math.electrical_networks.operations import (
    effective_resistance,
    laplacian_matrix,
    node_potentials,
)

__all__ = ["effective_resistance", "laplacian_matrix", "node_potentials"]
