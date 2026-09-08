"""Native-only projection of exact pairwise distances onto indexed edge colours.

Constructing the complete hypergraph and colour indices from
``geometry.points.distance_profile.compute`` is not a distinct public
operation.
"""

from jacobian.catalog.models import MathTools

TOOLS: MathTools = ()
