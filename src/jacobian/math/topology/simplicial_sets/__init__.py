"""Finite truncated simplicial sets with exact face/degeneracy tables."""

from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsResult,
    unnormalized_chains,
)
from jacobian.math.topology.simplicial_sets.complex_conversion import (
    simplicial_set_from_complex,
)
from jacobian.math.topology.simplicial_sets.complex_conversion_models import (
    ComplexFaceSimplexIndex,
    SimplicialComplexPrefixResult,
)
from jacobian.math.topology.simplicial_sets.coproduct import simplicial_set_coproduct
from jacobian.math.topology.simplicial_sets.coproduct_models import (
    SimplicialSetCoproductResult,
)
from jacobian.math.topology.simplicial_sets.degeneracy import (
    DegeneracyProfileResult,
    degeneracy_profile,
)
from jacobian.math.topology.simplicial_sets.image import (
    SimplicialMapImageResult,
    simplicial_map_image,
)
from jacobian.math.topology.simplicial_sets.maps import (
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    identity_simplicial_map,
    normalized_chains,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.product import simplicial_set_product
from jacobian.math.topology.simplicial_sets.product_models import (
    SimplicialSetProductResult,
)
from jacobian.math.topology.simplicial_sets.standard import (
    simplex_boundary,
    simplex_horn,
    standard_simplex,
)
from jacobian.math.topology.simplicial_sets.truncate import truncate_simplicial_set

__all__ = [
    "ComplexFaceSimplexIndex",
    "DegeneracyProfileResult",
    "FiniteTruncatedSimplicialSet",
    "SimplicialComplexPrefixResult",
    "SimplicialIdentityObstruction",
    "SimplicialMapImageResult",
    "SimplicialSetCoproductResult",
    "SimplicialSetProductResult",
    "SimplicialSetTablesResult",
    "TruncatedSimplicialMap",
    "UnnormalizedChainsResult",
    "compose_simplicial_maps",
    "degeneracy_profile",
    "from_tables",
    "identity_simplicial_map",
    "normalized_chains",
    "simplex_boundary",
    "simplex_horn",
    "simplicial_map",
    "simplicial_map_image",
    "simplicial_set_coproduct",
    "simplicial_set_from_complex",
    "simplicial_set_product",
    "standard_simplex",
    "truncate_simplicial_set",
    "unnormalized_chains",
]
