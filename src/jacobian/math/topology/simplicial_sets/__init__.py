"""Finite truncated simplicial sets with exact face/degeneracy tables."""

from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsRequest,
    UnnormalizedChainsResult,
    unnormalized_chains,
)
from jacobian.math.topology.simplicial_sets.complex_conversion import (
    simplicial_set_from_complex,
)
from jacobian.math.topology.simplicial_sets.complex_conversion_models import (
    ComplexFaceSimplexIndex,
    SimplicialComplexPrefixRequest,
    SimplicialComplexPrefixResult,
)
from jacobian.math.topology.simplicial_sets.coproduct import simplicial_set_coproduct
from jacobian.math.topology.simplicial_sets.coproduct_models import (
    SimplicialSetCoproductRequest,
    SimplicialSetCoproductResult,
)
from jacobian.math.topology.simplicial_sets.degeneracy import (
    DegeneracyProfileResult,
    degeneracy_profile,
)
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    identity_simplicial_map,
    normalized_chains,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.product import simplicial_set_product
from jacobian.math.topology.simplicial_sets.product_models import (
    SimplicialSetProductRequest,
    SimplicialSetProductResult,
)
from jacobian.math.topology.simplicial_sets.standard import (
    simplex_boundary,
    simplex_horn,
    standard_simplex,
)
from jacobian.math.topology.simplicial_sets.truncate import truncate_simplicial_set
from jacobian.math.topology.simplicial_sets.truncate_models import (
    SimplicialSetTruncateRequest,
)

__all__ = [
    "ComplexFaceSimplexIndex",
    "DegeneracyProfileResult",
    "FiniteTruncatedSimplicialSet",
    "SimplicialComplexPrefixRequest",
    "SimplicialComplexPrefixResult",
    "SimplicialIdentityObstruction",
    "SimplicialMapCompositionRequest",
    "SimplicialSetCoproductRequest",
    "SimplicialSetCoproductResult",
    "SimplicialSetProductRequest",
    "SimplicialSetProductResult",
    "SimplicialSetTablesResult",
    "SimplicialSetTruncateRequest",
    "TruncatedSimplicialMap",
    "UnnormalizedChainsRequest",
    "UnnormalizedChainsResult",
    "compose_simplicial_maps",
    "degeneracy_profile",
    "from_tables",
    "identity_simplicial_map",
    "normalized_chains",
    "simplex_boundary",
    "simplex_horn",
    "simplicial_map",
    "simplicial_set_coproduct",
    "simplicial_set_from_complex",
    "simplicial_set_product",
    "standard_simplex",
    "truncate_simplicial_set",
    "unnormalized_chains",
]
