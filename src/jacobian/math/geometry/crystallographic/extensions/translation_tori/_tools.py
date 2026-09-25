"""Public operation for checked parallelepiped translation tori."""

from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicFundamentalDomainResult,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori._models import (
    BieberbachTranslationTorusChains,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori.operations import (
    translation_torus_quotient_chains,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.translation_torus.quotient_chains.compute",
        title="Construct quotient chains of a translation 3-torus",
        description=(
            "Return the integral product cellular chain complex for a verified "
            "rank-three pure translation group with a parallelepiped fundamental "
            "domain. This bounded slice has eight vertices, six facets, and "
            "opposite facet translations; it does not construct general "
            "three-dimensional Bieberbach face orbits."
        ),
        request_type=CrystallographicFundamentalDomainResult,
        result_type=BieberbachTranslationTorusChains,
        run=translation_torus_quotient_chains,
        tags=("Bieberbach-group", "quotient-chains", "integral-homology", "exact"),
        discovery_terms=(
            "integral cellular chains of a three dimensional torus",
            "quotient homology of a crystallographic translation lattice",
            "Bieberbach translation group quotient chain complex",
        ),
        examples=(),
    ),
)

__all__ = ["TOOLS"]
