"""Every native canonical-complex consumer admits the authored face ledger."""

from typing import Any

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology import operations
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    FiniteSimplicialComplex,
    HomologyConvention,
    simplicial_complex_digest,
)


@pytest.mark.parametrize("size", [3, 9])
@pytest.mark.parametrize(
    "consumer",
    [
        "pseudomanifold",
        "barycentric_subdivision",
        "shelling_check",
        "chain_complex",
        "homology",
        "integral_homology",
    ],
)
def test_native_consumers_reject_fabricated_canonical_closure(
    size: int, consumer: str
) -> None:
    vertices = tuple(f"v{i}" for i in range(size))
    points = operations.canonicalize(vertices, tuple((v,) for v in vertices)).complex
    authored: dict[str, Any] = {
        "vertices": vertices,
        "maximal_simplices": (vertices,),
        "faces_by_dimension": points.faces_by_dimension,
        "dimension": 0,
        "f_vector": (size,),
        "closure_size": size,
    }
    claim = FiniteSimplicialComplex(
        **authored, complex_digest=simplicial_complex_digest(**authored)
    )
    claim = FiniteSimplicialComplex.model_validate_json(claim.model_dump_json())
    with pytest.raises(OperationDomainValidationError):
        if consumer == "shelling_check":
            operations.shelling_check(claim, (0,))
        elif consumer == "chain_complex":
            operations.chain_complex(
                claim, ChainCoefficientRing.INTEGER, None, HomologyConvention.UNREDUCED
            )
        elif consumer == "homology":
            operations.homology(claim, 2, HomologyConvention.UNREDUCED)
        elif consumer == "integral_homology":
            operations.integral_homology(claim, HomologyConvention.UNREDUCED)
        else:
            getattr(operations, consumer)(claim)
