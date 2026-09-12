"""Every native canonical-complex consumer admits the authored face ledger."""

import pytest

from jacobian.math.topology import operations
from jacobian.math.topology._models import FiniteSimplicialComplex


@pytest.mark.parametrize("size", [3, 9])
def test_canonical_value_rejects_fabricated_face_closure(size: int) -> None:
    vertices = tuple(f"v{i}" for i in range(size))
    points = operations.canonicalize(vertices, tuple((v,) for v in vertices)).complex
    authored = {
        "vertices": vertices,
        "maximal_simplices": (vertices,),
        "faces_by_dimension": points.faces_by_dimension,
        "dimension": 0,
        "f_vector": (size,),
        "closure_size": size,
    }
    with pytest.raises(ValueError):
        FiniteSimplicialComplex.model_validate(authored)
