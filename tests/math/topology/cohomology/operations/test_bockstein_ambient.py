"""Zero Bocksteins retain their coefficient field with optional ambient data."""

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.cohomology.operations import bockstein
from jacobian.math.topology.cohomology.operations._models import (
    BocksteinRequest,
    BocksteinResult,
)
from jacobian.math.topology.cohomology.operations._tools import TOOLS
from jacobian.math.topology.operations import canonicalize


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("representation", ["multiple", "duplicates", "normalized"])
@pytest.mark.parametrize("ambient_kind", ["absent", "raw", "canonical"])
def test_zero_bockstein_preserves_source_and_codomain(
    prime: int, degree: int, representation: str, ambient_kind: str
) -> None:
    vertices = tuple(range(degree + 2))
    ambient = tuple(
        face
        for size in range(1, len(vertices) + 1)
        for face in combinations(vertices, size)
    )
    face = vertices[:-1]
    values = (face, face) if representation == "duplicates" else (face,)
    coefficients = {
        "multiple": (prime,),
        "duplicates": (1, prime - 1),
        "normalized": (0,),
    }[representation]
    # The represented source is zero in GF(p); its zero lift has d(0)=0,
    # so beta(0)=0 in degree n+1 for the same coefficient field.
    assert sum(coefficients) % prime == 0
    labels = tuple(str(vertex) for vertex in vertices)
    complex_value = canonicalize(labels, (labels,)).complex
    request = BocksteinRequest(
        prime=prime,
        cochain_degree=degree,
        simplex_values=values,
        simplex_coefficients=coefficients,
        ambient_simplices=ambient if ambient_kind == "raw" else (),
        ambient_complex=complex_value if ambient_kind == "canonical" else None,
    )
    native = bockstein(
        prime,
        degree,
        values,
        coefficients,
        request.ambient_simplices,
        request.ambient_complex,
    )
    tool = next(t for t in TOOLS if t.operation_id == "cohomology.bockstein.compute")
    dispatched = tool.run(
        BocksteinRequest.model_validate_json(request.model_dump_json(), strict=True)
    )
    for result in (native, dispatched):
        assert result.prime == prime
        assert result.cochain_degree == degree
        assert result.simplex_values == values
        assert result.simplex_coefficients == coefficients
        assert result.ambient_simplices == request.ambient_simplices
        assert result.ambient_complex == request.ambient_complex
        assert result.result_degree == degree + 1
        assert result.result_simplex_values == ()
        assert result.result_simplex_coefficients == ()
        assert result.is_zero
        assert (
            BocksteinResult.model_validate_json(result.model_dump_json(), strict=True)
            == result
        )


@pytest.mark.parametrize(
    ("ambient", "code"),
    [
        (((1,),), "support_outside_ambient"),
        (((0,), (0, 1)), "ambient_not_downward_closed"),
        (((0,), (1,), (1, 0)), "simplex_vertices_not_canonical"),
    ],
)
def test_zero_bockstein_still_validates_ambient_and_recovers(
    ambient: tuple[tuple[int, ...], ...], code: str
) -> None:
    with pytest.raises(OperationDomainValidationError) as excinfo:
        bockstein(3, 0, ((0,),), (3,), ambient_simplices=ambient)
    assert excinfo.value.errors()[0]["type"] == f"cohomology_operation.{code}"
    assert bockstein(
        3, 0, ((0,),), (3,), ambient_simplices=((0,), (1,), (0, 1))
    ).is_zero


def test_nonzero_mod_prime_is_rejected_even_when_zero_mod_two() -> None:
    with pytest.raises(OperationDomainValidationError) as excinfo:
        bockstein(3, 0, ((0,),), (2,), ambient_simplices=((0,),))
    assert excinfo.value.errors()[0]["type"] == (
        "cohomology_operation.nonzero_bockstein_unsupported"
    )
