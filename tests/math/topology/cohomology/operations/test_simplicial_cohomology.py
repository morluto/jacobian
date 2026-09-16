"""Tests for prime-field simplicial cohomology with exact bases."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import HomologyConvention
from jacobian.math.topology.cohomology.operations._simplicial import (
    SimplicialCohomologyRequest,
    SimplicialCohomologyResult,
    simplicial_cohomology,
)
from jacobian.math.topology.cohomology.operations._tools import TOOLS
from jacobian.math.topology.operations import canonicalize


def _complex(vertices, facets):
    return canonicalize(tuple(vertices), tuple(facets)).complex


_CIRCLE = _complex(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
_POINT = _complex(["x"], [["x"]])
_EDGE = _complex(["a", "b"], [["a", "b"]])
_SPHERE = _complex(
    ["a", "b", "c", "d"],
    [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"], ["b", "c", "d"]],
)


def _torus():
    verts = [f"v{i}{j}" for i in range(3) for j in range(3)]
    facets = []
    for i in range(3):
        for j in range(3):
            a = f"v{i}{j}"
            b = f"v{(i + 1) % 3}{j}"
            c = f"v{i}{(j + 1) % 3}"
            d = f"v{(i + 1) % 3}{(j + 1) % 3}"
            facets.append([a, b, d])
            facets.append([a, d, c])
    return _complex(verts, facets)


def _wedge():
    return _complex(
        ["w", "a", "b", "c", "d", "p", "q", "r"],
        [
            ["w", "a"],
            ["a", "b"],
            ["b", "w"],
            ["w", "c"],
            ["c", "d"],
            ["d", "w"],
            ["w", "p", "q"],
            ["w", "p", "r"],
            ["w", "q", "r"],
            ["p", "q", "r"],
        ],
    )


def _bettis(result):
    return tuple(group.betti_number for group in result.groups)


def test_circle_known_answer_mod_two() -> None:
    result = simplicial_cohomology(_CIRCLE, 2, HomologyConvention.UNREDUCED)
    assert _bettis(result) == (1, 1)
    degree_one = result.groups[1]
    assert degree_one.cochain_dimension == 3
    assert degree_one.cocycle_dimension == 3
    assert degree_one.incoming_coboundary_rank == 2


def test_circle_known_answer_odd_prime() -> None:
    assert _bettis(simplicial_cohomology(_CIRCLE, 3, HomologyConvention.UNREDUCED)) == (
        1,
        1,
    )


def test_point_and_contractible_edge_are_acyclic() -> None:
    assert _bettis(simplicial_cohomology(_POINT, 2, HomologyConvention.UNREDUCED)) == (
        1,
    )
    assert _bettis(simplicial_cohomology(_EDGE, 2, HomologyConvention.UNREDUCED)) == (
        1,
        0,
    )


def test_sphere_top_class() -> None:
    assert _bettis(simplicial_cohomology(_SPHERE, 2, HomologyConvention.UNREDUCED)) == (
        1,
        0,
        1,
    )


def test_reduced_convention_kills_degree_zero() -> None:
    result = simplicial_cohomology(_CIRCLE, 2, HomologyConvention.REDUCED)
    assert _bettis(result) == (0, 1)


def test_torus_and_wedge_share_betti_numbers() -> None:
    """Torus and S1 v S1 v S2 agree on Betti numbers; cup products (later slice) separate them."""
    assert _bettis(
        simplicial_cohomology(_torus(), 2, HomologyConvention.UNREDUCED)
    ) == (1, 2, 1)
    assert _bettis(
        simplicial_cohomology(_wedge(), 2, HomologyConvention.UNREDUCED)
    ) == (1, 2, 1)
    assert _bettis(
        simplicial_cohomology(_torus(), 3, HomologyConvention.UNREDUCED)
    ) == (1, 2, 1)


def test_coboundary_square_zero_replay() -> None:
    """Recompute every coboundary from the complex and check d^{k+1} d^k = 0."""
    from jacobian.math.topology.cohomology.operations._simplicial import (
        _dense_boundary,
        _mat_mat_mod,
        _transpose,
    )

    prime = 2
    complex_ = _torus()
    boundaries = [
        _dense_boundary(complex_, degree, prime=prime)
        for degree in range(complex_.dimension + 1)
    ]
    coboundaries = [
        _transpose(boundaries[degree + 1]) if degree < complex_.dimension else []
        for degree in range(complex_.dimension + 1)
    ]
    for degree in range(complex_.dimension):
        if coboundaries[degree] and coboundaries[degree + 1]:
            product = _mat_mat_mod(
                coboundaries[degree + 1], coboundaries[degree], prime=prime
            )
            assert all(entry == 0 for row in product for entry in row)


def test_cocycle_and_coboundary_bases_satisfy_kernel_inclusions() -> None:
    from jacobian.math.topology.cohomology.operations._simplicial import (
        _dense_boundary,
        _mat_vec_mod,
        _transpose,
    )

    prime = 3
    complex_ = _torus()
    result = simplicial_cohomology(complex_, prime, HomologyConvention.UNREDUCED)
    boundaries = [
        _dense_boundary(complex_, degree, prime=prime)
        for degree in range(complex_.dimension + 1)
    ]
    for group in result.groups:
        outgoing = (
            _transpose(boundaries[group.dimension + 1])
            if group.dimension < complex_.dimension
            else []
        )
        for vector in (*group.cocycle_basis, *group.coboundary_basis):
            assert _mat_vec_mod(outgoing, vector.coefficients, prime=prime) == [
                0
            ] * len(outgoing)


def test_betti_numbers_agree_with_homology_operation() -> None:
    from jacobian.math.topology.operations import homology

    for prime in (2, 5):
        cohomology_result = simplicial_cohomology(
            _torus(), prime, HomologyConvention.UNREDUCED
        )
        homology_result = homology(_torus(), prime, HomologyConvention.UNREDUCED)
        assert _bettis(cohomology_result) == tuple(
            group.betti_number for group in homology_result.groups
        )


def test_non_prime_rejected() -> None:
    with pytest.raises(OperationDomainValidationError):
        simplicial_cohomology(_CIRCLE, 4, HomologyConvention.UNREDUCED)


def test_inconsistent_face_closure_rejected() -> None:
    broken = _CIRCLE.model_copy(
        update={"f_vector": (3, 2), "closure_size": 5},
    )
    with pytest.raises(OperationDomainValidationError):
        simplicial_cohomology(broken, 2, HomologyConvention.UNREDUCED)


def test_native_vs_catalog_parity_and_serialization() -> None:
    request = SimplicialCohomologyRequest(
        complex=_CIRCLE, prime=2, convention=HomologyConvention.UNREDUCED
    )
    tool = next(
        t for t in TOOLS if t.operation_id == "topology.simplicial.cohomology.compute"
    )
    native = simplicial_cohomology(request.complex, request.prime, request.convention)
    via_catalog = tool.run(request)
    assert native == via_catalog
    decoded = SimplicialCohomologyResult.model_validate_json(native.model_dump_json())
    assert decoded == native


def test_published_example_executes() -> None:
    tool = next(
        t for t in TOOLS if t.operation_id == "topology.simplicial.cohomology.compute"
    )
    assert tool.examples
    for example in tool.examples:
        request = SimplicialCohomologyRequest.model_validate(example.input)
        result = tool.run(request)
        assert isinstance(result, SimplicialCohomologyResult)
