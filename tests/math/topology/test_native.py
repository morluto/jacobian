"""Native topology surface: the simplicial chain producer composes with
the canonical chain-complex consumers unchanged."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import MathTool
from jacobian.math.chain_complexes._models import (
    ComputeHomologyRequest,
    TensorProductRequest,
)
from jacobian.math.chain_complexes.operations import (
    compute_homology,
    compute_tensor_product,
)
from jacobian.math.chain_complexes.values import CoefficientField
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexRequest,
    HomologyConvention,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)
from jacobian.math.topology._tools import TOOLS
from jacobian.math.topology.native import simplicial_chain_complex_value


def _operation(operation_id: str) -> MathTool:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _circle() -> object:
    request = SimplicialComplexRequest(
        vertices=("a", "b", "c"),
        facets=(("a", "b"), ("b", "c"), ("a", "c")),
    )
    return _operation("topology.simplicial_complex.canonicalize").run(request).complex


def _prime_field_chain(complex_: object, prime: int):
    return _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=prime,
            convention=HomologyConvention.UNREDUCED,
        )
    )


def test_gf_p_chain_result_enters_homology_unchanged() -> None:
    """A small GF(p) producer output passes through the consumer's typed
    boundary without caller-side reconstruction."""
    value = simplicial_chain_complex_value(_prime_field_chain(_circle(), 2))
    assert value.coefficient_field is CoefficientField.PRIME_FIELD
    assert value.prime == 2
    assert value.basis_sizes == (3, 3)

    result = compute_homology(ComputeHomologyRequest(complex=value))
    assert [(group.degree, group.betti_number) for group in result.homology_groups] == [
        (0, 1),
        (1, 1),
    ]


def test_converted_profile_matches_topology_homology_producer() -> None:
    """The composed profile agrees with the topology domain's own exact
    homology of the same complex over the same field."""
    complex_ = _circle()
    value = simplicial_chain_complex_value(_prime_field_chain(complex_, 3))
    composed = compute_homology(ComputeHomologyRequest(complex=value))
    topology = _operation("topology.simplicial_homology.compute").run(
        SimplicialHomologyRequest(complex=complex_, prime=3)
    )
    assert (
        [group.betti_number for group in composed.homology_groups]
        == [group.betti_number for group in topology.groups]
        == [1, 1]
    )


def test_serialized_value_round_trips_into_consumers() -> None:
    """The converted value survives serialization and keeps composing."""
    from jacobian.math.chain_complexes.values import ChainComplexValue

    payload = simplicial_chain_complex_value(_prime_field_chain(_circle(), 2))
    value = ChainComplexValue.model_validate(payload.model_dump())
    result = compute_homology(ComputeHomologyRequest(complex=value))
    assert result.homology_groups[1].betti_number == 1
    tensor = compute_tensor_product(TensorProductRequest(left=value, right=value))
    assert tensor.value.basis_sizes == (9, 18, 9)


def test_point_complex_degenerate_value_keeps_full_context() -> None:
    request = SimplicialComplexRequest(vertices=("a",), facets=(("a",),))
    complex_ = (
        _operation("topology.simplicial_complex.canonicalize").run(request).complex
    )
    value = simplicial_chain_complex_value(_prime_field_chain(complex_, 5))
    assert value.basis_sizes == (1,)
    assert value.differential_matrices == ()
    assert (value.degree_min, value.degree_max) == (0, 0)
    result = compute_homology(ComputeHomologyRequest(complex=value))
    assert result.homology_groups[0].betti_number == 1


def test_integral_ring_is_outside_the_canonical_domain() -> None:
    integral = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=_circle(),
            coefficient_ring=ChainCoefficientRing.INTEGER,
            convention=HomologyConvention.UNREDUCED,
        )
    )
    with pytest.raises(ValueError, match="prime-field"):
        simplicial_chain_complex_value(integral)


def test_reduced_chains_carry_an_unrepresentable_augmentation() -> None:
    reduced = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=_circle(),
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
            convention=HomologyConvention.REDUCED,
        )
    )
    with pytest.raises(ValueError, match="augmentation"):
        simplicial_chain_complex_value(reduced)


def test_oversized_simplicial_groups_stay_outside_the_canonical_domain() -> None:
    """A producer result whose groups exceed the canonical value's bounds
    is rejected explicitly instead of failing inside construction."""
    from jacobian.math.topology._models import (
        MAX_TOPOLOGY_CHAIN_GROUP,
        SimplexBasis,
        SparseBoundaryMatrix,
    )
    from jacobian.math.topology._models import HomologyConvention as Convention

    oversized = type(_prime_field_chain(_circle(), 2))(
        complex_digest="sha256:" + "0" * 64,
        coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
        prime=2,
        convention=Convention.UNREDUCED,
        simplex_bases=(
            SimplexBasis(dimension=0, simplices=(("a",),) * MAX_TOPOLOGY_CHAIN_GROUP),
        ),
        boundary_matrices=(
            SparseBoundaryMatrix(
                source_dimension=0,
                target_dimension=-1,
                rows=0,
                columns=MAX_TOPOLOGY_CHAIN_GROUP,
            ),
        ),
        boundary_squared_zero=(),
    )
    with pytest.raises(ValueError, match="basis bound"):
        simplicial_chain_complex_value(oversized)
