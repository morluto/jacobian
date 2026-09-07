"""Native topology surface: the simplicial chain producer composes with
the canonical chain-complex consumers unchanged."""

from __future__ import annotations

from typing import Any, Literal, cast, overload

import pytest

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.topology._homology import (
    IntegralSimplicialHomologyRequest,
    IntegralSimplicialHomologyResult,
    SimplicialHomologyRequest,
    SimplicialHomologyResult,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexRequest,
    ChainComplexResult,
    FiniteSimplicialComplex,
    HomologyConvention,
    SimplicialComplexCanonicalizationResult,
    SimplicialComplexRequest,
)
from jacobian.math.topology._tools import TOOLS
from jacobian.math.topology.chain_complexes.operations import (
    homology_groups,
    tensor_product_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    HomologyGroupValue,
    HomologyResult,
    IntegralHomologyGroupValue,
)
from jacobian.math.topology.operations import simplicial_chain_complex_value


@overload
def _operation(
    operation_id: Literal["topology.simplicial_complex.canonicalize"],
) -> MathTool[SimplicialComplexRequest, SimplicialComplexCanonicalizationResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_complex.chain_complex.compute"],
) -> MathTool[ChainComplexRequest, ChainComplexResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_homology.compute"],
) -> MathTool[SimplicialHomologyRequest, SimplicialHomologyResult]: ...


@overload
def _operation(
    operation_id: Literal["topology.simplicial_homology.integral.compute"],
) -> MathTool[IntegralSimplicialHomologyRequest, IntegralSimplicialHomologyResult]: ...


def _operation(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _field_groups(result: HomologyResult) -> tuple[HomologyGroupValue, ...]:
    assert all(
        isinstance(group, HomologyGroupValue) for group in result.homology_groups
    )
    return cast(tuple[HomologyGroupValue, ...], result.homology_groups)


def _integral_groups(
    result: HomologyResult,
) -> tuple[IntegralHomologyGroupValue, ...]:
    assert all(
        isinstance(group, IntegralHomologyGroupValue)
        for group in result.homology_groups
    )
    return cast(tuple[IntegralHomologyGroupValue, ...], result.homology_groups)


def _circle() -> FiniteSimplicialComplex:
    request = SimplicialComplexRequest(
        vertices=("a", "b", "c"),
        facets=(("a", "b"), ("b", "c"), ("a", "c")),
    )
    return _operation("topology.simplicial_complex.canonicalize").run(request).complex


def _prime_field_chain(
    complex_: FiniteSimplicialComplex, prime: int
) -> ChainComplexResult:
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
    assert value.coefficient_ring is CoefficientRing.PRIME_FIELD
    assert value.prime == 2
    assert value.basis_sizes == (3, 3)

    result = homology_groups(value)
    assert [(group.degree, group.betti_number) for group in _field_groups(result)] == [
        (0, 1),
        (1, 1),
    ]


def test_producer_result_carries_its_canonical_value() -> None:
    """The public GF(p) producer exposes the canonical chain-complex
    value on the serialized result boundary itself."""
    result = _prime_field_chain(_circle(), 2)
    assert result.canonical_value is not None
    assert result.canonical_value == simplicial_chain_complex_value(result)
    homology = homology_groups(result.canonical_value)
    assert [group.betti_number for group in _field_groups(homology)] == [1, 1]


def test_producer_canonical_differentials_equal_its_sparse_boundaries() -> None:
    result = _prime_field_chain(_circle(), 2)
    expected = []
    for matrix in result.boundary_matrices[1:]:
        dense = [[0] * matrix.columns for _ in range(matrix.rows)]
        for entry in matrix.entries:
            dense[entry.row][entry.column] = entry.value
        expected.append(tuple(tuple(str(value) for value in row) for row in dense))

    assert result.canonical_value.differential_matrices == tuple(expected)


def test_integral_producer_value_enters_homology_unchanged() -> None:
    """The simplicial producer and generic consumer share the canonical ZZ value."""
    integral = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=_circle(),
            coefficient_ring=ChainCoefficientRing.INTEGER,
            convention=HomologyConvention.UNREDUCED,
        )
    )
    value = simplicial_chain_complex_value(integral)
    assert value.coefficient_ring is CoefficientRing.INTEGER
    assert value.basis_sizes == (3, 3)
    homology = homology_groups(value)
    assert [group.free_rank for group in _integral_groups(homology)] == [1, 1]


def test_converted_profile_matches_topology_homology_producer() -> None:
    """The composed profile agrees with the topology domain's own exact
    homology of the same complex over the same field."""
    complex_ = _circle()
    value = simplicial_chain_complex_value(_prime_field_chain(complex_, 3))
    composed = homology_groups(value)
    topology = _operation("topology.simplicial_homology.compute").run(
        SimplicialHomologyRequest(complex=complex_, prime=3)
    )
    assert (
        [group.betti_number for group in _field_groups(composed)]
        == [group.betti_number for group in topology.groups]
        == [1, 1]
    )


def test_serialized_value_round_trips_into_consumers() -> None:
    """The converted value survives serialization and keeps composing."""
    from jacobian.math.topology.chain_complexes.values import ChainComplexValue

    payload = simplicial_chain_complex_value(_prime_field_chain(_circle(), 2))
    value = ChainComplexValue.model_validate(payload.model_dump())
    result = homology_groups(value)
    assert _field_groups(result)[1].betti_number == 1
    tensor = tensor_product_complex(value, value)
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
    result = homology_groups(value)
    assert _field_groups(result)[0].betti_number == 1


def test_serialized_integral_value_round_trips_into_homology() -> None:
    integral = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=_circle(),
            coefficient_ring=ChainCoefficientRing.INTEGER,
            convention=HomologyConvention.UNREDUCED,
        )
    )
    payload = simplicial_chain_complex_value(integral).model_dump(mode="json")
    value = ChainComplexValue.model_validate(payload)
    result = homology_groups(value)
    assert result.coefficient_ring is CoefficientRing.INTEGER
    assert [group.free_rank for group in _integral_groups(result)] == [1, 1]


def test_reduced_chains_encode_augmentation_as_degree_minus_one() -> None:
    reduced = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=_circle(),
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
            convention=HomologyConvention.REDUCED,
        )
    )
    value = simplicial_chain_complex_value(reduced)
    assert (value.degree_min, value.degree_max) == (-1, 1)
    assert value.basis_sizes == (1, 3, 3)
    assert value.differential_matrices[0] == (("1", "1", "1"),)
    result = homology_groups(value)
    assert [group.betti_number for group in _field_groups(result)] == [0, 0, 1]


def test_integral_topology_result_wraps_the_chain_owned_result() -> None:
    complex_ = _circle()
    chain = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.INTEGER,
            convention=HomologyConvention.REDUCED,
        )
    )
    composed = homology_groups(chain.canonical_value)
    topology = _operation("topology.simplicial_homology.integral.compute").run(
        IntegralSimplicialHomologyRequest(
            complex=complex_, convention=HomologyConvention.REDUCED
        )
    )
    assert topology.homology == composed
    assert (
        IntegralSimplicialHomologyResult.model_validate_json(topology.model_dump_json())
        == topology
    )


def test_oversized_simplicial_groups_stay_outside_the_canonical_domain() -> None:
    """A GF(p) request whose groups exceed the canonical value's basis
    bound is rejected at admission: every accepted producer result must
    carry its canonical chain-complex value."""
    from jacobian.math.topology.chain_complexes.values import MAX_BASIS_SIZE

    labels = tuple(f"v{i}" for i in range(12))
    facets = tuple((labels[a], labels[b]) for a in range(12) for b in range(a + 1, 12))
    assert len(facets) > MAX_BASIS_SIZE
    big = _operation("topology.simplicial_complex.canonicalize").run(
        SimplicialComplexRequest(vertices=labels, facets=facets)
    )
    with pytest.raises(OperationDomainValidationError):
        _operation("topology.simplicial_complex.chain_complex.compute").run(
            ChainComplexRequest(
                complex=big.complex,
                coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
                prime=2,
                convention=HomologyConvention.UNREDUCED,
            )
        )


def test_aggregate_canonical_cell_bound_allows_the_widened_profile() -> None:
    """The widened canonical value admits ten disjoint tetrahedra.

    Their 5,200 aggregate boundary cells fit the 16,384-cell canonical
    envelope, even though the profile exceeds the former 4,096-cell limit.
    """
    vertices = tuple(f"v{i}" for i in range(40))
    facets = tuple(tuple(f"v{4 * k + j}" for j in range(4)) for k in range(10))
    complex_ = (
        _operation("topology.simplicial_complex.canonicalize")
        .run(SimplicialComplexRequest(vertices=vertices, facets=facets))
        .complex
    )
    assert complex_.f_vector == (40, 60, 40, 10)
    result = _operation("topology.simplicial_complex.chain_complex.compute").run(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
            convention=HomologyConvention.UNREDUCED,
        )
    )
    assert result.canonical_value is not None
    assert result.canonical_value.basis_sizes == (40, 60, 40, 10)
