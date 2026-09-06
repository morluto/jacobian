"""Regression tests binding graph-spectrum results to their source graph."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.spectra import verify_spectrum
from jacobian.math.graphs.spectra._models import (
    GraphCharacteristicPolynomialRequest,
    GraphSpectrumRequest,
    GraphSpectrumResult,
)
from jacobian.math.graphs.spectra._tools import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _graph(
    vertex_count: int, edges: tuple[tuple[int, int], ...]
) -> IndexedSimpleUndirectedGraph:
    return IndexedSimpleUndirectedGraph(vertex_count=vertex_count, edges=edges)


def test_indexed_graph_value_composes_with_coloring_and_spectral_requests() -> None:
    from jacobian.math.graphs.coloring._models import KColorabilityRequest

    graph = _graph(3, ((0, 1), (1, 2)))
    assert KColorabilityRequest(graph=graph, colors=2).graph is graph
    assert GraphSpectrumRequest(graph=graph).graph is graph


def test_null_graph_composes_and_yields_the_empty_spectrum() -> None:
    """The shared canonical value admits the null graph end to end: both
    spectral conventions decide it exactly with no eigenvalues."""
    from jacobian.math.graphs.coloring._models import KColorabilityRequest

    null_graph = _graph(0, ())
    assert KColorabilityRequest(graph=null_graph, colors=2).graph is null_graph

    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=null_graph))
    assert adjacency.eigenvalues == ()
    assert adjacency.multiplicities == ()
    assert GraphSpectrumResult.model_validate(adjacency.model_dump()) == adjacency

    laplacian = compute_laplacian_spectrum(GraphSpectrumRequest(graph=null_graph))
    assert laplacian.eigenvalues == ()
    assert laplacian.multiplicities == ()
    assert GraphSpectrumResult.model_validate(laplacian.model_dump()) == laplacian


def test_spectral_request_rejects_the_shared_value_outside_its_envelope() -> None:
    graph = _graph(33, tuple((index, index + 1) for index in range(32)))

    request = GraphSpectrumRequest(graph=graph)
    with pytest.raises(
        OperationDomainValidationError, match="spectral operations support"
    ):
        compute_adjacency_spectrum(request)


def test_producer_spectra_retain_source() -> None:
    path = _graph(3, ((0, 1), (1, 2)))

    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    assert adjacency.graph == path
    assert adjacency.matrix_convention == "ADJACENCY"
    assert {
        (entry.value.polynomial, entry.value.real_root_index)
        for entry in adjacency.spectrum
    } == {
        (("1", "0", "-2"), 0),
        (("1", "0", "-2"), 1),
        (("1", "0"), 0),
    }
    assert adjacency.multiplicities == (1, 1, 1)
    assert GraphSpectrumResult.model_validate(adjacency.model_dump()) == adjacency
    assert verify_spectrum(adjacency)

    laplacian = compute_laplacian_spectrum(GraphSpectrumRequest(graph=path))
    assert laplacian.matrix_convention == "LAPLACIAN"
    assert {
        (entry.value.polynomial, entry.value.real_root_index)
        for entry in laplacian.spectrum
    } == {
        (("1", "0"), 0),
        (("1", "-1"), 0),
        (("1", "-3"), 0),
    }
    assert GraphSpectrumResult.model_validate(laplacian.model_dump()) == laplacian
    assert verify_spectrum(laplacian)


def test_structural_constraints_reject_forged_payloads() -> None:
    path = _graph(3, ((0, 1), (1, 2)))
    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    dumped = adjacency.model_dump()

    length_mismatch = copy.deepcopy(dumped)
    length_mismatch["spectrum"] = length_mismatch["spectrum"][:2]
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(length_mismatch)

    negative_multiplicity = copy.deepcopy(dumped)
    negative_multiplicity["spectrum"][1]["multiplicity"] = -7
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(negative_multiplicity)

    zero_multiplicity = copy.deepcopy(dumped)
    zero_multiplicity["spectrum"][0]["multiplicity"] = 0
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(zero_multiplicity)

    wrong_total = copy.deepcopy(dumped)
    wrong_total["spectrum"][0]["multiplicity"] = 2
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(wrong_total)


def test_permuted_pair_order_remains_structurally_valid() -> None:

    complete_graph = _graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    complete = compute_adjacency_spectrum(GraphSpectrumRequest(graph=complete_graph))
    dumped = complete.model_dump()
    dumped["spectrum"] = list(reversed(dumped["spectrum"]))

    permuted = GraphSpectrumResult.model_validate(dumped)

    assert sorted(
        (entry.value.polynomial, entry.value.real_root_index, entry.multiplicity)
        for entry in permuted.spectrum
    ) == sorted(
        (entry.value.polynomial, entry.value.real_root_index, entry.multiplicity)
        for entry in complete.spectrum
    )


def test_duplicate_eigenvalue_entries_are_rejected() -> None:
    path = _graph(3, ((0, 1), (1, 2)))
    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    duplicated = copy.deepcopy(adjacency.model_dump())
    duplicated["spectrum"] = list(duplicated["spectrum"])
    duplicated["spectrum"][1] = copy.deepcopy(duplicated["spectrum"][0])

    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(duplicated)


def test_degenerate_and_repeated_spectra_stay_exact() -> None:
    empty = compute_adjacency_spectrum(GraphSpectrumRequest(graph=_graph(2, ())))
    assert empty.eigenvalues[0].polynomial == ("1", "0")
    assert empty.multiplicities == (2,)
    assert GraphSpectrumResult.model_validate(empty.model_dump()) == empty

    complete_graph = _graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    complete = compute_adjacency_spectrum(GraphSpectrumRequest(graph=complete_graph))
    assert {
        (entry.value.polynomial, entry.value.real_root_index): entry.multiplicity
        for entry in complete.spectrum
    } == {
        (("1", "1"), 0): 3,
        (("1", "-3"), 0): 1,
    }

    laplacian_complete = compute_laplacian_spectrum(
        GraphSpectrumRequest(graph=complete_graph)
    )
    assert {
        (entry.value.polynomial, entry.value.real_root_index): entry.multiplicity
        for entry in laplacian_complete.spectrum
    } == {
        (("1", "0"), 0): 1,
        (("1", "-4"), 0): 3,
    }


def test_v2_spectrum_operations_are_published() -> None:
    from jacobian.math.graphs.spectra._tools import TOOLS

    operation_ids = {tool.operation_id for tool in TOOLS}
    assert {
        "graph.spectrum.adjacency.compute",
        "graph.spectrum.laplacian.compute",
    } <= operation_ids


def test_spectrum_reconstructs_the_characteristic_polynomial() -> None:
    """The spectrum factorization matches the source-bound charpoly producer."""

    from sympy import Poly, symbols, together

    from jacobian.math.graphs.spectra._tools import (
        compute_adjacency_characteristic_polynomial,
    )
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    x = symbols("x")
    path = _graph(4, ((0, 1), (1, 2), (2, 3)))
    request = GraphSpectrumRequest(graph=path)
    spectrum = compute_adjacency_spectrum(request)
    charpoly_result = compute_adjacency_characteristic_polynomial(
        GraphCharacteristicPolynomialRequest(graph=path)
    )
    expected = Poly(
        rational_polynomial_to_sympy(charpoly_result.polynomial).as_expr(), x
    )

    factors = x**0
    for entry in spectrum.spectrum:
        polynomial = Poly.from_list(
            [int(coefficient) for coefficient in entry.value.polynomial],
            gens=x,
        )
        value = polynomial.all_roots()[entry.value.real_root_index]
        factors *= (x - value) ** entry.multiplicity
    claimed = together(factors.expand())
    assert together(expected.as_expr() - claimed) == 0
    assert sum(spectrum.multiplicities) == path.vertex_count


def test_forged_well_formed_algebraic_spectrum_is_rejected_by_verifier() -> None:
    path = _graph(3, ((0, 1), (1, 2)))
    result = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    forged = result.model_dump(mode="json")
    forged["spectrum"][0]["value"] = {
        "polynomial": ["1", "-1"],
        "real_root_index": 0,
    }
    decoded = GraphSpectrumResult.model_validate(forged)
    assert not verify_spectrum(decoded)


def test_arbitrary_text_is_not_a_spectrum_value() -> None:
    result = compute_adjacency_spectrum(
        GraphSpectrumRequest(graph=_graph(2, ((0, 1),)))
    )
    forged = result.model_dump(mode="json")
    forged["spectrum"][0]["value"] = "not-an-algebraic-number"
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(forged)
