"""Regression tests binding graph-spectrum results to their source graph."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.spectra._models import (
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
    assert set(adjacency.eigenvalues) == {"-sqrt(2)", "0", "sqrt(2)"}
    assert adjacency.multiplicities == (1, 1, 1)
    assert GraphSpectrumResult.model_validate(adjacency.model_dump()) == adjacency

    laplacian = compute_laplacian_spectrum(GraphSpectrumRequest(graph=path))
    assert laplacian.matrix_convention == "LAPLACIAN"
    assert set(laplacian.eigenvalues) == {"0", "1", "3"}
    assert GraphSpectrumResult.model_validate(laplacian.model_dump()) == laplacian


def test_structural_constraints_reject_forged_payloads() -> None:
    path = _graph(3, ((0, 1), (1, 2)))
    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    dumped = adjacency.model_dump()

    length_mismatch = copy.deepcopy(dumped)
    length_mismatch["multiplicities"] = [1, 1]
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(length_mismatch)

    negative_multiplicity = copy.deepcopy(dumped)
    negative_multiplicity["multiplicities"] = [1, -7, 1]
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(negative_multiplicity)

    zero_multiplicity = copy.deepcopy(dumped)
    zero_multiplicity["multiplicities"] = [0, 2, 1]
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(zero_multiplicity)

    wrong_total = copy.deepcopy(dumped)
    wrong_total["multiplicities"] = [1, 1, 2]
    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(wrong_total)


def test_permuted_pair_order_remains_structurally_valid() -> None:

    complete_graph = _graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    complete = compute_adjacency_spectrum(GraphSpectrumRequest(graph=complete_graph))
    dumped = complete.model_dump()
    dumped["eigenvalues"] = list(reversed(dumped["eigenvalues"]))
    dumped["multiplicities"] = list(reversed(dumped["multiplicities"]))

    permuted = GraphSpectrumResult.model_validate(dumped)

    assert sorted(
        zip(permuted.eigenvalues, permuted.multiplicities, strict=True)
    ) == sorted(zip(complete.eigenvalues, complete.multiplicities, strict=True))


def test_duplicate_eigenvalue_entries_are_rejected() -> None:
    path = _graph(3, ((0, 1), (1, 2)))
    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    duplicated = copy.deepcopy(adjacency.model_dump())
    duplicated["eigenvalues"] = ["0", "0", "sqrt(2)"]
    duplicated["multiplicities"] = [1, 1, 1]

    with pytest.raises(ValidationError):
        GraphSpectrumResult.model_validate(duplicated)


def test_degenerate_and_repeated_spectra_stay_exact() -> None:
    empty = compute_adjacency_spectrum(GraphSpectrumRequest(graph=_graph(2, ())))
    assert empty.eigenvalues == ("0",)
    assert empty.multiplicities == (2,)
    assert GraphSpectrumResult.model_validate(empty.model_dump()) == empty

    complete_graph = _graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    complete = compute_adjacency_spectrum(GraphSpectrumRequest(graph=complete_graph))
    assert dict(zip(complete.eigenvalues, complete.multiplicities, strict=True)) == {
        "-1": 3,
        "3": 1,
    }

    laplacian_complete = compute_laplacian_spectrum(
        GraphSpectrumRequest(graph=complete_graph)
    )
    assert dict(
        zip(
            laplacian_complete.eigenvalues,
            laplacian_complete.multiplicities,
            strict=True,
        )
    ) == {
        "0": 1,
        "4": 3,
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

    from sympy import Poly, symbols, sympify, together

    from jacobian.math.graphs.spectra._tools import (
        compute_adjacency_characteristic_polynomial,
    )
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    x = symbols("x")
    path = _graph(4, ((0, 1), (1, 2), (2, 3)))
    request = GraphSpectrumRequest(graph=path)
    spectrum = compute_adjacency_spectrum(request)
    charpoly_result = compute_adjacency_characteristic_polynomial(request)
    expected = Poly(
        rational_polynomial_to_sympy(charpoly_result.polynomial).as_expr(), x
    )

    factors = x**0
    # These strings are generated by the typed producer, not accepted caller input.
    for value, multiplicity in zip(
        spectrum.eigenvalues, spectrum.multiplicities, strict=True
    ):
        factors *= (x - sympify(value)) ** multiplicity
    claimed = together(factors.expand())
    assert together(expected.as_expr() - claimed) == 0
    assert sum(spectrum.multiplicities) == path.vertex_count
