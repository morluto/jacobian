"""Regression tests binding graph-spectrum results to their source graph."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.spectral._models import (
    GraphSpectrumRequest,
    GraphSpectrumResult,
)
from jacobian.math.graphs.spectral._operations import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
    verify_graph_spectrum_result,
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

    with pytest.raises(ValidationError, match="spectral operations support"):
        GraphSpectrumRequest(graph=graph)


def test_producer_spectra_retain_source_and_verify() -> None:
    path = _graph(3, ((0, 1), (1, 2)))

    adjacency = compute_adjacency_spectrum(GraphSpectrumRequest(graph=path))
    assert adjacency.graph == path
    assert adjacency.matrix_convention == "ADJACENCY"
    assert set(adjacency.eigenvalues) == {"-sqrt(2)", "0", "sqrt(2)"}
    assert adjacency.multiplicities == (1, 1, 1)
    assert GraphSpectrumResult.model_validate(adjacency.model_dump()) == adjacency
    assert verify_graph_spectrum_result(adjacency)

    laplacian = compute_laplacian_spectrum(GraphSpectrumRequest(graph=path))
    assert laplacian.matrix_convention == "LAPLACIAN"
    assert set(laplacian.eigenvalues) == {"0", "1", "3"}
    assert GraphSpectrumResult.model_validate(laplacian.model_dump()) == laplacian
    assert verify_graph_spectrum_result(laplacian)


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

    forged_value = copy.deepcopy(dumped)
    forged_value["eigenvalues"] = ["-1", "0", "sqrt(5)"]
    assert not verify_graph_spectrum_result(
        GraphSpectrumResult.model_validate(forged_value)
    )

    foreign_source = copy.deepcopy(dumped)
    foreign_source["graph"] = {"vertex_count": 3, "edges": [[0, 1]]}
    assert not verify_graph_spectrum_result(
        GraphSpectrumResult.model_validate(foreign_source)
    )

    swapped_convention = copy.deepcopy(dumped)
    swapped_convention["matrix_convention"] = "LAPLACIAN"
    assert not verify_graph_spectrum_result(
        GraphSpectrumResult.model_validate(swapped_convention)
    )


def test_permuted_pair_order_still_verifies_against_source() -> None:
    """A valid spectrum serialized in a different pair order remains exact."""

    complete = compute_adjacency_spectrum(
        GraphSpectrumRequest(
            graph=_graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        )
    )
    dumped = complete.model_dump()
    dumped["eigenvalues"] = list(reversed(dumped["eigenvalues"]))
    dumped["multiplicities"] = list(reversed(dumped["multiplicities"]))

    permuted = GraphSpectrumResult.model_validate(dumped)

    assert sorted(
        zip(permuted.eigenvalues, permuted.multiplicities, strict=True)
    ) == sorted(zip(complete.eigenvalues, complete.multiplicities, strict=True))
    assert verify_graph_spectrum_result(permuted)


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

    complete = compute_adjacency_spectrum(
        GraphSpectrumRequest(
            graph=_graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        )
    )
    assert dict(zip(complete.eigenvalues, complete.multiplicities, strict=True)) == {
        "-1": 3,
        "3": 1,
    }

    laplacian_complete = compute_laplacian_spectrum(
        GraphSpectrumRequest(
            graph=_graph(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        )
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


def test_v2_spectrum_operations_are_readmitted_with_source_bound_rationale() -> None:
    """The materially changed v2 contracts carry fresh owner-local decisions."""

    from jacobian.catalog.admission import AdmissionDecision
    from jacobian.math.graphs.spectral._admission import ADMISSIONS

    records = {
        record.operation_id: record
        for record in ADMISSIONS
        if record.operation_id
        in (
            "graph.spectrum.adjacency.compute",
            "graph.spectrum.laplacian.compute",
        )
    }
    assert set(records) == {
        "graph.spectrum.adjacency.compute",
        "graph.spectrum.laplacian.compute",
    }
    for operation_id, record in sorted(records.items()):
        assert record.decision is AdmissionDecision.KEEP, operation_id
        rationale = record.rationale.lower()
        assert "source graph" in rationale, operation_id
        assert "verifier" in rationale, operation_id


def test_spectrum_reconstructs_the_characteristic_polynomial() -> None:
    """The spectrum factorization matches the source-bound charpoly producer."""

    from sympy import Poly, symbols, together

    from jacobian.math.graphs.spectral._operations import (
        compute_adjacency_characteristic_polynomial,
    )
    from jacobian.math.graphs.spectral.operations import _adjacency_matrix
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy

    x = symbols("x")
    path = _graph(4, ((0, 1), (1, 2), (2, 3)))
    request = GraphSpectrumRequest(graph=path)
    spectrum = compute_adjacency_spectrum(request)
    charpoly_result = compute_adjacency_characteristic_polynomial(request)
    expected = Poly(
        rational_polynomial_to_sympy(charpoly_result.polynomial).as_expr(), x
    )

    factors = 1
    for value, multiplicity in _adjacency_matrix(path).eigenvals().items():
        factors *= (x - value) ** multiplicity
    claimed = together(factors.expand())
    assert together(expected.as_expr() - claimed) == 0
    assert sum(spectrum.multiplicities) == path.vertex_count
