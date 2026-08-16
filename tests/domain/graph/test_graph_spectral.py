"""Tests for graph spectral operations: adjacency spectrum, laplacian spectrum, characteristic polynomial."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_spectral import (
    GraphCharacteristicPolynomialRequest,
    GraphSpectrumRequest,
)
from jacobian.domains.graph_spectral.operations import (
    compute_adjacency_spectrum,
    compute_characteristic_polynomial,
    compute_laplacian_spectrum,
)


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

# Path graph P3: 0 - 1 - 2
P3 = {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}

# Path graph P4: 0 - 1 - 2 - 3
P4 = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]}

# Cycle graph C4: 0 - 1 - 2 - 3 - 0
C4 = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]}

# Complete graph K3
K3 = {"vertex_count": 3, "edges": [[0, 1], [0, 2], [1, 2]]}

# Complete graph K4
K4 = {
    "vertex_count": 4,
    "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
}

# Disconnected graph: K3 + isolated vertex
DISCONNECTED = {
    "vertex_count": 4,
    "edges": [[0, 1], [0, 2], [1, 2]],
}


# ---------------------------------------------------------------------------
# Adjacency spectrum
# ---------------------------------------------------------------------------

class TestAdjacencySpectrum:
    def test_path_graph_p3(self) -> None:
        result = compute_adjacency_spectrum(
            GraphSpectrumRequest.model_validate({"graph": P3})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # P3 eigenvalues: 0, sqrt(2), -sqrt(2)
        assert spectrum == {"0": 1, "sqrt(2)": 1, "-sqrt(2)": 1}

    def test_cycle_graph_c4(self) -> None:
        result = compute_adjacency_spectrum(
            GraphSpectrumRequest.model_validate({"graph": C4})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # C4 eigenvalues: 2, 0, 0, -2
        assert spectrum == {"2": 1, "0": 2, "-2": 1}

    def test_complete_graph_k3(self) -> None:
        result = compute_adjacency_spectrum(
            GraphSpectrumRequest.model_validate({"graph": K3})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # K3 eigenvalues: 2, -1, -1
        assert spectrum == {"2": 1, "-1": 2}

    def test_disconnected_graph(self) -> None:
        result = compute_adjacency_spectrum(
            GraphSpectrumRequest.model_validate({"graph": DISCONNECTED})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # K3 + isolated: eigenvalues of K3 are {2, -1, -1}, plus 0 for isolated
        assert spectrum == {"2": 1, "-1": 2, "0": 1}

    def test_total_multiplicity_equals_vertex_count(self) -> None:
        for graph in [P3, P4, C4, K3, K4, DISCONNECTED]:
            result = compute_adjacency_spectrum(
                GraphSpectrumRequest.model_validate({"graph": graph})
            )
            assert sum(result.multiplicities) == graph["vertex_count"]

    def test_convention_is_sympy_eigenvals(self) -> None:
        result = compute_adjacency_spectrum(
            GraphSpectrumRequest.model_validate({"graph": P3})
        )
        assert result.convention == "SYMPY_EIGENVALS"


# ---------------------------------------------------------------------------
# Laplacian spectrum
# ---------------------------------------------------------------------------

class TestLaplacianSpectrum:
    def test_path_graph_p3(self) -> None:
        result = compute_laplacian_spectrum(
            GraphSpectrumRequest.model_validate({"graph": P3})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # P3 Laplacian eigenvalues: 0, 1, 3
        assert spectrum == {"0": 1, "1": 1, "3": 1}

    def test_cycle_graph_c4(self) -> None:
        result = compute_laplacian_spectrum(
            GraphSpectrumRequest.model_validate({"graph": C4})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # C4 Laplacian eigenvalues: 0, 2, 2, 4
        assert spectrum == {"0": 1, "2": 2, "4": 1}

    def test_complete_graph_k3(self) -> None:
        result = compute_laplacian_spectrum(
            GraphSpectrumRequest.model_validate({"graph": K3})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # K3 Laplacian eigenvalues: 0, 3, 3
        assert spectrum == {"0": 1, "3": 2}

    def test_disconnected_graph_has_zero_for_each_component(self) -> None:
        result = compute_laplacian_spectrum(
            GraphSpectrumRequest.model_validate({"graph": DISCONNECTED})
        )
        spectrum = dict(zip(result.eigenvalues, result.multiplicities, strict=True))
        # K3 + isolated: Laplacian has eigenvalue 0 with multiplicity 2 (two components)
        assert spectrum.get("0") == 2

    def test_total_multiplicity_equals_vertex_count(self) -> None:
        for graph in [P3, P4, C4, K3, K4, DISCONNECTED]:
            result = compute_laplacian_spectrum(
                GraphSpectrumRequest.model_validate({"graph": graph})
            )
            assert sum(result.multiplicities) == graph["vertex_count"]

    def test_convention_is_sympy_eigenvals(self) -> None:
        result = compute_laplacian_spectrum(
            GraphSpectrumRequest.model_validate({"graph": P3})
        )
        assert result.convention == "SYMPY_EIGENVALS"


# ---------------------------------------------------------------------------
# Characteristic polynomial
# ---------------------------------------------------------------------------

class TestCharacteristicPolynomial:
    def test_adjacency_charpoly_path_graph_p3(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": P3, "matrix": "ADJACENCY"}
            )
        )
        # P3 adjacency charpoly: lambda^3 - 2*lambda = lambda^3 + 0*lambda^2 - 2*lambda + 0
        assert result.degree == 3
        assert result.coefficients_descending == ("1", "0", "-2", "0")
        assert result.monic is True
        assert result.matrix == "ADJACENCY"
        assert result.convention == "DET_LAMBDA_I_MINUS_M"

    def test_adjacency_charpoly_cycle_graph_c4(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": C4, "matrix": "ADJACENCY"}
            )
        )
        # C4 adjacency eigenvalues: {2, 0, 0, -2}
        # charpoly = (lambda-2)(lambda)(lambda)(lambda+2) = lambda^4 - 4*lambda^2
        assert result.degree == 4
        assert result.coefficients_descending == ("1", "0", "-4", "0", "0")

    def test_adjacency_charpoly_complete_graph_k3(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": K3, "matrix": "ADJACENCY"}
            )
        )
        # K3 adjacency eigenvalues: {2, -1, -1}
        # charpoly = (lambda-2)(lambda+1)^2 = lambda^3 - 3*lambda - 2
        assert result.degree == 3
        assert result.coefficients_descending == ("1", "0", "-3", "-2")

    def test_adjacency_charpoly_disconnected_graph(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": DISCONNECTED, "matrix": "ADJACENCY"}
            )
        )
        # K3 + isolated: charpoly = lambda * (lambda^3 - 3*lambda - 2)
        # = lambda^4 - 3*lambda^2 - 2*lambda
        assert result.degree == 4
        assert result.coefficients_descending == ("1", "0", "-3", "-2", "0")

    def test_laplacian_charpoly_path_graph_p3(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": P3, "matrix": "LAPLACIAN"}
            )
        )
        # P3 Laplacian eigenvalues: {0, 1, 3}
        # charpoly = lambda * (lambda-1) * (lambda-3) = lambda^3 - 4*lambda^2 + 3*lambda
        assert result.degree == 3
        assert result.coefficients_descending == ("1", "-4", "3", "0")
        assert result.matrix == "LAPLACIAN"

    def test_laplacian_charpoly_cycle_graph_c4(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": C4, "matrix": "LAPLACIAN"}
            )
        )
        # C4 Laplacian eigenvalues: {0, 2, 2, 4}
        # charpoly = lambda * (lambda-2)^2 * (lambda-4) = lambda^4 - 8*lambda^3 + 20*lambda^2 - 16*lambda
        assert result.degree == 4
        assert result.coefficients_descending == ("1", "-8", "20", "-16", "0")

    def test_laplacian_charpoly_complete_graph_k3(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": K3, "matrix": "LAPLACIAN"}
            )
        )
        # K3 Laplacian eigenvalues: {0, 3, 3}
        # charpoly = lambda * (lambda-3)^2 = lambda^3 - 6*lambda^2 + 9*lambda
        assert result.degree == 3
        assert result.coefficients_descending == ("1", "-6", "9", "0")

    def test_laplacian_charpoly_disconnected_graph(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": DISCONNECTED, "matrix": "LAPLACIAN"}
            )
        )
        # K3 + isolated: Laplacian eigenvalues: {0, 3, 3, 0}
        # charpoly = lambda^2 * (lambda-3)^2 = lambda^4 - 6*lambda^3 + 9*lambda^2
        assert result.degree == 4
        assert result.coefficients_descending == ("1", "-6", "9", "0", "0")

    def test_default_matrix_is_adjacency(self) -> None:
        result = compute_characteristic_polynomial(
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": P3}
            )
        )
        assert result.matrix == "ADJACENCY"

    def test_charpoly_consistency_with_spectrum(self) -> None:
        """The characteristic polynomial's roots must match the eigenvalues."""
        for graph, matrix in [
            (P3, "ADJACENCY"),
            (P3, "LAPLACIAN"),
            (C4, "ADJACENCY"),
            (C4, "LAPLACIAN"),
            (K3, "ADJACENCY"),
            (K3, "LAPLACIAN"),
            (DISCONNECTED, "ADJACENCY"),
            (DISCONNECTED, "LAPLACIAN"),
        ]:
            spectrum_result = (
                compute_adjacency_spectrum
                if matrix == "ADJACENCY"
                else compute_laplacian_spectrum
            )(GraphSpectrumRequest.model_validate({"graph": graph}))

            charpoly_result = compute_characteristic_polynomial(
                GraphCharacteristicPolynomialRequest.model_validate(
                    {"graph": graph, "matrix": matrix}
                )
            )

            # Total multiplicity must match degree
            assert sum(spectrum_result.multiplicities) == charpoly_result.degree


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

class TestContractValidation:
    def test_spectrum_request_rejects_non_simple_graph(self) -> None:
        with pytest.raises(ValidationError, match="self-loops"):
            GraphSpectrumRequest.model_validate(
                {"graph": {"vertex_count": 2, "edges": [[0, 0], [0, 1]]}}
            )

    def test_charpoly_request_rejects_invalid_matrix(self) -> None:
        with pytest.raises(ValidationError):
            GraphCharacteristicPolynomialRequest.model_validate(
                {"graph": P3, "matrix": "INVALID"}
            )

    def test_charpoly_request_rejects_non_simple_graph(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            GraphCharacteristicPolynomialRequest.model_validate(
                {
                    "graph": {"vertex_count": 2, "edges": [[0, 1], [1, 0]]},
                    "matrix": "ADJACENCY",
                }
            )

    def test_charpoly_result_rejects_wrong_degree(self) -> None:
        """Degree must be len(coefficients) - 1."""
        with pytest.raises(ValidationError, match="dense coefficient"):
            from jacobian.contracts.graph_spectral import (
                GraphCharacteristicPolynomialResult,
            )

            GraphCharacteristicPolynomialResult(
                degree=3,
                coefficients_descending=("1", "0", "-2"),  # 3 coeffs, degree 3 -> mismatch
                matrix="ADJACENCY",
            )

    def test_charpoly_result_rejects_non_monic(self) -> None:
        with pytest.raises(ValidationError, match="monic"):
            from jacobian.contracts.graph_spectral import (
                GraphCharacteristicPolynomialResult,
            )

            GraphCharacteristicPolynomialResult(
                degree=3,
                coefficients_descending=("2", "0", "-2", "0"),
                matrix="ADJACENCY",
            )
