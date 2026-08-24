"""Discovery coverage for exact finite-Abelian Fourier orthogonality."""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest


def test_exact_finite_abelian_fourier_query_routes_to_spectral_pair_decision() -> None:
    result = Catalog.open().search(
        OperationDiscoveryRequest(
            query="exact finite Abelian Fourier orthogonality spectral pair",
            limit=5,
        )
    )

    assert result.matches
    assert result.matches[0].operation_id == (
        "finite_abelian_group.spectral_pair.decide"
    )
