"""Discovery metadata for exact polynomial-expression normalization."""

from pathlib import Path

from tests.component.providers.polynomial.polynomial_normalization_support import (
    open_polynomial_normalization_services,
)


def test_polynomial_normalizer_advertises_expansion_vocabulary(
    tmp_path: Path,
) -> None:
    with open_polynomial_normalization_services(tmp_path) as services:
        descriptor = next(
            item
            for item in services.core.capabilities.catalog().capabilities
            if item.capability_id == "polynomial.expression.normalize"
        )

    assert {
        "expansion",
        "product",
        "power",
        "coefficients",
    } <= set(descriptor.tags)
    assert "Exactly expand sums, products, and bounded integer powers" in (
        descriptor.description
    )
