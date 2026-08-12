"""Discovery metadata for typed polynomial-expression normalization."""


def test_normalizer_exposes_expansion_vocabulary(
    polynomial_normalization_services,
) -> None:
    descriptor = next(
        descriptor
        for descriptor in (
            polynomial_normalization_services.core.capabilities.catalog().capabilities
        )
        if descriptor.capability_id == "polynomial.expression.normalize"
    )

    assert {"expansion", "product", "power", "coefficients"} <= set(
        descriptor.tags
    )
    assert "Exactly expand sums, products, and bounded integer powers" in (
        descriptor.description
    )
