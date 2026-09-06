"""Catalog contract for rational generating-function series values."""

from jacobian.catalog.catalog import Catalog


def test_catalog_schema_publishes_the_canonical_series_example() -> None:
    descriptor = Catalog.open().inspect(
        "combinatorics.generating_function.coefficients.compute"
    )
    assert descriptor is not None
    assert set(descriptor.output_schema["properties"]) == {
        "numerator",
        "denominator",
        "coefficient_convention",
        "expansion_point",
        "truncation_order",
        "series",
    }
    example = descriptor.output_schema["examples"][0]
    assert example["series"]["variable"] == "x"
    assert example["series"]["truncation_order"] == 3
