"""Public catalog coverage for versioned RSK contracts."""

from jacobian.catalog.catalog import Catalog


def test_permutation_rsk_descriptor_publishes_version_two() -> None:
    descriptor = Catalog.open().inspect("combinatorics.rsk.permutation.compute")

    assert descriptor is not None
    assert descriptor.version == "2"
    assert (
        descriptor.input_schema["properties"]["convention"]["const"]
        == "ROW_INSERTION_RSK_V1"
    )
    assert (
        descriptor.output_schema["properties"]["convention"]["const"]
        == "ROW_INSERTION_RSK_V1"
    )
