"""Public catalog coverage for versioned RSK contracts."""

from jacobian.catalog.catalog import Catalog


def test_permutation_rsk_descriptor_publishes_version_two() -> None:
    descriptor = Catalog.open().inspect("combinatorics.rsk.permutation.compute")

    assert descriptor is not None
    assert (
        descriptor.input_schema["properties"]["convention"]["const"]
        == "ROW_INSERTION_RSK_V1"
    )
    assert (
        descriptor.output_schema["properties"]["convention"]["const"]
        == "ROW_INSERTION_RSK_V1"
    )


def test_inverse_word_rsk_descriptor_publishes_canonical_word_result() -> None:
    descriptor = Catalog.open().inspect("tableau.rsk.inverse_word.compute")

    assert descriptor is not None
    assert set(descriptor.output_schema["properties"]) == {"alphabet", "letters"}


def test_word_rsk_descriptor_publishes_version_two() -> None:
    descriptor = Catalog.open().inspect("tableau.rsk.word.compute")

    assert descriptor is not None
    assert (
        descriptor.input_schema["properties"]["convention"]["const"]
        == "ROW_INSERTION_RSK_V1"
    )
    assert (
        "minItems"
        not in (
            descriptor.input_schema["$defs"]["FiniteWord"]["properties"]["alphabet"]
        )
    )
