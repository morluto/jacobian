"""Profile descriptions distinguish dimensions, derived values, and geometric scope."""

from jacobian.catalog.catalog import Catalog


def test_ordered_difference_description_distinguishes_dimension_and_source_bound() -> (
    None
):
    descriptor = Catalog.open().inspect("additive.ordered_difference_profile.compute")
    assert descriptor is not None
    assert "share one dimension" in descriptor.description
    assert "equal-length" not in descriptor.description
    assert "source coordinate" in descriptor.description
    assert "6 digits" in descriptor.description
    assert "7 digits" in descriptor.description
    assert "6 digits" in descriptor.input_schema["properties"]["vectors"]["description"]


def test_circumradius_description_states_squared_values_and_geometric_scope() -> None:
    descriptor = Catalog.open().inspect("geometry.points.circumradius_profile.compute")
    assert descriptor is not None
    assert "squared" in descriptor.title
    assert "collinear" in descriptor.description
    assert "source-labelled triple indices" in descriptor.description
    assert "general position is not checked" in descriptor.description
