import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_model_isomorphism_public_example_composes_through_dispatch() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.isomorphism.decide")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["isomorphic"] is True
    assert result.output["scaling"] is not None


def test_group_structure_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.group_structure.compute")
    assert operation is not None
    invocation = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    result = operation.result_type.model_validate_json(json.dumps(invocation.output))
    assert tuple(result.group.invariant_factors) == (9,)
    assert len(result.generators) == 1


def test_point_order_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.point.order.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["group_cardinality"] == 9
    assert result.output["order"] == 9
    assert result.output["annihilating_multiple"]["at_infinity"] is True
    assert [
        (witness["prime"], witness["reduced_scalar"])
        for witness in result.output["prime_divisor_witnesses"]
    ] == [(3, 3)]


def test_isogeny_class_operation_is_published_and_serializable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.isogeny_class.decide")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["same_isogeny_class"] is False
    assert result.output["first_frobenius_polynomial"] == [5, 3, 1]
    assert result.output["second_frobenius_polynomial"] == [5, 1, 1]


def test_finite_field_addition_is_published_and_runs_through_catalog() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("elliptic_curve.finite_field.point.add.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["point"]["at_infinity"] is False
    assert result.output["point"]["x"]["coordinates"] == ["3"]
    assert result.output["point"]["y"]["coordinates"] == ["4"]


def test_extension_count_catalog_example_runs_with_typed_output() -> None:
    operation = Catalog.open().operation(
        "elliptic_curve.finite_field.extension_counts.compute"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, Catalog.open()
    )
    validated = operation.result_type.model_validate_json(json.dumps(result.output))
    assert validated.counts[0].cardinality == 9
    assert validated.counts[1].cardinality == 27
