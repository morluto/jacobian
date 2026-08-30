"""Exact order for field elements at a recognized real embedding."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
    SimpleNumberFieldRealEmbeddingOrder,
    compare_real_embedding_elements,
    embeddings,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldRealEmbeddingOrderRequest,
)
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    NumberFieldRealEmbeddingOrderError,
)
from jacobian.math.number_theory.number_fields._tools import TOOLS
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbeddingRecord,
)


def _field(*coefficients: str) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(coefficients_descending=coefficients)


def _element(
    field: SimpleNumberFieldPresentation,
    *coordinates: int | Fraction,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(coordinate))
            for coordinate in coordinates
        ),
    )


@pytest.fixture(scope="module")
def sqrt_two_records() -> tuple[
    RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
]:
    profile = embeddings(_field("1", "0", "-2"))
    negative, positive = profile.records
    assert isinstance(negative, RealNumberFieldEmbeddingRecord)
    assert isinstance(positive, RealNumberFieldEmbeddingRecord)
    return negative, positive


def _binding(
    element: SimpleNumberFieldElement,
    record: RealNumberFieldEmbeddingRecord,
) -> SimpleNumberFieldRealEmbeddingBinding:
    return SimpleNumberFieldRealEmbeddingBinding(
        element=element,
        embedding_record=record,
    )


def test_same_abstract_element_has_opposite_sign_at_two_real_embeddings(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    negative, positive = sqrt_two_records
    alpha = _element(negative.embedding.presentation, 0, 1)
    zero = _element(negative.embedding.presentation, 0, 0)

    negative_order = compare_real_embedding_elements(
        _binding(alpha, negative), _binding(zero, negative)
    )
    positive_order = compare_real_embedding_elements(
        _binding(alpha, positive), _binding(zero, positive)
    )

    assert negative_order.order == "LT"
    assert negative_order.difference_enclosure.upper.as_fraction() < 0
    assert positive_order.order == "GT"
    assert positive_order.difference_enclosure.lower.as_fraction() > 0
    assert negative_order.difference.element == positive_order.difference.element


def test_exact_quotient_equality_does_not_depend_on_the_selected_embedding(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    negative, positive = sqrt_two_records
    value = _element(negative.embedding.presentation, Fraction(2, 3), -5)

    for record in (negative, positive):
        result = compare_real_embedding_elements(
            _binding(value, record), _binding(value, record)
        )
        assert result.order == "EQ"
        assert all(
            coordinate.as_fraction() == 0
            for coordinate in result.difference.element.coefficients_ascending
        )
        assert result.difference_enclosure.lower.as_fraction() == 0
        assert result.difference_enclosure.interval_type == "SINGLETON"


def test_rational_difference_uses_exact_singleton_evidence(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[0]
    result = compare_real_embedding_elements(
        _binding(_element(record.embedding.presentation, Fraction(3, 2), 0), record),
        _binding(_element(record.embedding.presentation, 1, 0), record),
    )

    assert result.order == "GT"
    assert result.difference_enclosure.lower.as_fraction() == Fraction(1, 2)
    assert result.difference_enclosure.interval_type == "SINGLETON"


def test_binding_is_structural_but_consumer_rejects_a_forged_record(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    forged_data = record.model_dump(mode="json")
    forged_data["isolating_interval"] = {
        "lower": {"num": "2", "den": "1"},
        "upper": {"num": "3", "den": "1"},
        "interval_type": "OPEN",
    }
    forged = RealNumberFieldEmbeddingRecord.model_validate(forged_data)
    binding = _binding(_element(record.embedding.presentation, 0, 1), forged)

    with pytest.raises(NumberFieldRealEmbeddingOrderError) as caught:
        compare_real_embedding_elements(binding, binding)

    assert caught.value.reason == "embedding_record_not_recognized"


def test_consumer_rejects_a_structurally_valid_reducible_presentation() -> None:
    field = _field("1", "0", "-1")
    record = RealNumberFieldEmbeddingRecord.model_validate(
        {
            "kind": "REAL",
            "embedding": {
                "kind": "REAL",
                "presentation": field.model_dump(mode="json"),
                "root": {
                    "polynomial": ["1", "0", "-1"],
                    "real_root_index": 0,
                },
            },
            "isolating_interval": {
                "lower": {"num": "-2", "den": "1"},
                "upper": {"num": "0", "den": "1"},
                "interval_type": "OPEN",
            },
        }
    )
    binding = _binding(_element(field, 0, 1), record)

    with pytest.raises(NumberFieldRealEmbeddingOrderError) as caught:
        compare_real_embedding_elements(binding, binding)

    assert caught.value.reason == "embedding_not_irreducible"


def test_comparison_rejects_foreign_selected_records_before_recognition(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    negative, positive = sqrt_two_records
    alpha = _element(negative.embedding.presentation, 0, 1)

    with pytest.raises(NumberFieldRealEmbeddingOrderError) as caught:
        compare_real_embedding_elements(
            _binding(alpha, negative), _binding(alpha, positive)
        )

    assert caught.value.reason == "embedding_record_mismatch"


def test_binding_schema_rejects_a_nonreal_embedding_record() -> None:
    nonreal = embeddings(_field("1", "0", "1")).records[0]

    with pytest.raises(ValidationError):
        SimpleNumberFieldRealEmbeddingBinding.model_validate(
            {
                "element": _element(nonreal.embedding.presentation, 0, 1).model_dump(
                    mode="json"
                ),
                "embedding_record": nonreal.model_dump(mode="json"),
            }
        )


def test_result_round_trips_without_replaying_recognition(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    result = compare_real_embedding_elements(
        _binding(_element(record.embedding.presentation, 0, 1), record),
        _binding(_element(record.embedding.presentation, 1, 0), record),
    )

    assert (
        SimpleNumberFieldRealEmbeddingOrder.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )


def test_catalog_operation_runs_its_declared_example() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "number_field.real_embedding.element_order.compare"
    )
    request = NumberFieldRealEmbeddingOrderRequest.model_validate(
        operation.examples[0].input
    )

    result = operation.run(request)

    assert result.order == "GT"
    assert result.difference_enclosure.lower.as_fraction() == Fraction(1, 2)


def test_catalog_operation_projects_unrecognized_record_as_a_typed_error(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[0]
    forged_data = record.model_dump(mode="json")
    forged_data["isolating_interval"] = {
        "lower": {"num": "-3", "den": "1"},
        "upper": {"num": "-2", "den": "1"},
        "interval_type": "OPEN",
    }
    forged = RealNumberFieldEmbeddingRecord.model_validate(forged_data)
    binding = _binding(_element(record.embedding.presentation, 0, 1), forged)
    request = NumberFieldRealEmbeddingOrderRequest(left=binding, right=binding)
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "number_field.real_embedding.element_order.compare"
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        operation.run(request)

    error = caught.value.errors()[0]
    assert error["loc"] == ("left", "embedding_record")
    assert error["type"] == (
        "number_field.real_embedding_order.embedding_record_not_recognized"
    )
