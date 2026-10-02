"""Cheap relation encodings are native helpers rather than public presentations."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest

_REMOVED = "number_theory.numerical_semigroup.presentation_binomials.compute"
_PRESENTATION = "number_theory.numerical_semigroup.minimal_presentation.compute"


@pytest.mark.parametrize(
    "need", (_REMOVED, "numerical semigroup presentation binomials")
)
def test_relation_projection_is_absent_from_catalog_and_discovery(need: str) -> None:
    catalog = Catalog.open()
    assert catalog.operation(_REMOVED) is None
    assert _REMOVED not in {item.operation_id for item in catalog.snapshot().operations}
    discovered = catalog.match(OperationMatchRequest(need=need))
    assert _REMOVED not in {item.operation_id for item in discovered.matches}


def test_actual_minimal_presentation_remains_published_and_discoverable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_PRESENTATION)
    assert operation is not None
    assert operation.examples[0].input == {"generators": ["3", "5"]}
    assert set(operation.request_type.model_fields) == {"generators"}
    assert set(operation.result_type.model_fields) == {
        "minimal_generators",
        "betti_elements",
        "relations",
    }
    discovered = catalog.match(
        OperationMatchRequest(need="numerical semigroup minimal presentation")
    )
    assert _PRESENTATION in {item.operation_id for item in discovered.matches}
