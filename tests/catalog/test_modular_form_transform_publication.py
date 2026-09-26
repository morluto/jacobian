"""Transform declarations remain published by the owner-local catalog manifest."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.modular_forms.transform_tools import (
    TOOLS as TRANSFORM_TOOLS,
)


def test_modular_form_transforms_are_live_catalog_operations() -> None:
    catalog = Catalog.open()
    assert {tool.operation_id for tool in TRANSFORM_TOOLS} == {
        "modular_form.named.q_expansion.compute",
        "modular_form.space.sturm_bound.compute",
        "modular_form.formal_q_series.u_operator.compute",
        "modular_form.formal_q_series.v_operator.compute",
    }
    for declaration in TRANSFORM_TOOLS:
        assert catalog.operation(declaration.operation_id) is declaration
