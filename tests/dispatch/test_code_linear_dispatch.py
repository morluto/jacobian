"""Dispatch boundaries for linear-code operations."""

from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_serialized_dual_parity_check_dispatches_into_syndrome() -> None:
    catalog = Catalog.open()
    dual = invoke_operation(
        "code.linear.dual.compute",
        {
            "encoder": {
                "field_order": 2,
                "message_axis": ["m0", "m1"],
                "coordinate_axis": ["left", "right"],
                "generator_matrix": [[1, 0], [0, 1]],
            }
        },
        catalog,
    )
    parity_check = dual.output["parity_check"]
    syndrome = invoke_operation(
        "code.linear.syndrome.compute",
        {
            "parity_check": parity_check,
            "coordinate_axis": parity_check["coordinate_axis"],
            "word": [1, 1],
        },
        catalog,
    )

    assert syndrome.output["syndrome"] == []
    assert syndrome.output["is_member"] is True
