"""Wire admission classification for finite delta-matroid relabelling."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation


def test_oversized_target_labels_keep_resource_classification_on_wire() -> None:
    payload = {
        "delta_matroid": {"ground": ["a", "b"], "feasible": [[]]},
        "target_ground": ["x" * 1_025, "y" * 1_024],
        "target_to_source": [0, 1],
    }

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        invoke_operation("delta_matroid.relabel.compute", payload, Catalog.open())

    assert (
        exc_info.value.errors()[0]["type"]
        == "delta_matroid.relabel_target_bytes"
    )
