from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.domain_operations import DurableOperationOutput
from jacobian.contracts.number_theory import IntegerValueResult

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64


def test_complete_materialized_preview_requires_a_value() -> None:
    output_model = DurableOperationOutput[IntegerValueResult]

    with pytest.raises(ValidationError, match="requires a preview"):
        output_model(
            input_uri=_ARTIFACT_URI,
            result_uri=_ARTIFACT_URI,
            preview_complete=True,
            backend_version="test",
        )

    partial = output_model(
        input_uri=_ARTIFACT_URI,
        result_uri=_ARTIFACT_URI,
        preview=IntegerValueResult(value="1"),
        preview_complete=False,
        backend_version="test",
    )
    assert partial.preview is not None
