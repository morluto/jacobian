"""Published Clifford transport admission uses native structural units."""

from jacobian.math.quantum.stabilizer_clifford._models import (
    StabilizerCliffordTransportRequest,
)


def test_transport_schema_description_agrees_with_result_cell_limit() -> None:
    schema = StabilizerCliffordTransportRequest.model_json_schema()
    description = schema["description"]
    limit = schema["admission_limits"]["max_result_cells"]
    assert f"{limit:,} structural result cells" in description
    assert "result bytes" not in description
    assert "serialized result" not in description
