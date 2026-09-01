"""Dispatch-boundary tests for finite-dimensional algebra operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.dispatch import parse_operation_input
from jacobian.math.finite_dim_algebras._models import (
    MAX_DIM,
    MAX_STRUCTURE_CONSTANT_ENTRIES,
    CenterRequest,
    StructureConstants,
)


def test_published_dimension_matches_structural_request_bound() -> None:
    """Worst-case valid tensors survive strict dispatch parsing."""
    n = MAX_DIM
    residue = 250
    inner = [residue] * n
    row = [inner] * n
    payload = {
        "algebra": {
            "dimension": n,
            "field_order": 251,
            "multiplication": [row] * n,
        }
    }
    encoded = encode_strict_json(payload)
    assert len(encoded) <= CanonicalLimits().max_input_bytes
    parsed = parse_operation_input(CenterRequest, payload)
    assert parsed.algebra.dimension == n
    schema = StructureConstants.model_json_schema()
    assert schema["properties"]["dimension"]["maximum"] == n
    assert MAX_DIM**3 == MAX_STRUCTURE_CONSTANT_ENTRIES

    oversized_inner = [residue] * (n + 1)
    oversized_row = [oversized_inner] * (n + 1)
    oversized = {
        "algebra": {
            "dimension": n + 1,
            "field_order": 251,
            "multiplication": [oversized_row] * (n + 1),
        }
    }
    with pytest.raises(ValidationError):
        parse_operation_input(CenterRequest, oversized)
