from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import StringConstraints, ValidationError

from jacobian.capability_adapters import parse_capability_input
from jacobian.contracts.results import ContractModel


class _TupleRequest(ContractModel):
    labels: tuple[Annotated[str, StringConstraints(strict=True)], ...]
    limit: int


def test_parse_capability_input_accepts_json_arrays_for_constrained_tuples() -> None:
    parsed = parse_capability_input(
        _TupleRequest,
        {"labels": ["left", "right"], "limit": 2},
    )

    assert parsed.labels == ("left", "right")
    assert isinstance(parsed.labels, tuple)


def test_tuple_normalization_does_not_enable_scalar_coercion() -> None:
    with pytest.raises(ValidationError):
        parse_capability_input(
            _TupleRequest,
            {"labels": ["left"], "limit": "1"},
        )
