from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import pytest
from pydantic import StrictInt, StringConstraints, ValidationError

from jacobian.contracts.graph_composition import GraphEnumerationRequest
from jacobian.contracts.results import ContractModel
from jacobian.operation_adapters import parse_operation_input


class _Label(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class _TupleRequest(ContractModel):
    labels: tuple[Annotated[str, StringConstraints(strict=True)], ...]
    limit: StrictInt


class _EnumRequest(ContractModel):
    side: _Label
    pairs: tuple[tuple[str, str], ...] | None = None


def test_parse_operation_input_accepts_json_arrays_for_constrained_tuples() -> None:
    parsed = parse_operation_input(
        _TupleRequest,
        {"labels": ["left", "right"], "limit": 2},
    )

    assert parsed.labels == ("left", "right")
    assert isinstance(parsed.labels, tuple)


def test_parse_operation_input_rejects_numeric_strings_for_integers() -> None:
    with pytest.raises(ValidationError, match="int_type"):
        parse_operation_input(GraphEnumerationRequest, {"order": "3"})


def test_tuple_normalization_does_not_enable_scalar_coercion() -> None:
    with pytest.raises(ValidationError):
        parse_operation_input(
            _TupleRequest,
            {"labels": ["left"], "limit": "1"},
        )


def test_parse_operation_input_accepts_json_enums_and_optional_tuples() -> None:
    parsed = parse_operation_input(
        _EnumRequest,
        {"side": "left", "pairs": [["a", "b"]]},
    )

    assert parsed.side is _Label.LEFT
    assert parsed.pairs == (("a", "b"),)
