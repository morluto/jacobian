"""Passive base contract shared by domain-owned public values."""

from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, model_validator


def _unwrap_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _unwrap_annotation(args[0]) if args else annotation
    return annotation


def _is_tuple_annotation(annotation: Any) -> bool:
    return get_origin(_unwrap_annotation(annotation)) is tuple


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


class ContractModel(BaseModel):
    """Closed, immutable base for public semantic and wire values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def accept_json_arrays_for_tuple_fields(cls, data: Any) -> Any:
        """Decode JSON arrays into declared tuple fields once."""

        if isinstance(data, cls) or not isinstance(data, dict):
            return data
        coerced = dict(data)
        for name, field in cls.model_fields.items():
            if name not in coerced:
                continue
            value = coerced[name]
            if _is_tuple_annotation(field.annotation) and isinstance(value, list):
                coerced[name] = _lists_to_tuples(value)
        return coerced
