"""Narrow strict-model primitive shared by unrelated wire owners."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, model_validator


def _unwrap_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _unwrap_annotation(args[0]) if args else annotation
    return annotation


def _tuple_annotation(annotation: Any) -> Any | None:
    annotation = _unwrap_annotation(annotation)
    origin = get_origin(annotation)
    if origin is tuple:
        return annotation
    if origin in {Union, UnionType}:
        tuple_args = [
            arg
            for arg in get_args(annotation)
            if arg is not type(None) and get_origin(_unwrap_annotation(arg)) is tuple
        ]
        if len(tuple_args) == 1:
            return tuple_args[0]
    return None


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def _dataclass_annotation(annotation: Any) -> type[Any] | None:
    annotation = _unwrap_annotation(annotation)
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        dataclass_args = [
            arg
            for arg in get_args(annotation)
            if arg is not type(None) and _dataclass_annotation(arg) is not None
        ]
        if len(dataclass_args) == 1:
            return _dataclass_annotation(dataclass_args[0])
        return None
    if (
        isinstance(annotation, type)
        and is_dataclass(annotation)
        and not issubclass(annotation, BaseModel)
    ):
        return annotation
    return None


def _dataclass_from_json(cls: type[Any], value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        return value
    allowed = {field.name for field in dataclass_fields(cls)}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"unexpected dataclass fields: {', '.join(unexpected)}")
    prepared = {
        name: _lists_to_tuples(item) if isinstance(item, list) else item
        for name, item in value.items()
    }
    return cls(**prepared)


class StrictModel(BaseModel):
    """Closed, immutable model with JSON tuple/dataclass decoding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def accept_json_wire_shapes(cls, data: Any) -> Any:
        """Decode JSON arrays and objects into declared tuple and dataclass fields."""

        if isinstance(data, cls) or not isinstance(data, dict):
            return data
        coerced = dict(data)
        for name, field in cls.model_fields.items():
            if name not in coerced:
                continue
            value = coerced[name]
            dataclass_type = _dataclass_annotation(field.annotation)
            if dataclass_type is not None:
                coerced[name] = _dataclass_from_json(dataclass_type, value)
                continue
            if _tuple_annotation(field.annotation) is None or not isinstance(
                value, list
            ):
                continue
            coerced[name] = _lists_to_tuples(value)
        return coerced
