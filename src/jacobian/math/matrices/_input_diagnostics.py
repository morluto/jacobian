"""Owner-local diagnostics for ambiguous matrix input representations.

Successful union parsing is unchanged. Explicit branch tags identify only the
provenance introduced by the owning union; failed alternatives stay in bounded
messages rather than caller field paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any

from pydantic import ValidationError, ValidationInfo, ValidatorFunctionWrapHandler
from pydantic_core import InitErrorDetails, PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
    RationalMatrix,
    SparseRationalMatrix,
)

_MATRIX_STORAGE_BRANCH_PREFIXES = {
    "dense": "Dense matrix: ",
    "sparse": "Sparse matrix: ",
}
_MAX_MATRIX_ERROR_MESSAGE_LENGTH = 4096


def _matrix_storage_branch(
    value: object, *, maximum_dense_axis: int = MAX_RATIONAL_MATRIX_ORDER
) -> str | None:
    """Recognize only bounded, unambiguous storage, never choose a parser."""

    if type(value) is RationalMatrix:
        return "dense"
    if type(value) is SparseRationalMatrix:
        return "sparse"
    # Exact keys avoid caller-defined hashing/equality during the lookup. Native
    # subclasses and unusual containers provide no evidence for dropping errors.
    if (
        type(value) is not dict
        or len(value) > 4
        or any(type(key) is not str for key in value)
    ):
        return None
    entries = value.get("entries")
    if (
        (type(entries) is not list and type(entries) is not tuple)
        or not entries
        or len(entries) > MAX_SPARSE_RATIONAL_MATRIX_NONZEROS
    ):
        return None
    if all(type(entry) is list or type(entry) is tuple for entry in entries):
        if len(entries) > maximum_dense_axis:
            return None
        inspected_scalars = 0
        for row in entries:
            if len(row) > maximum_dense_axis:
                return None
            inspected_scalars += len(row)
            if inspected_scalars > MAX_RATIONAL_MATRIX_ORDER**2:
                return None
            for scalar in row:
                if type(scalar) is CanonicalRational:
                    continue
                if (
                    type(scalar) is not dict
                    or len(scalar) > 3
                    or any(type(key) is not str for key in scalar)
                    or "row" in scalar
                    or "column" in scalar
                ):
                    return None
        return "dense"
    # A mapping can instead be a rational whose enclosing dense row was omitted.
    # Require coordinate evidence, including for incomplete sparse entries.
    for entry in entries:
        if (
            type(entry) is not dict
            or len(entry) > 3
            or any(
                type(key) is not str or key not in ("row", "column", "value")
                for key in entry
            )
            or ("row" not in entry and "column" not in entry)
        ):
            return None
    return "sparse"


def _matrix_message_context(message: str, context: object) -> bool:
    """Avoid re-interpreting a rendered message or touching custom context."""

    if len(message) > _MAX_MATRIX_ERROR_MESSAGE_LENGTH:
        return False
    if context is None:
        return True
    if type(context) is not dict or len(context) > 8:
        return False
    for key, value in context.items():
        if (
            type(key) is not str
            or len(key) > 128
            or "{" + key + "}" in message
            or (
                type(value) is not str
                and type(value) is not int
                and type(value) is not float
                and type(value) is not bool
                and value is not None
            )
        ):
            return False
        if (
            (type(value) is str and len(value) > 1024)
            or (type(value) is int and value.bit_length() > 1024)
            or (type(value) is float and not isfinite(value))
        ):
            return False
    return True


def _matrix_union_validation(
    value: Any,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
    *,
    branch_for_value: Callable[[object], str | None] = _matrix_storage_branch,
    branch_prefixes: Mapping[str, str] = _MATRIX_STORAGE_BRANCH_PREFIXES,
) -> Any:
    """Project only the provenance introduced by this field's tagged union.

    Successful union parsing, including its carrier preference, is untouched.
    Ambiguous failures keep both alternatives, named in messages instead of
    caller paths. A future custom context that cannot be reconstructed exactly
    leaves the original error intact. Prefixed built-in errors retain their
    code/context/input but, as custom errors, have no Pydantic documentation URL.
    """

    try:
        return handler(value)
    except ValidationError as exc:
        branch = branch_for_value(value)
        records: list[InitErrorDetails] = []
        tags: list[str] = []
        for error in exc.errors():
            location = error["loc"]
            if (
                not location
                or not isinstance(location[0], str)
                or location[0] not in branch_prefixes
            ):
                raise
            tag = location[0]
            if branch is not None and tag != branch:
                continue
            code = error["type"]
            record: InitErrorDetails = {
                "type": code,
                "loc": location[1:],
                "input": error["input"],
            }
            context = error.get("ctx")
            if context is not None:
                record["ctx"] = context
            if branch is None and not _matrix_message_context(error["msg"], context):
                raise
            # Pydantic supplies documentation URLs only for native errors.
            # A custom error may reuse a native code with a different message.
            if "url" not in error:
                message = error["msg"]
                if not _matrix_message_context(message, context):
                    raise
                record["type"] = PydanticCustomError(code, message, context)
            records.append(record)
            tags.append(tag)
        if not records:
            raise
        projected = ValidationError.from_exception_data(
            exc.title, records, input_type=info.mode
        )
        if branch is None:
            # Render native codes in the outer mode before adding a prefix:
            # e.g. model_type has distinct JSON and Python messages.
            for record, error, tag in zip(
                records, projected.errors(include_url=False), tags, strict=True
            ):
                prefix = branch_prefixes[tag]
                if len(error["msg"]) + len(prefix) > _MAX_MATRIX_ERROR_MESSAGE_LENGTH:
                    raise
                message = prefix + error["msg"]
                context = error.get("ctx")
                if not _matrix_message_context(message, context):
                    raise
                record["type"] = PydanticCustomError(error["type"], message, context)
            projected = ValidationError.from_exception_data(
                exc.title, records, input_type=info.mode
            )
        raise projected from None
