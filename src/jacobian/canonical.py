"""Bounded canonical JSON for mathematical artifact identity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NoReturn

import rfc8785

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_DECIMAL_CHUNK_BASE = 1_000_000_000
_DECIMAL_CHUNK_DIGITS = 9


def sha256_digest(data: bytes) -> str:
    """Return the canonical prefixed SHA-256 digest for immutable bytes."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


class CanonicalizationError(ValueError):
    """The input cannot be represented by Jacobian's canonical JSON profile."""


def parse_canonical_integer(value: str) -> int:
    """Parse a canonical decimal integer without Python's string-digit limit."""

    if _INTEGER.fullmatch(value) is None:
        raise CanonicalizationError(
            "rational components must be canonical decimal integers"
        )
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    first_chunk_digits = len(digits) % _DECIMAL_CHUNK_DIGITS or _DECIMAL_CHUNK_DIGITS
    parsed = int(digits[:first_chunk_digits])
    for offset in range(first_chunk_digits, len(digits), _DECIMAL_CHUNK_DIGITS):
        chunk = digits[offset : offset + _DECIMAL_CHUNK_DIGITS]
        parsed = parsed * _DECIMAL_CHUNK_BASE + int(chunk)
    return -parsed if negative else parsed


def format_canonical_integer(value: int) -> str:
    """Format an integer without Python's string-digit limit."""

    if value == 0:
        return "0"
    negative = value < 0
    remaining = abs(value)
    chunks: list[str] = []
    while remaining:
        remaining, chunk = divmod(remaining, _DECIMAL_CHUNK_BASE)
        chunks.append(str(chunk).zfill(_DECIMAL_CHUNK_DIGITS))
    formatted = "".join(reversed(chunks)).lstrip("0")
    return f"-{formatted}" if negative else formatted


@dataclass(frozen=True, slots=True)
class CanonicalLimits:
    max_input_bytes: int = 10 * 1024 * 1024
    max_output_bytes: int = 10 * 1024 * 1024
    max_depth: int = 64
    max_integer_digits: int = 32_768


def _reject_float(_value: str) -> NoReturn:
    raise CanonicalizationError("JSON floating-point numbers are not allowed")


def _reject_constant(_value: str) -> NoReturn:
    raise CanonicalizationError("non-finite JSON value is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError(
                f"duplicate JSON object key: {key!r}. Remove or rename one occurrence."
            )
        result[key] = value
    return result


def loads_strict_json(
    value: str | bytes | bytearray,
    *,
    limits: CanonicalLimits | None = None,
) -> Any:
    """Parse JSON while rejecting duplicate keys, floats, and oversized input."""

    active_limits = limits or CanonicalLimits()
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(raw) > active_limits.max_input_bytes:
        raise CanonicalizationError("JSON input exceeds the configured size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalizationError("JSON input must be valid UTF-8") from exc
    if text.startswith("\ufeff"):
        raise CanonicalizationError("JSON input must not contain a UTF-8 BOM")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CanonicalizationError("invalid or excessively nested JSON") from exc


def _normalize_rational(
    value: dict[str, Any],
    *,
    limits: CanonicalLimits,
) -> dict[str, str] | None:
    if set(value) != {"num", "den"}:
        return None
    numerator = value["num"]
    denominator = value["den"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        return None
    if not _INTEGER.fullmatch(numerator) or not _INTEGER.fullmatch(denominator):
        raise CanonicalizationError(
            "rational components must be canonical decimal integers"
        )
    if (
        len(numerator.lstrip("-")) > limits.max_integer_digits
        or len(denominator.lstrip("-")) > limits.max_integer_digits
    ):
        raise CanonicalizationError("rational component exceeds the digit limit")
    den = parse_canonical_integer(denominator)
    if den == 0:
        raise CanonicalizationError("rational denominator cannot be zero")
    fraction = Fraction(parse_canonical_integer(numerator), den)
    return {
        "num": format_canonical_integer(fraction.numerator),
        "den": format_canonical_integer(fraction.denominator),
    }


def _normalize(value: Any, *, limits: CanonicalLimits, depth: int) -> Any:
    if depth > limits.max_depth:
        raise CanonicalizationError("JSON nesting exceeds the configured depth limit")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise CanonicalizationError("JSON floating-point numbers are not allowed")
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise CanonicalizationError(
                "JSON integers outside the interoperable range must be encoded as strings"
            )
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize(item, limits=limits, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return _normalize_object(value, limits=limits, depth=depth)
    raise CanonicalizationError("unsupported JSON value type")


def _validate_json_value(value: Any, *, limits: CanonicalLimits, depth: int) -> None:
    """Validate bounded interoperable JSON without changing semantic values."""

    if depth > limits.max_depth:
        raise CanonicalizationError("JSON nesting exceeds the configured depth limit")
    if isinstance(value, float):
        raise CanonicalizationError("JSON floating-point numbers are not allowed")
    if isinstance(value, int):
        _validate_json_integer(value)
        return
    if isinstance(value, list):
        _validate_json_items(value, limits=limits, depth=depth)
        return
    if isinstance(value, dict):
        _validate_json_object(value, limits=limits, depth=depth)
        return
    if value is None or isinstance(value, str):
        return
    raise CanonicalizationError("unsupported JSON value type")


def _validate_json_integer(value: int) -> None:
    if abs(value) > _MAX_SAFE_JSON_INTEGER:
        raise CanonicalizationError(
            "JSON integers outside the interoperable range must be encoded as strings"
        )


def _validate_json_items(
    values: list[Any], *, limits: CanonicalLimits, depth: int
) -> None:
    for value in values:
        _validate_json_value(value, limits=limits, depth=depth + 1)


def _validate_json_object(
    value: dict[Any, Any], *, limits: CanonicalLimits, depth: int
) -> None:
    for key, nested in value.items():
        if not isinstance(key, str):
            raise CanonicalizationError("JSON object keys must be strings")
        _validate_json_value(nested, limits=limits, depth=depth + 1)


def encode_strict_json(
    value: Any,
    *,
    limits: CanonicalLimits | None = None,
) -> bytes:
    """Encode bounded JSON deterministically without semantic normalization."""

    active_limits = limits or CanonicalLimits()
    _validate_json_value(value, limits=active_limits, depth=0)
    try:
        encoded = rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, RecursionError) as exc:
        raise CanonicalizationError("value cannot be encoded as strict JSON") from exc
    if len(encoded) > active_limits.max_output_bytes:
        raise CanonicalizationError("JSON exceeds the configured size limit")
    return encoded


def _normalize_object(
    value: dict[str, Any], *, limits: CanonicalLimits, depth: int
) -> dict[str, Any]:
    rational = _normalize_rational(value, limits=limits)
    if rational is not None:
        return rational
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise CanonicalizationError("JSON object keys must be strings")
        key = unicodedata.normalize("NFC", raw_key)
        if key in result:
            raise CanonicalizationError(
                "object keys collide after Unicode normalization"
            )
        result[key] = _normalize(
            raw_value,
            limits=limits,
            depth=depth + 1,
        )
    return result


def canonicalize_json(
    value: Any,
    *,
    limits: CanonicalLimits | None = None,
) -> bytes:
    """Normalize exact JSON and encode it using RFC 8785 key ordering."""

    active_limits = limits or CanonicalLimits()
    parsed = (
        loads_strict_json(value, limits=active_limits)
        if isinstance(value, (str, bytes, bytearray))
        else value
    )
    normalized = _normalize(parsed, limits=active_limits, depth=0)
    try:
        encoded = rfc8785.dumps(normalized)
    except (rfc8785.CanonicalizationError, RecursionError) as exc:
        raise CanonicalizationError("value cannot be canonically encoded") from exc
    if len(encoded) > active_limits.max_output_bytes:
        raise CanonicalizationError("canonical JSON exceeds the configured size limit")
    return encoded
