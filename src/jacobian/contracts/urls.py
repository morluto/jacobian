"""Canonical URL primitives shared by public contract modules."""

from __future__ import annotations

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

_HTTP_URL = TypeAdapter(AnyHttpUrl)


def normalize_http_url(value: object, *, label: str) -> str:
    """Return one canonical HTTP(S) URL before downstream bounds apply."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a valid HTTP(S) URL")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    try:
        return str(_HTTP_URL.validate_python(value))
    except ValidationError as exc:
        raise ValueError(f"{label} must be a valid HTTP(S) URL") from exc
