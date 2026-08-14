"""Shared scalar types used by language-neutral wire contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$", strict=True),
]
