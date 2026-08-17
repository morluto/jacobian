"""Private digest wire primitive shared by unrelated mathematical owners."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$", strict=True),
]
