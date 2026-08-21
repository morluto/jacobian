"""Narrow strict-model primitive shared by unrelated wire owners."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel"]
