"""Passive base contract shared by domain-owned public values."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Closed, immutable base for public semantic and wire values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
