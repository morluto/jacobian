"""Typed requests for character-valued degeneracy maps."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


class ModularCharacterVDegeneracyRequest(StrictModel):
    """Apply V_d using an explicitly represented character-space inclusion."""

    form: ModularFormCoordinates
    target_space: ModularFormSpace


__all__ = ["ModularCharacterVDegeneracyRequest"]
