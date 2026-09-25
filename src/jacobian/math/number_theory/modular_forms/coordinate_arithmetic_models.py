"""Requests for exact arithmetic on canonical modular-form coordinates."""

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.values import ModularFormCoordinates


class ModularFormCoordinatesAddRequest(StrictModel):
    """Add two forms expressed in one exact space and canonical basis."""

    left: ModularFormCoordinates
    right: ModularFormCoordinates


__all__ = ["ModularFormCoordinatesAddRequest"]
